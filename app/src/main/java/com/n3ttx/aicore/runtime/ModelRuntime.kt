package com.n3ttx.aicore.runtime

import android.app.ActivityManager
import android.content.Context
import android.os.Process
import android.util.Log
import com.n3ttx.aicore.data.ModelSpec
import com.n3ttx.aicore.data.ModelStore
import com.n3ttx.aicore.data.SettingsStore
import com.n3ttx.aicore.engine.LlamaModel
import com.n3ttx.aicore.engine.LoadOptions
import com.n3ttx.aicore.engine.onnx.OnnxModel
import com.n3ttx.aicore.engine.onnx.OnnxOptions
import com.n3ttx.aicore.sdk.AICoreContract
import com.n3ttx.aicore.sdk.AICoreContract.ModelType
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.Closeable
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors

data class LoadedModelInfo(val id: String, val name: String, val type: String, val sizeBytes: Long, val lastUsedAt: Long)

data class ActiveRequest(val requestId: String, val modelId: String, val clientLabel: String, val kind: String, val startedAt: Long)

/**
 * Owns all loaded models. Every inference runs on one dedicated thread, so
 * requests from different apps are queued instead of competing for CPU and
 * memory. Models stay loaded between requests (shared by all apps), are
 * evicted least-recently-used when the memory budget is exceeded, and are
 * unloaded after the configured idle time.
 */
class ModelRuntime(
    private val context: Context,
    private val store: ModelStore,
    private val settings: SettingsStore,
    private val scope: CoroutineScope,
) {
    /** [engine] is a [LlamaModel] or an [OnnxModel]; [options] decides whether a reload is needed. */
    private class Loaded(val spec: ModelSpec, val engine: Closeable, val options: Any, var lastUsedAt: Long, val loadedAt: Long, val sizeBytes: Long)

    private class Running(val uid: Int, val job: Job) {
        @Volatile var model: LlamaModel? = null
        @Volatile var cancelled = false
    }

    private val dispatcher = Executors.newSingleThreadExecutor { r ->
        Thread({
            Process.setThreadPriority(Process.THREAD_PRIORITY_FOREGROUND)
            r.run()
        }, "inference")
    }.asCoroutineDispatcher()

    // Access order = LRU. Only touched on the inference thread.
    private val loaded = LinkedHashMap<String, Loaded>(8, 0.75f, true)
    private val running = ConcurrentHashMap<String, Running>()

    private val loadedState = MutableStateFlow<List<LoadedModelInfo>>(emptyList())
    val loadedModels: StateFlow<List<LoadedModelInfo>> = loadedState.asStateFlow()

    private val activeState = MutableStateFlow<ActiveRequest?>(null)
    val active: StateFlow<ActiveRequest?> = activeState.asStateFlow()

    /** Clears [activeState] a moment after a request, unless the next one has started (see [withEngine]). */
    private var idleJob: Job? = null

    private val lastUsedState = MutableStateFlow<Map<String, Long>>(emptyMap())

    /** When each loaded model last finished a request (not part of [loadedModels], which only changes on load/unload). */
    val lastUsed: StateFlow<Map<String, Long>> = lastUsedState.asStateFlow()

    private val loadingState = MutableStateFlow<String?>(null)

    /** Name of the model being loaded right now, if any. */
    val loading: StateFlow<String?> = loadingState.asStateFlow()

    private val queued = MutableStateFlow(0)
    val queuedCount: StateFlow<Int> = queued.asStateFlow()

    init {
        scope.launch {
            while (isActive) {
                delay(60_000)
                unloadIdle()
            }
        }
    }

    fun newRequestId(): String = UUID.randomUUID().toString()

    /** Runs [block] as a cancellable request on behalf of [uid]. */
    fun submit(requestId: String, uid: Int, block: suspend () -> Unit) {
        val job = scope.launch(start = kotlinx.coroutines.CoroutineStart.LAZY) {
            try {
                block()
            } finally {
                running.remove(requestId)
            }
        }
        running[requestId] = Running(uid, job)
        job.start()
    }

    /** Cancels a request. Apps can only cancel their own requests. */
    fun cancel(requestId: String, callerUid: Int) {
        val r = running[requestId] ?: return
        if (r.uid != callerUid && callerUid != Process.myUid()) return
        r.cancelled = true
        r.model?.cancel()
        r.job.cancel()
    }

    fun isCancelled(requestId: String): Boolean = running[requestId]?.cancelled ?: false

    /**
     * Runs [block] with [spec] loaded, on the inference thread.
     */
    suspend fun <T> withModel(
        requestId: String, spec: ModelSpec, clientLabel: String, kind: String, background: Boolean = false, block: (LlamaModel) -> T,
    ): T = withEngine(requestId, spec, clientLabel, kind, background) { block(it as LlamaModel) }

    /** Runs [block] with the ONNX package [spec] loaded, on the inference thread. */
    suspend fun <T> withOnnx(
        requestId: String, spec: ModelSpec, clientLabel: String, kind: String, background: Boolean = false, block: (OnnxModel) -> T,
    ): T = withEngine(requestId, spec, clientLabel, kind, background) { block(it as OnnxModel) }

    /**
     * Interactive requests waiting or running. Background requests (an app indexing its photos) wait while there are any:
     * a chat or a search goes first, the batch job continues after it, between two of its items.
     */
    private val interactive = MutableStateFlow(0)

    private val backgroundWaiting = MutableStateFlow(0)

    /** Background requests waiting for interactive ones, for the debug notification. */
    val waitingBackground: StateFlow<Int> = backgroundWaiting.asStateFlow()

    private suspend fun <T> withEngine(
        requestId: String, spec: ModelSpec, clientLabel: String, kind: String, background: Boolean, block: (Closeable) -> T,
    ): T {
        if (background) {
            if (interactive.value > 0) {
                backgroundWaiting.update { it + 1 }
                try {
                    interactive.first { it == 0 }
                } finally {
                    backgroundWaiting.update { it - 1 }
                }
            }
        } else {
            interactive.update { it + 1 }
        }
        try {
            return runQueued(requestId, spec, clientLabel, kind, block)
        } finally {
            if (!background) interactive.update { it - 1 }
        }
    }

    private suspend fun <T> runQueued(requestId: String, spec: ModelSpec, clientLabel: String, kind: String, block: (Closeable) -> T): T {
        queued.update { it + 1 }
        var dequeued = false
        try {
            return withContext(dispatcher) {
                queued.update { it - 1 }
                dequeued = true
                val l = acquire(spec)
                val r = running[requestId]
                r?.model = l.engine as? LlamaModel
                idleJob?.cancel()
                activeState.value = ActiveRequest(requestId, spec.id, clientLabel, kind, System.currentTimeMillis())
                try {
                    if (r?.cancelled == true) throw kotlinx.coroutines.CancellationException("Cancelled")
                    block(l.engine)
                } finally {
                    r?.model = null
                    l.lastUsedAt = System.currentTimeMillis()
                    lastUsedState.update { it + (spec.id to l.lastUsedAt) }
                    // Batch jobs (e.g. a gallery indexing one photo per request) would switch "Working" to idle and back for
                    // every request: the status only turns idle when no request follows within a moment.
                    idleJob = scope.launch {
                        delay(IDLE_DISPLAY_DELAY_MS)
                        val current = activeState.value
                        if (current?.requestId == requestId) activeState.compareAndSet(current, null)
                    }
                }
            }
        } finally {
            if (!dequeued) queued.update { it - 1 }
        }
    }

    private fun optionsFor(spec: ModelSpec): Any {
        val s = settings.current
        if (spec.engine == AICoreContract.Engine.ONNX) return OnnxOptions(threads = s.threads, xnnpack = s.useXnnpack)
        val embeddings = spec.type == ModelType.EMBEDDING
        val ctx = if (embeddings) 2048 else s.contextLength
        return LoadOptions(contextLength = ctx, batchSize = 512, threads = s.threads, embeddings = embeddings)
    }

    private fun acquire(spec: ModelSpec): Loaded {
        val options = optionsFor(spec)
        loaded[spec.id]?.let { existing ->
            if (existing.options == options) return existing
            unloadNow(spec.id) // settings changed, reload
        }
        makeRoomFor(spec)
        Log.i(TAG, "Loading ${spec.id} with $options")
        loadingState.value = spec.name
        val engine: Closeable = try {
            if (options is OnnxOptions) {
                OnnxModel(store.componentFiles(spec), options)
            } else {
                LlamaModel.load(store.ggufOf(spec).absolutePath, options as LoadOptions)
            }
        } finally {
            loadingState.value = null
        }
        val now = System.currentTimeMillis()
        val l = Loaded(spec, engine, options, now, now, store.sizeOnDisk(spec))
        loaded[spec.id] = l
        publish()
        return l
    }

    private fun estimateBytes(spec: ModelSpec): Long {
        val file = store.sizeOnDisk(spec)
        // Weights plus a rough allowance for KV cache and compute buffers.
        return file + file / 5 + 256L * 1024 * 1024
    }

    private fun makeRoomFor(spec: ModelSpec) {
        val budget = settings.current.memoryBudgetMb.toLong() * 1024 * 1024
        val need = estimateBytes(spec)
        while (loaded.isNotEmpty() && loaded.values.sumOf { estimateBytes(it.spec) } + need > budget) {
            unloadNow(loaded.keys.first())
        }
        val am = context.getSystemService(ActivityManager::class.java)
        val info = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        if (info.availMem < need && loaded.isNotEmpty()) {
            Log.i(TAG, "Low memory (${info.availMem shr 20} MB free), unloading all models")
            loaded.keys.toList().forEach { unloadNow(it) }
        }
    }

    private fun unloadNow(id: String) {
        val l = loaded.remove(id) ?: return
        Log.i(TAG, "Unloading $id")
        l.engine.close()
        publish()
    }

    /**
     * Publishes the loaded models when one is loaded or unloaded (not after every request), in the order they were loaded:
     * the LRU order of [loaded] changes with every request, and the list in the UI would jump around.
     */
    private fun publish() {
        loadedState.value = loaded.values.sortedBy { it.loadedAt }.map {
            LoadedModelInfo(it.spec.id, it.spec.name, it.spec.type, it.sizeBytes, it.lastUsedAt)
        }
    }

    private suspend fun unloadIdle() = withContext(dispatcher) {
        val limit = settings.current.idleUnloadMinutes * 60_000L
        if (limit <= 0) return@withContext
        val now = System.currentTimeMillis()
        loaded.values.filter { now - it.lastUsedAt > limit }.map { it.spec.id }.forEach { unloadNow(it) }
    }

    suspend fun unload(id: String) = withContext(dispatcher) { unloadNow(id) }

    suspend fun unloadAll() = withContext(dispatcher) { loaded.keys.toList().forEach { unloadNow(it) } }

    companion object {
        private const val TAG = "ModelRuntime"
        private const val IDLE_DISPLAY_DELAY_MS = 1500L
    }
}
