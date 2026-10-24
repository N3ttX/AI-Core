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
    private class Loaded(val spec: ModelSpec, val engine: Closeable, val options: Any, var lastUsedAt: Long)

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
    suspend fun <T> withModel(requestId: String, spec: ModelSpec, clientLabel: String, kind: String, block: (LlamaModel) -> T): T =
        withEngine(requestId, spec, clientLabel, kind) { block(it as LlamaModel) }

    /** Runs [block] with the ONNX package [spec] loaded, on the inference thread. */
    suspend fun <T> withOnnx(requestId: String, spec: ModelSpec, clientLabel: String, kind: String, block: (OnnxModel) -> T): T =
        withEngine(requestId, spec, clientLabel, kind) { block(it as OnnxModel) }

    private suspend fun <T> withEngine(requestId: String, spec: ModelSpec, clientLabel: String, kind: String, block: (Closeable) -> T): T {
        queued.update { it + 1 }
        var dequeued = false
        try {
            return withContext(dispatcher) {
                queued.update { it - 1 }
                dequeued = true
                val l = acquire(spec)
                val r = running[requestId]
                r?.model = l.engine as? LlamaModel
                activeState.value = ActiveRequest(requestId, spec.id, clientLabel, kind, System.currentTimeMillis())
                try {
                    if (r?.cancelled == true) throw kotlinx.coroutines.CancellationException("Cancelled")
                    block(l.engine)
                } finally {
                    r?.model = null
                    activeState.value = null
                    l.lastUsedAt = System.currentTimeMillis()
                    publish()
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
        val engine: Closeable = if (options is OnnxOptions) {
            OnnxModel(store.componentFiles(spec), options)
        } else {
            LlamaModel.load(store.fileOf(spec).absolutePath, options as LoadOptions)
        }
        val l = Loaded(spec, engine, options, System.currentTimeMillis())
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

    private fun publish() {
        loadedState.value = loaded.values.map {
            LoadedModelInfo(it.spec.id, it.spec.name, it.spec.type, store.sizeOnDisk(it.spec), it.lastUsedAt)
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
    }
}
