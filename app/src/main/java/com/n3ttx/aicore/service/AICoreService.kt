package com.n3ttx.aicore.service

import android.app.Service
import android.content.Intent
import android.os.Binder
import android.os.Bundle
import android.os.IBinder
import android.os.Process
import android.os.RemoteException
import android.util.Log
import com.n3ttx.aicore.AppGraph
import com.n3ttx.aicore.BuildConfig
import com.n3ttx.aicore.data.ModelRequest
import com.n3ttx.aicore.data.ModelSpec
import com.n3ttx.aicore.engine.EngineException
import com.n3ttx.aicore.engine.Grammars
import com.n3ttx.aicore.engine.LlamaModel
import com.n3ttx.aicore.engine.SamplingParams
import com.n3ttx.aicore.engine.StopReason
import com.n3ttx.aicore.graph
import com.n3ttx.aicore.runtime.OutputFilter
import com.n3ttx.aicore.sdk.AICoreContract
import com.n3ttx.aicore.sdk.AICoreContract.AccessState
import com.n3ttx.aicore.sdk.AICoreContract.Error
import com.n3ttx.aicore.sdk.AICoreContract.Key
import com.n3ttx.aicore.sdk.AICoreContract.ModelRequestStatus
import com.n3ttx.aicore.sdk.AICoreContract.ModelType
import com.n3ttx.aicore.sdk.IAICoreService
import com.n3ttx.aicore.sdk.IEmbeddingCallback
import com.n3ttx.aicore.sdk.IGenerationCallback
import kotlinx.coroutines.CancellationException

/** The bound service other apps talk to. See IAICoreService.aidl. */
class AICoreService : Service() {
    private lateinit var g: AppGraph

    override fun onCreate() {
        super.onCreate()
        g = graph
    }

    override fun onBind(intent: Intent): IBinder = binder

    private data class Caller(val uid: Int, val packageName: String, val label: String)

    private sealed interface Auth {
        data class Ok(val caller: Caller) : Auth
        data class Rejected(val code: Int, val message: String, val state: String) : Auth
    }

    private fun authorize(): Auth {
        val uid = Binder.getCallingUid()
        if (uid == Process.myUid()) return Auth.Ok(Caller(uid, packageName, "AI Core Hub"))
        val packages = packageManager.getPackagesForUid(uid).orEmpty()
        if (packages.isEmpty()) return Auth.Rejected(Error.NOT_AUTHORIZED, "Unknown caller", AccessState.DENIED)
        packages.firstOrNull { g.access.stateOf(it) == AccessState.APPROVED }?.let {
            g.access.recordUse(it)
            return Auth.Ok(Caller(uid, it, g.access.labelOf(it)))
        }
        if (packages.any { g.access.stateOf(it) == AccessState.DENIED }) {
            return Auth.Rejected(Error.NOT_AUTHORIZED, "Access was denied by the user", AccessState.DENIED)
        }
        val pkg = packages.first()
        if (g.access.markPending(pkg)) g.notifier.accessRequested(pkg, g.access.labelOf(pkg))
        return Auth.Rejected(Error.PENDING_APPROVAL, "Waiting for the user to approve this app in AI Core Hub", AccessState.PENDING)
    }

    private fun error(code: Int, message: String) = Bundle().apply {
        putInt(Key.ERROR_CODE, code)
        putString(Key.ERROR_MESSAGE, message)
    }

    private fun resolveModel(id: String?, type: String): ModelSpec? =
        if (id.isNullOrBlank()) g.defaultModel(type) else g.store.get(id)

    private fun modelBundle(spec: ModelSpec, isDefault: Boolean) = Bundle().apply {
        putString(Key.ID, spec.id)
        putString(Key.NAME, spec.name)
        putString(Key.TYPE, spec.type)
        putLong(Key.SIZE_BYTES, spec.sizeBytes)
        putInt(Key.CONTEXT_LENGTH, spec.contextLength)
        putStringArray(Key.LANGUAGES, spec.languages.toTypedArray())
        putBoolean(Key.IS_DEFAULT, isDefault)
        spec.queryPrefix?.let { putString(Key.QUERY_PREFIX, it) }
        spec.documentPrefix?.let { putString(Key.DOCUMENT_PREFIX, it) }
    }

    /** Builds the prompt for a generation request on [model]. */
    private fun buildPrompt(request: Bundle, spec: ModelSpec, model: LlamaModel): String {
        request.getString(Key.PROMPT)?.let { return it }
        @Suppress("DEPRECATION")
        val messages = request.getParcelableArrayList<Bundle>(Key.MESSAGES).orEmpty()
            .map { it.getString(Key.ROLE).orEmpty() to it.getString(Key.CONTENT).orEmpty() }
        require(messages.isNotEmpty()) { "Request needs messages or a prompt" }
        return model.applyChatTemplate(messages, spec.chatTemplate) + spec.assistantPrefix.orEmpty()
    }

    private fun samplingOf(request: Bundle): SamplingParams {
        val grammar = request.getString(Key.GRAMMAR)
            ?: if (request.getBoolean(Key.JSON_MODE)) Grammars.JSON else null
        return SamplingParams(
            maxTokens = request.getInt(Key.MAX_TOKENS, 512).coerceIn(1, 32_768),
            temperature = request.getFloat(Key.TEMPERATURE, 0.7f).coerceIn(0f, 5f),
            topK = request.getInt(Key.TOP_K, 40).coerceIn(0, 1000),
            topP = request.getFloat(Key.TOP_P, 0.95f).coerceIn(0f, 1f),
            minP = request.getFloat(Key.MIN_P, 0.05f).coerceIn(0f, 1f),
            repeatPenalty = request.getFloat(Key.REPEAT_PENALTY, 1.1f).coerceIn(0f, 3f),
            seed = request.getInt(Key.SEED, -1),
            grammar = grammar,
        )
    }

    private fun codeOf(e: Throwable): Int = when (e) {
        is CancellationException -> Error.CANCELLED
        is IllegalArgumentException -> Error.INVALID_REQUEST
        is EngineException -> when (e.kind) {
            EngineException.Kind.LOAD_FAILED -> Error.MODEL_LOAD_FAILED
            EngineException.Kind.CONTEXT_OVERFLOW -> Error.CONTEXT_OVERFLOW
            EngineException.Kind.TEMPLATE -> Error.INVALID_REQUEST
            EngineException.Kind.FAILED -> Error.INTERNAL
        }
        else -> Error.INTERNAL
    }

    private val binder = object : IAICoreService.Stub() {
        override fun getApiVersion(): Int = AICoreContract.API_VERSION

        override fun getStatus(): Bundle {
            val auth = authorize()
            return Bundle().apply {
                putInt(Key.API_VERSION, AICoreContract.API_VERSION)
                putString(Key.SERVICE_VERSION, BuildConfig.VERSION_NAME)
                putString(Key.ACCESS_STATE, if (auth is Auth.Rejected) auth.state else AccessState.APPROVED)
                if (auth is Auth.Ok) {
                    putStringArray(Key.LOADED_MODELS, g.runtime.loadedModels.value.map { it.id }.toTypedArray())
                }
            }
        }

        override fun listModels(): Bundle {
            val auth = authorize()
            if (auth is Auth.Rejected) return error(auth.code, auth.message)
            val chat = g.defaultModel(ModelType.CHAT)?.id
            val embedding = g.defaultModel(ModelType.EMBEDDING)?.id
            return Bundle().apply {
                putParcelableArrayList(
                    Key.MODELS,
                    ArrayList(g.store.installed.value.map { modelBundle(it, it.id == chat || it.id == embedding) }),
                )
                chat?.let { putString(Key.DEFAULT_CHAT_MODEL, it) }
                embedding?.let { putString(Key.DEFAULT_EMBEDDING_MODEL, it) }
            }
        }

        override fun requestModel(spec: Bundle?): Bundle {
            val auth = authorize()
            if (auth is Auth.Rejected) return error(auth.code, auth.message)
            val caller = (auth as Auth.Ok).caller
            spec ?: return error(Error.INVALID_REQUEST, "Missing spec")
            val id = spec.getString(Key.ID)
            val url = spec.getString(Key.URL)
            val model: ModelSpec
            val fromCatalog: Boolean
            val catalogEntry = id?.let { g.catalog.find(it) }
            when {
                id != null && g.store.get(id) != null -> return status(ModelRequestStatus.INSTALLED)
                catalogEntry != null -> {
                    model = ModelSpec.fromCatalog(catalogEntry)
                    fromCatalog = true
                }
                url != null -> {
                    if (!url.startsWith("https://")) return error(Error.INVALID_REQUEST, "Only https URLs are allowed")
                    val fileName = url.substringBefore('?').substringAfterLast('/')
                    if (!fileName.endsWith(".gguf", ignoreCase = true)) return error(Error.INVALID_REQUEST, "URL must point to a .gguf file")
                    val name = spec.getString(Key.NAME) ?: fileName.removeSuffix(".gguf")
                    model = ModelSpec(
                        id = id ?: ModelSpec.customId(fileName),
                        name = name,
                        type = spec.getString(Key.TYPE)?.takeIf { it == ModelType.CHAT || it == ModelType.EMBEDDING }
                            ?: ModelSpec.guessType(name),
                        url = url,
                        fileName = fileName.replace(Regex("[^A-Za-z0-9._-]"), "_"),
                        sha256 = spec.getString(Key.SHA256),
                    )
                    fromCatalog = false
                }
                else -> return error(Error.INVALID_REQUEST, "Unknown model id and no URL given")
            }
            if (g.store.get(model.id) != null) return status(ModelRequestStatus.INSTALLED)
            if (g.downloads.downloads.value.containsKey(model.id)) return status(ModelRequestStatus.DOWNLOADING)
            val added = g.modelRequests.add(
                ModelRequest(model, caller.packageName, caller.label, spec.getString(Key.REASON), fromCatalog, System.currentTimeMillis()),
            )
            if (added) g.notifier.modelRequested(caller.label, model.name)
            return status(ModelRequestStatus.PENDING_USER)
        }

        private fun status(s: String) = Bundle().apply { putString(Key.STATUS, s) }

        override fun generate(request: Bundle?, callback: IGenerationCallback?): Bundle {
            val auth = authorize()
            if (auth is Auth.Rejected) return error(auth.code, auth.message)
            val caller = (auth as Auth.Ok).caller
            if (request == null || callback == null) return error(Error.INVALID_REQUEST, "Missing request or callback")
            val spec = resolveModel(request.getString(Key.MODEL), ModelType.CHAT)
                ?: return error(Error.MODEL_NOT_INSTALLED, "Model is not installed: ${request.getString(Key.MODEL) ?: "no chat model"}")
            if (spec.type != ModelType.CHAT) return error(Error.INVALID_REQUEST, "${spec.name} is not a chat model")
            val params = samplingOf(request)
            val stops = request.getStringArray(Key.STOP)?.toList().orEmpty()
            val requestId = g.runtime.newRequestId()

            g.runtime.submit(requestId, caller.uid) {
                val started = System.currentTimeMillis()
                try {
                    var clientGone = false
                    val result = g.runtime.withModel(requestId, spec, caller.label, "generate") { model ->
                        val prompt = buildPrompt(request, spec, model)
                        val filter = OutputFilter(stops, g.settings.current.hideThinking) { text ->
                            try {
                                callback.onToken(requestId, text)
                                true
                            } catch (_: RemoteException) {
                                clientGone = true
                                false
                            }
                        }
                        val stats = model.generate(prompt, params) { filter.push(it) }
                        filter.finish()
                        Triple(filter, stats, model)
                    }
                    val (filter, stats) = result
                    if (clientGone) return@submit
                    val reason = when {
                        filter.stoppedBySequence -> AICoreContract.StopReason.STOP_SEQUENCE
                        stats.stopReason == StopReason.EOS -> AICoreContract.StopReason.EOS
                        stats.stopReason == StopReason.LENGTH -> AICoreContract.StopReason.LENGTH
                        else -> AICoreContract.StopReason.CANCELLED
                    }
                    if (reason == AICoreContract.StopReason.CANCELLED) {
                        callback.onError(requestId, Error.CANCELLED, "Cancelled")
                        return@submit
                    }
                    callback.onComplete(requestId, Bundle().apply {
                        putString(Key.TEXT, filter.text)
                        putString(Key.STOP_REASON, reason)
                        putString(Key.MODEL, spec.id)
                        putInt(Key.PROMPT_TOKENS, stats.promptTokens)
                        putInt(Key.COMPLETION_TOKENS, stats.completionTokens)
                        putInt(Key.CACHED_TOKENS, stats.cachedTokens)
                        putLong(Key.DURATION_MS, System.currentTimeMillis() - started)
                    })
                } catch (e: Throwable) {
                    if (e !is CancellationException) Log.w(TAG, "Generation $requestId failed", e)
                    runCatching { callback.onError(requestId, codeOf(e), e.message ?: e.javaClass.simpleName) }
                    if (e is CancellationException) throw e
                }
            }
            return Bundle().apply { putString(Key.REQUEST_ID, requestId) }
        }

        override fun embed(request: Bundle?, callback: IEmbeddingCallback?): Bundle {
            val auth = authorize()
            if (auth is Auth.Rejected) return error(auth.code, auth.message)
            val caller = (auth as Auth.Ok).caller
            if (request == null || callback == null) return error(Error.INVALID_REQUEST, "Missing request or callback")
            val texts = request.getStringArray(Key.TEXTS)?.toList().orEmpty()
            if (texts.isEmpty()) return error(Error.INVALID_REQUEST, "No texts given")
            if (texts.size > MAX_EMBED_TEXTS) return error(Error.INVALID_REQUEST, "At most $MAX_EMBED_TEXTS texts per request")
            val spec = resolveModel(request.getString(Key.MODEL), ModelType.EMBEDDING)
                ?: return error(Error.MODEL_NOT_INSTALLED, "Model is not installed: ${request.getString(Key.MODEL) ?: "no embedding model"}")
            if (spec.type != ModelType.EMBEDDING) return error(Error.INVALID_REQUEST, "${spec.name} is not an embedding model")
            val normalize = request.getBoolean(Key.NORMALIZE, true)
            val requestId = g.runtime.newRequestId()

            g.runtime.submit(requestId, caller.uid) {
                val started = System.currentTimeMillis()
                try {
                    val dim = g.runtime.withModel(requestId, spec, caller.label, "embed") { model ->
                        var dim = 0
                        texts.forEachIndexed { i, text ->
                            if (g.runtime.isCancelled(requestId)) throw CancellationException("Cancelled")
                            val v = model.embed(text, normalize)
                            dim = v.size
                            callback.onEmbedding(requestId, i, v)
                        }
                        dim
                    }
                    callback.onComplete(requestId, Bundle().apply {
                        putString(Key.MODEL, spec.id)
                        putInt(Key.COUNT, texts.size)
                        putInt(Key.DIMENSION, dim)
                        putLong(Key.DURATION_MS, System.currentTimeMillis() - started)
                    })
                } catch (e: Throwable) {
                    if (e !is CancellationException && e !is RemoteException) Log.w(TAG, "Embedding $requestId failed", e)
                    runCatching { callback.onError(requestId, codeOf(e), e.message ?: e.javaClass.simpleName) }
                    if (e is CancellationException) throw e
                }
            }
            return Bundle().apply { putString(Key.REQUEST_ID, requestId) }
        }

        override fun cancel(requestId: String?) {
            requestId ?: return
            g.runtime.cancel(requestId, Binder.getCallingUid())
        }

        override fun countTokens(request: Bundle?): Bundle {
            val auth = authorize()
            if (auth is Auth.Rejected) return error(auth.code, auth.message)
            val caller = (auth as Auth.Ok).caller
            request ?: return error(Error.INVALID_REQUEST, "Missing request")
            val spec = resolveModel(request.getString(Key.MODEL), ModelType.CHAT)
                ?: return error(Error.MODEL_NOT_INSTALLED, "Model is not installed")
            // Binder threads may block; this waits for the inference thread.
            return try {
                val count = kotlinx.coroutines.runBlocking {
                    g.runtime.withModel(g.runtime.newRequestId(), spec, caller.label, "count") { model ->
                        model.countTokens(buildPrompt(request, spec, model))
                    }
                }
                Bundle().apply { putInt(Key.TOKEN_COUNT, count) }
            } catch (e: Exception) {
                error(codeOf(e), e.message ?: e.javaClass.simpleName)
            }
        }
    }

    companion object {
        private const val TAG = "AICoreService"
        private const val MAX_EMBED_TEXTS = 4096
    }
}
