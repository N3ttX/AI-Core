package com.n3ttx.aicore.sdk

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.IBinder
import android.os.RemoteException
import com.n3ttx.aicore.sdk.AICoreContract.Error
import com.n3ttx.aicore.sdk.AICoreContract.Key
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.io.Closeable
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Client for AI Core Hub. Create one per process, call [close] when done.
 *
 * ```
 * val core = AICoreClient(context)
 * core.generate(GenerationRequest(listOf(ChatMessage.user("Hi")))).collect { ... }
 * ```
 *
 * A new app has to be approved by the user once: launch [accessRequestIntent]
 * with startActivityForResult (RESULT_OK means approved), or call any method,
 * which fails with [Error.PENDING_APPROVAL] and makes AI Core Hub show a notification.
 */
class AICoreClient(context: Context) : Closeable {
    private val appContext = context.applicationContext
    private val service = MutableStateFlow<IAICoreService?>(null)
    private val connected = MutableStateFlow(false)
    private var bound = false

    val isConnected: StateFlow<Boolean> = connected.asStateFlow()

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, binder: IBinder) {
            service.value = IAICoreService.Stub.asInterface(binder)
            connected.value = true
        }

        override fun onServiceDisconnected(name: ComponentName) {
            // The system rebinds automatically when the service comes back.
            service.value = null
            connected.value = false
        }

        override fun onBindingDied(name: ComponentName) {
            service.value = null
            connected.value = false
            unbind()
            connect()
        }

        override fun onNullBinding(name: ComponentName) {
            unbind()
        }
    }

    /** Whether AI Core Hub is installed on this device. */
    fun isServiceInstalled(): Boolean = try {
        appContext.packageManager.getPackageInfo(AICoreContract.SERVICE_PACKAGE, 0)
        true
    } catch (_: PackageManager.NameNotFoundException) {
        false
    }

    /** Binds to the service. Called automatically by every request. */
    @Synchronized
    fun connect(): Boolean {
        if (bound) return true
        val intent = Intent(AICoreContract.ACTION_BIND).setPackage(AICoreContract.SERVICE_PACKAGE)
        bound = appContext.bindService(intent, connection, Context.BIND_AUTO_CREATE)
        return bound
    }

    @Synchronized
    private fun unbind() {
        if (!bound) return
        bound = false
        runCatching { appContext.unbindService(connection) }
    }

    override fun close() {
        unbind()
        service.value = null
        connected.value = false
    }

    /** Intent for startActivityForResult that asks the user to approve this app. */
    fun accessRequestIntent(): Intent =
        Intent(AICoreContract.ACTION_REQUEST_ACCESS).setPackage(AICoreContract.SERVICE_PACKAGE)

    /** Intent that opens the model manager of AI Core Hub. */
    fun openModelsIntent(): Intent =
        Intent(AICoreContract.ACTION_OPEN_MODELS)
            .setPackage(AICoreContract.SERVICE_PACKAGE)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)

    private suspend fun awaitService(timeoutMs: Long = 10_000): IAICoreService {
        service.value?.let { return it }
        if (!isServiceInstalled()) throw AICoreException(Error.SERVICE_UNAVAILABLE, "AI Core Hub is not installed")
        if (!connect()) throw AICoreException(Error.SERVICE_UNAVAILABLE, "Could not bind to AI Core Hub")
        return withTimeoutOrNull(timeoutMs) { service.filterNotNull().first() }
            ?: throw AICoreException(Error.SERVICE_UNAVAILABLE, "AI Core Hub did not respond")
    }

    private suspend fun <T> call(block: (IAICoreService) -> T): T = withContext(Dispatchers.IO) {
        val s = awaitService()
        try {
            block(s)
        } catch (e: RemoteException) {
            throw AICoreException(Error.SERVICE_UNAVAILABLE, e.message)
        }
    }

    suspend fun status(): CoreStatus = call { it.status }.let { b ->
        b.throwIfError()
        CoreStatus(
            apiVersion = b.getInt(Key.API_VERSION),
            serviceVersion = b.getString(Key.SERVICE_VERSION).orEmpty(),
            accessState = b.getString(Key.ACCESS_STATE).orEmpty(),
            loadedModels = b.getStringArray(Key.LOADED_MODELS)?.toList().orEmpty(),
        )
    }

    suspend fun listModels(): ModelList = call { it.listModels() }.let { b ->
        b.throwIfError()
        @Suppress("DEPRECATION")
        val models = b.getParcelableArrayList<Bundle>(Key.MODELS).orEmpty().map(ModelInfo::fromBundle)
        ModelList(models, b.getString(Key.DEFAULT_CHAT_MODEL), b.getString(Key.DEFAULT_EMBEDDING_MODEL))
    }

    /**
     * Asks the user to install a model. Pass a catalog id, or a direct GGUF download
     * [url] with a display [name] and [type] for models outside the catalog.
     * Returns one of [AICoreContract.ModelRequestStatus].
     */
    suspend fun requestModel(
        id: String? = null,
        url: String? = null,
        name: String? = null,
        type: String = AICoreContract.ModelType.CHAT,
        sha256: String? = null,
        reason: String? = null,
    ): String {
        val spec = Bundle().apply {
            id?.let { putString(Key.ID, it) }
            url?.let { putString(Key.URL, it) }
            name?.let { putString(Key.NAME, it) }
            putString(Key.TYPE, type)
            sha256?.let { putString(Key.SHA256, it) }
            reason?.let { putString(Key.REASON, it) }
        }
        val b = call { it.requestModel(spec) }
        b.throwIfError()
        return b.getString(Key.STATUS).orEmpty()
    }

    /** Streams generated text. Cancelling the collector cancels the request. */
    fun generate(request: GenerationRequest): Flow<GenerationEvent> = callbackFlow {
        val s = awaitService()
        val finished = AtomicBoolean(false)
        val callback = object : IGenerationCallback.Stub() {
            override fun onToken(requestId: String, text: String) {
                trySend(GenerationEvent.Token(text))
            }

            override fun onComplete(requestId: String, result: Bundle) {
                finished.set(true)
                trySend(GenerationEvent.Completed(GenerationResult.fromBundle(result)))
                channel.close()
            }

            override fun onError(requestId: String, code: Int, message: String?) {
                finished.set(true)
                channel.close(AICoreException(code, message))
            }
        }
        val response = try {
            s.generate(request.toBundle(), callback)
        } catch (e: RemoteException) {
            throw AICoreException(Error.SERVICE_UNAVAILABLE, e.message)
        }
        response.throwIfError()
        val requestId = response.getString(Key.REQUEST_ID)
        val death = IBinder.DeathRecipient {
            channel.close(AICoreException(Error.SERVICE_UNAVAILABLE, "AI Core Hub stopped"))
        }
        try {
            s.asBinder().linkToDeath(death, 0)
        } catch (e: RemoteException) {
            throw AICoreException(Error.SERVICE_UNAVAILABLE, e.message)
        }
        awaitClose {
            runCatching { s.asBinder().unlinkToDeath(death, 0) }
            if (!finished.get() && requestId != null) runCatching { s.cancel(requestId) }
        }
    }.buffer(Channel.UNLIMITED).flowOn(Dispatchers.IO)

    /** Generates and returns the full result. */
    suspend fun complete(request: GenerationRequest): GenerationResult {
        var result: GenerationResult? = null
        generate(request).collect { if (it is GenerationEvent.Completed) result = it.result }
        return result ?: throw AICoreException(Error.INTERNAL, "Generation ended without a result")
    }

    /** Embeds [texts] with the given model (null = default embedding model). */
    suspend fun embed(texts: List<String>, model: String? = null, normalize: Boolean = true): EmbeddingResult =
        withContext(Dispatchers.IO) {
            val s = awaitService()
            val vectors = arrayOfNulls<FloatArray>(texts.size)
            val request = Bundle().apply {
                putStringArray(Key.TEXTS, texts.toTypedArray())
                putBoolean(Key.NORMALIZE, normalize)
                model?.let { putString(Key.MODEL, it) }
            }
            suspendCancellableCoroutine { cont ->
                val death = IBinder.DeathRecipient {
                    if (cont.isActive) cont.resumeWithException(AICoreException(Error.SERVICE_UNAVAILABLE, "AI Core Hub stopped"))
                }
                val unlink = { runCatching { s.asBinder().unlinkToDeath(death, 0) } }
                val callback = object : IEmbeddingCallback.Stub() {
                    override fun onEmbedding(requestId: String, index: Int, vector: FloatArray) {
                        if (index in vectors.indices) vectors[index] = vector
                    }

                    override fun onComplete(requestId: String, result: Bundle) {
                        unlink()
                        val list = vectors.map { it ?: FloatArray(0) }
                        if (cont.isActive) cont.resume(
                            EmbeddingResult(
                                model = result.getString(Key.MODEL).orEmpty(),
                                dimension = result.getInt(Key.DIMENSION),
                                vectors = list,
                            )
                        )
                    }

                    override fun onError(requestId: String, code: Int, message: String?) {
                        unlink()
                        if (cont.isActive) cont.resumeWithException(AICoreException(code, message))
                    }
                }
                try {
                    s.asBinder().linkToDeath(death, 0)
                    val response = s.embed(request, callback)
                    val error = response.errorOrNull()
                    if (error != null) {
                        unlink()
                        cont.resumeWithException(error)
                    } else {
                        val requestId = response.getString(Key.REQUEST_ID)
                        cont.invokeOnCancellation {
                            unlink()
                            if (requestId != null) runCatching { s.cancel(requestId) }
                        }
                    }
                } catch (e: RemoteException) {
                    if (cont.isActive) cont.resumeWithException(AICoreException(Error.SERVICE_UNAVAILABLE, e.message))
                }
            }
        }

    /** Counts the tokens [request] would use as prompt, including the chat template. */
    suspend fun countTokens(request: GenerationRequest): Int {
        val b = call { it.countTokens(request.toBundle()) }
        b.throwIfError()
        return b.getInt(Key.TOKEN_COUNT)
    }
}

internal fun Bundle.errorOrNull(): AICoreException? {
    val code = getInt(Key.ERROR_CODE, 0)
    return if (code == 0) null else AICoreException(code, getString(Key.ERROR_MESSAGE))
}

internal fun Bundle.throwIfError() {
    errorOrNull()?.let { throw it }
}
