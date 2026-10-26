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
        val defaults = b.getBundle(Key.DEFAULTS)?.let { d -> d.keySet().associateWith { d.getString(it).orEmpty() } }.orEmpty()
        ModelList(models, b.getString(Key.DEFAULT_CHAT_MODEL), b.getString(Key.DEFAULT_EMBEDDING_MODEL), defaults)
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
        /** A full model package as JSON in the catalog.json format, for multi-file ONNX models. */
        packageJson: String? = null,
    ): String {
        val spec = Bundle().apply {
            packageJson?.let { putString(Key.PACKAGE, it) }
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

    /**
     * Embeds [texts]. With an image-text model (CLIP, SigLIP) the vectors live in the same space as
     * [embedImages], for text-to-image search. [model] null uses the default model of [type].
     */
    suspend fun embed(
        texts: List<String>,
        model: String? = null,
        normalize: Boolean = true,
        type: String = AICoreContract.ModelType.EMBEDDING,
        /** A batch job: interactive requests of any app go first. */
        background: Boolean = false,
    ): EmbeddingResult {
        val request = Bundle().apply {
            if (background) putBoolean(Key.BACKGROUND, true)
            putStringArray(Key.TEXTS, texts.toTypedArray())
            putBoolean(Key.NORMALIZE, normalize)
            putString(Key.TYPE, type)
            model?.let { putString(Key.MODEL, it) }
        }
        return collectEmbeddings(texts.size) { s, cb -> s.embed(request, cb) }
    }

    /**
     * Embeds images with an image-text model (null = default one). Vectors are L2-normalized and
     * comparable with text vectors from [embed] with the same model.
     */
    suspend fun embedImages(images: List<ImageInput>, model: String? = null, maxDimension: Int = 0, background: Boolean = false): EmbeddingResult {
        val request = Bundle().apply {
            if (background) putBoolean(Key.BACKGROUND, true)
            putParcelableArrayList(Key.IMAGES, ArrayList(images.map { it.toBundle() }))
            model?.let { putString(Key.MODEL, it) }
            if (maxDimension > 0) putInt(Key.MAX_DIMENSION, maxDimension)
        }
        return collectEmbeddings(images.size) { s, cb -> s.embedImages(request, cb) }
    }

    private suspend fun collectEmbeddings(
        count: Int,
        start: (IAICoreService, IEmbeddingCallback) -> Bundle,
    ): EmbeddingResult =
        withContext(Dispatchers.IO) {
            val s = awaitService()
            val vectors = arrayOfNulls<FloatArray>(count)
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
                    val response = start(s, callback)
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

    /**
     * Detects faces in each image with a face model (null = default one). Results come in the
     * order of [images]. Embeddings are L2-normalized; same person ≈ cosine similarity above the
     * model's "assign_threshold" (see [ModelInfo.config]).
     */
    suspend fun detectFaces(
        images: List<ImageInput>,
        model: String? = null,
        threshold: Float = 0.55f,
        minFaceSize: Float = 20f,
        withEmbeddings: Boolean = true,
        maxDimension: Int = 0,
        /** A batch job (scanning a gallery): interactive requests of any app go first. */
        background: Boolean = false,
    ): List<FaceResult> {
        val request = Bundle().apply {
            if (background) putBoolean(Key.BACKGROUND, true)
            putParcelableArrayList(Key.IMAGES, ArrayList(images.map { it.toBundle() }))
            model?.let { putString(Key.MODEL, it) }
            putFloat(Key.THRESHOLD, threshold)
            putFloat(Key.MIN_FACE_SIZE, minFaceSize)
            putBoolean(Key.WITH_EMBEDDINGS, withEmbeddings)
            if (maxDimension > 0) putInt(Key.MAX_DIMENSION, maxDimension)
        }
        return collectResults(images.size) { s, cb -> s.detectFaces(request, cb) }.map { FaceResult.fromBundle(it) }
    }

    /**
     * Runs one [component] (e.g. "vision", "text", "model") of an installed ONNX model with raw
     * tensors. [outputs] null returns all outputs. Use this for models AI Core has no task for.
     */
    suspend fun runModel(model: String, component: String, inputs: List<Tensor>, outputs: List<String>? = null, background: Boolean = false): List<Tensor> {
        val request = Bundle().apply {
            if (background) putBoolean(Key.BACKGROUND, true)
            putString(Key.MODEL, model)
            putString(Key.COMPONENT, component)
            putParcelableArrayList(Key.INPUTS, ArrayList(inputs.map { it.toBundle() }))
            outputs?.let { putStringArray(Key.OUTPUTS, it.toTypedArray()) }
        }
        val result = collectResults(1) { s, cb -> s.runModel(request, cb) }.firstOrNull()
            ?: throw AICoreException(Error.INTERNAL, "No result")
        @Suppress("DEPRECATION")
        return result.getParcelableArrayList<Bundle>(Key.OUTPUTS).orEmpty().map { Tensor.fromBundle(it) }
    }

    private suspend fun collectResults(count: Int, start: (IAICoreService, IResultCallback) -> Bundle): List<Bundle> =
        withContext(Dispatchers.IO) {
            val s = awaitService()
            val results = arrayOfNulls<Bundle>(count)
            suspendCancellableCoroutine { cont ->
                val death = IBinder.DeathRecipient {
                    if (cont.isActive) cont.resumeWithException(AICoreException(Error.SERVICE_UNAVAILABLE, "AI Core Hub stopped"))
                }
                val unlink = { runCatching { s.asBinder().unlinkToDeath(death, 0) } }
                val callback = object : IResultCallback.Stub() {
                    override fun onResult(requestId: String, index: Int, result: Bundle) {
                        if (index in results.indices) results[index] = result
                    }

                    override fun onComplete(requestId: String, result: Bundle) {
                        unlink()
                        if (cont.isActive) cont.resume(results.map { it ?: Bundle() })
                    }

                    override fun onError(requestId: String, code: Int, message: String?) {
                        unlink()
                        if (cont.isActive) cont.resumeWithException(AICoreException(code, message))
                    }
                }
                try {
                    s.asBinder().linkToDeath(death, 0)
                    val response = start(s, callback)
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
                    unlink()
                    if (cont.isActive) cont.resumeWithException(AICoreException(Error.SERVICE_UNAVAILABLE, e.message))
                }
            }
        }

    /** What this AI Core installation supports. Check [Capabilities.apiVersion] before using newer calls. */
    suspend fun capabilities(): Capabilities {
        val version = call { it.apiVersion }
        if (version < 2) {
            return Capabilities(version, "", listOf(AICoreContract.Task.CHAT, AICoreContract.Task.TEXT_EMBEDDING), listOf("llama"), listOf("chat", "embedding"), 0)
        }
        val b = call { it.capabilities }
        b.throwIfError()
        return Capabilities(
            apiVersion = b.getInt(Key.API_VERSION),
            serviceVersion = b.getString(Key.SERVICE_VERSION).orEmpty(),
            tasks = b.getStringArray(Key.TASKS)?.toList().orEmpty(),
            engines = b.getStringArray(Key.ENGINES)?.toList().orEmpty(),
            modelTypes = b.getStringArray(Key.MODEL_TYPES)?.toList().orEmpty(),
            maxBatch = b.getInt(Key.MAX_BATCH),
        )
    }

    /** Models AI Core can install, so an app can offer them and call [requestModel] with the id. */
    suspend fun catalog(): List<CatalogModel> {
        val b = call { it.catalog }
        b.throwIfError()
        @Suppress("DEPRECATION")
        return b.getParcelableArrayList<Bundle>(Key.MODELS).orEmpty().map {
            CatalogModel(
                id = it.getString(Key.ID).orEmpty(),
                name = it.getString(Key.NAME).orEmpty(),
                type = it.getString(Key.TYPE).orEmpty(),
                engine = it.getString(Key.ENGINE).orEmpty(),
                description = it.getString(Key.DESCRIPTION).orEmpty(),
                sizeBytes = it.getLong(Key.SIZE_BYTES),
                languages = it.getStringArray(Key.LANGUAGES)?.toList().orEmpty(),
                installed = it.getBoolean(Key.INSTALLED),
                downloading = it.getBoolean(Key.DOWNLOADING),
                config = it.getString(Key.CONFIG),
                tasks = it.getStringArray(Key.TASKS)?.toList().orEmpty(),
            )
        }
    }

    /** Token ids of [text] with the tokenizer of [model] (padded to the model's length for image-text models). */
    suspend fun tokenize(text: String, model: String? = null, type: String = AICoreContract.ModelType.CHAT): LongArray {
        val b = call {
            it.tokenize(Bundle().apply {
                putString(Key.TEXT, text)
                putString(Key.TYPE, type)
                model?.let { m -> putString(Key.MODEL, m) }
            })
        }
        b.throwIfError()
        return b.getLongArray(Key.TOKEN_IDS) ?: LongArray(0)
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
