package com.n3ttx.aicore.engine

/** Receives raw UTF-8 bytes of each generated token. Return false to stop. */
interface TokenSink {
    fun onPiece(bytes: ByteArray): Boolean
}

/** JNI entry points, see llama_jni.cpp. Not thread safe per handle. */
internal object LlamaNative {
    const val STOP_EOS = 0
    const val STOP_LENGTH = 1
    const val STOP_SINK = 2
    const val STOP_CANCELLED = 3
    const val ERR_GENERIC = -1
    const val ERR_CONTEXT_OVERFLOW = -2

    init {
        System.loadLibrary("aicore_engine")
        backendInit()
    }

    external fun backendInit()
    external fun lastError(): String
    external fun loadModel(path: String, nCtx: Int, nBatch: Int, nThreads: Int, embeddings: Boolean): Long
    external fun free(handle: Long)
    external fun cancel(handle: Long)
    external fun info(handle: Long): String
    external fun applyChatTemplate(handle: Long, templateOverride: String?, roles: Array<ByteArray>, contents: Array<ByteArray>, addAssistant: Boolean): ByteArray?
    external fun tokenCount(handle: Long, text: ByteArray): Int
    external fun tokenize(handle: Long, text: ByteArray): IntArray

    external fun generate(
        handle: Long,
        prompt: ByteArray,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        repeatPenalty: Float,
        seed: Int,
        grammar: String?,
        sink: TokenSink,
        stats: IntArray,
    ): Int

    external fun embed(handle: Long, text: ByteArray, normalize: Boolean): FloatArray?

    external fun loadProjector(handle: Long, path: String): Boolean

    external fun generateWithImages(
        handle: Long,
        prompt: ByteArray,
        images: Array<ByteArray>,
        dims: IntArray,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        repeatPenalty: Float,
        seed: Int,
        grammar: String?,
        sink: TokenSink,
        stats: IntArray,
    ): Int
}
