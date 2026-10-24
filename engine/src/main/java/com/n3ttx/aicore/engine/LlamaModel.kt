package com.n3ttx.aicore.engine

import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.CharBuffer
import java.nio.charset.CodingErrorAction
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

class EngineException(val kind: Kind, message: String) : Exception(message) {
    enum class Kind { LOAD_FAILED, CONTEXT_OVERFLOW, TEMPLATE, FAILED }
}

data class LoadOptions(
    val contextLength: Int = 8192,
    val batchSize: Int = 512,
    val threads: Int = 4,
    val embeddings: Boolean = false,
)

data class SamplingParams(
    val maxTokens: Int = 512,
    val temperature: Float = 0.7f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val minP: Float = 0.05f,
    val repeatPenalty: Float = 1.1f,
    val seed: Int = -1,
    val grammar: String? = null,
)

enum class StopReason { EOS, LENGTH, STOPPED, CANCELLED }

data class GenerationStats(
    val stopReason: StopReason,
    val promptTokens: Int,
    val completionTokens: Int,
    val cachedTokens: Int,
)

data class ModelMetadata(
    val description: String,
    val contextLength: Int,
    val trainedContextLength: Int,
    val embeddingDim: Int,
    val parameterCount: Long,
    val sizeBytes: Long,
    val hasChatTemplate: Boolean,
)

/**
 * A GGUF model loaded into memory together with its llama.cpp context.
 * All calls are serialized; [cancel] may be called from any thread.
 */
class LlamaModel private constructor(private var handle: Long, val options: LoadOptions) : Closeable {
    private val lock = ReentrantLock()

    val metadata: ModelMetadata by lazy {
        lock.withLock {
            val map = LlamaNative.info(checkOpen()).lines()
                .mapNotNull { line -> line.indexOf('=').takeIf { it > 0 }?.let { line.substring(0, it) to line.substring(it + 1) } }
                .toMap()
            ModelMetadata(
                description = map["description"].orEmpty(),
                contextLength = map["n_ctx"]?.toIntOrNull() ?: 0,
                trainedContextLength = map["n_ctx_train"]?.toIntOrNull() ?: 0,
                embeddingDim = map["n_embd"]?.toIntOrNull() ?: 0,
                parameterCount = map["n_params"]?.toLongOrNull() ?: 0,
                sizeBytes = map["size"]?.toLongOrNull() ?: 0,
                hasChatTemplate = map["has_chat_template"] == "1",
            )
        }
    }

    private fun checkOpen(): Long {
        check(handle != 0L) { "Model is closed" }
        return handle
    }

    /**
     * Formats chat messages (role, content) with the model's template, or with the
     * llama.cpp built-in template named [templateOverride] (e.g. "chatml").
     */
    fun applyChatTemplate(
        messages: List<Pair<String, String>>,
        templateOverride: String? = null,
        addAssistant: Boolean = true,
    ): String = lock.withLock {
        val roles = messages.map { it.first.toByteArray() }.toTypedArray()
        val contents = messages.map { it.second.toByteArray() }.toTypedArray()
        val bytes = LlamaNative.applyChatTemplate(checkOpen(), templateOverride?.takeIf { it.isNotBlank() }, roles, contents, addAssistant)
            ?: throw EngineException(EngineException.Kind.TEMPLATE, LlamaNative.lastError())
        String(bytes, Charsets.UTF_8)
    }

    fun countTokens(text: String): Int = lock.withLock { LlamaNative.tokenCount(checkOpen(), text.toByteArray()) }

    /**
     * Generates a completion for [prompt]. [onText] receives decoded text chunks and
     * returns false to stop early.
     */
    fun generate(prompt: String, params: SamplingParams, onText: (String) -> Boolean): GenerationStats = lock.withLock {
        val decoder = StreamingUtf8Decoder()
        val sink = object : TokenSink {
            override fun onPiece(bytes: ByteArray): Boolean {
                val text = decoder.decode(bytes)
                return text.isEmpty() || onText(text)
            }
        }
        val stats = IntArray(3)
        val rc = LlamaNative.generate(
            checkOpen(), prompt.toByteArray(), params.maxTokens, params.temperature, params.topK, params.topP,
            params.minP, params.repeatPenalty, params.seed, params.grammar, sink, stats,
        )
        val reason = when (rc) {
            LlamaNative.STOP_EOS -> StopReason.EOS
            LlamaNative.STOP_LENGTH -> StopReason.LENGTH
            LlamaNative.STOP_SINK -> StopReason.STOPPED
            LlamaNative.STOP_CANCELLED -> StopReason.CANCELLED
            LlamaNative.ERR_CONTEXT_OVERFLOW -> throw EngineException(EngineException.Kind.CONTEXT_OVERFLOW, LlamaNative.lastError())
            else -> throw EngineException(EngineException.Kind.FAILED, LlamaNative.lastError())
        }
        if (reason != StopReason.STOPPED) decoder.flush().takeIf { it.isNotEmpty() }?.let { onText(it) }
        GenerationStats(reason, stats[0], stats[1], stats[2])
    }

    fun embed(text: String, normalize: Boolean = true): FloatArray = lock.withLock {
        LlamaNative.embed(checkOpen(), text.toByteArray(), normalize)
            ?: throw EngineException(EngineException.Kind.FAILED, LlamaNative.lastError())
    }

    /** Aborts the running generation or prompt evaluation, if any. */
    fun cancel() {
        val h = handle
        if (h != 0L) LlamaNative.cancel(h)
    }

    override fun close() {
        cancel()
        lock.withLock {
            if (handle != 0L) {
                LlamaNative.free(handle)
                handle = 0L
            }
        }
    }

    companion object {
        fun load(path: String, options: LoadOptions): LlamaModel {
            val handle = LlamaNative.loadModel(path, options.contextLength, options.batchSize, options.threads, options.embeddings)
            if (handle == 0L) throw EngineException(EngineException.Kind.LOAD_FAILED, LlamaNative.lastError())
            return LlamaModel(handle, options)
        }
    }
}

/** Decodes UTF-8 split at arbitrary byte boundaries (tokens can end mid-character). */
internal class StreamingUtf8Decoder {
    private val decoder = Charsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPLACE)
        .onUnmappableCharacter(CodingErrorAction.REPLACE)
    private var pending = ByteArray(0)

    fun decode(bytes: ByteArray): String {
        val input = ByteBuffer.wrap(pending + bytes)
        val out = CharBuffer.allocate((input.remaining() + 1) * 2)
        decoder.decode(input, out, false)
        pending = ByteArray(input.remaining()).also { input.get(it) }
        out.flip()
        return out.toString()
    }

    fun flush(): String {
        if (pending.isEmpty()) return ""
        val text = String(pending, Charsets.UTF_8)
        pending = ByteArray(0)
        return text
    }
}

/** GBNF grammar that accepts any JSON value with an object or array at the top. */
object Grammars {
    val JSON = """
        root   ::= object | array
        value  ::= object | array | string | number | ("true" | "false" | "null") ws
        object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
        array  ::= "[" ws ( value ("," ws value)* )? "]" ws
        string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt/] | "u" [0-9a-fA-F]{4}) )* "\"" ws
        number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws
        ws     ::= | " " | "\n" [ \t]{0,20}
    """.trimIndent()
}
