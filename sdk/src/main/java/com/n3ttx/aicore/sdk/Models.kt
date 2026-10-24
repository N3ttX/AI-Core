package com.n3ttx.aicore.sdk

import android.os.Bundle
import com.n3ttx.aicore.sdk.AICoreContract.Key

data class ChatMessage(val role: String, val content: String) {
    companion object {
        fun system(content: String) = ChatMessage("system", content)
        fun user(content: String) = ChatMessage("user", content)
        fun assistant(content: String) = ChatMessage("assistant", content)
    }
}

data class GenerationOptions(
    val maxTokens: Int = 512,
    val temperature: Float = 0.7f,
    val topP: Float = 0.95f,
    val topK: Int = 40,
    val minP: Float = 0.05f,
    val repeatPenalty: Float = 1.1f,
    /** Negative means random. */
    val seed: Int = -1,
    val stop: List<String> = emptyList(),
    /** Constrains the output to valid JSON. */
    val jsonMode: Boolean = false,
    /** A GBNF grammar (root rule "root"). Takes precedence over [jsonMode]. */
    val grammar: String? = null,
)

/**
 * Either [messages] (formatted with the model's chat template) or a raw [prompt]
 * (passed to the model as is) must be set.
 */
data class GenerationRequest(
    val messages: List<ChatMessage> = emptyList(),
    val prompt: String? = null,
    /** Model id; null uses the user's default chat model. */
    val model: String? = null,
    val options: GenerationOptions = GenerationOptions(),
) {
    fun toBundle(): Bundle = Bundle().apply {
        model?.let { putString(Key.MODEL, it) }
        prompt?.let { putString(Key.PROMPT, it) }
        if (messages.isNotEmpty()) {
            putParcelableArrayList(Key.MESSAGES, ArrayList(messages.map {
                Bundle().apply {
                    putString(Key.ROLE, it.role)
                    putString(Key.CONTENT, it.content)
                }
            }))
        }
        putInt(Key.MAX_TOKENS, options.maxTokens)
        putFloat(Key.TEMPERATURE, options.temperature)
        putFloat(Key.TOP_P, options.topP)
        putInt(Key.TOP_K, options.topK)
        putFloat(Key.MIN_P, options.minP)
        putFloat(Key.REPEAT_PENALTY, options.repeatPenalty)
        putInt(Key.SEED, options.seed)
        if (options.stop.isNotEmpty()) putStringArray(Key.STOP, options.stop.toTypedArray())
        putBoolean(Key.JSON_MODE, options.jsonMode)
        options.grammar?.let { putString(Key.GRAMMAR, it) }
    }
}

data class GenerationResult(
    val text: String,
    val stopReason: String,
    val model: String,
    val promptTokens: Int,
    val completionTokens: Int,
    /** Prompt tokens reused from the previous request on the same model. */
    val cachedTokens: Int,
    val durationMs: Long,
) {
    companion object {
        fun fromBundle(b: Bundle) = GenerationResult(
            text = b.getString(Key.TEXT).orEmpty(),
            stopReason = b.getString(Key.STOP_REASON).orEmpty(),
            model = b.getString(Key.MODEL).orEmpty(),
            promptTokens = b.getInt(Key.PROMPT_TOKENS),
            completionTokens = b.getInt(Key.COMPLETION_TOKENS),
            cachedTokens = b.getInt(Key.CACHED_TOKENS),
            durationMs = b.getLong(Key.DURATION_MS),
        )
    }
}

sealed interface GenerationEvent {
    data class Token(val text: String) : GenerationEvent
    data class Completed(val result: GenerationResult) : GenerationEvent
}

data class EmbeddingResult(
    val model: String,
    val dimension: Int,
    val vectors: List<FloatArray>,
)

data class ModelInfo(
    val id: String,
    val name: String,
    val type: String,
    val sizeBytes: Long,
    val contextLength: Int,
    val embeddingDim: Int,
    val languages: List<String>,
    val isDefault: Boolean,
    /** For embedding models: prefix to put before search queries, if the model needs one. */
    val queryPrefix: String?,
    /** For embedding models: prefix to put before documents, if the model needs one. */
    val documentPrefix: String?,
    /** [AICoreContract.Engine] that runs this model. */
    val engine: String,
    /** What this model can do, see [AICoreContract.Task]. */
    val tasks: List<String>,
    /** Model specific settings as JSON (image size, face match threshold, …), or null. */
    val config: String?,
    /** ONNX components (e.g. "vision", "text") usable with runModel. */
    val components: List<String>,
) {
    companion object {
        fun fromBundle(b: Bundle) = ModelInfo(
            id = b.getString(Key.ID).orEmpty(),
            name = b.getString(Key.NAME).orEmpty(),
            type = b.getString(Key.TYPE).orEmpty(),
            sizeBytes = b.getLong(Key.SIZE_BYTES),
            contextLength = b.getInt(Key.CONTEXT_LENGTH),
            embeddingDim = b.getInt(Key.EMBEDDING_DIM),
            languages = b.getStringArray(Key.LANGUAGES)?.toList().orEmpty(),
            isDefault = b.getBoolean(Key.IS_DEFAULT),
            queryPrefix = b.getString(Key.QUERY_PREFIX),
            documentPrefix = b.getString(Key.DOCUMENT_PREFIX),
            engine = b.getString(Key.ENGINE) ?: AICoreContract.Engine.LLAMA,
            tasks = b.getStringArray(Key.TASKS)?.toList().orEmpty(),
            config = b.getString(Key.CONFIG),
            components = b.getStringArray(Key.COMPONENTS)?.toList().orEmpty(),
        )
    }
}

data class ModelList(
    val models: List<ModelInfo>,
    val defaultChatModel: String?,
    val defaultEmbeddingModel: String?,
    /** Default model id per [AICoreContract.ModelType]. */
    val defaults: Map<String, String> = emptyMap(),
) {
    fun default(type: String): ModelInfo? = defaults[type]?.let { id -> models.firstOrNull { it.id == id } }
    fun ofType(type: String): List<ModelInfo> = models.filter { it.type == type }
}

data class CoreStatus(
    val apiVersion: Int,
    val serviceVersion: String,
    val accessState: String,
    val loadedModels: List<String>,
)

class AICoreException(val code: Int, message: String?) : Exception(message ?: "AI Core error $code")
