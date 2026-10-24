package com.n3ttx.aicore.sdk

/** Constants shared by AI Core Hub and its clients. Changing a value breaks compatibility. */
object AICoreContract {
    const val API_VERSION = 1

    const val SERVICE_PACKAGE = "com.n3ttx.aicore"
    const val ACTION_BIND = "com.n3ttx.aicore.action.BIND"
    const val ACTION_REQUEST_ACCESS = "com.n3ttx.aicore.action.REQUEST_ACCESS"
    const val ACTION_OPEN_MODELS = "com.n3ttx.aicore.action.OPEN_MODELS"

    object ModelType {
        const val CHAT = "chat"
        const val EMBEDDING = "embedding"
    }

    object AccessState {
        const val APPROVED = "approved"
        const val PENDING = "pending"
        const val DENIED = "denied"
    }

    object ModelRequestStatus {
        const val INSTALLED = "installed"
        const val PENDING_USER = "pending_user"
        const val DOWNLOADING = "downloading"
        const val REJECTED = "rejected"
    }

    object StopReason {
        const val EOS = "eos"
        const val LENGTH = "length"
        const val STOP_SEQUENCE = "stop_sequence"
        const val CANCELLED = "cancelled"
    }

    object Error {
        const val NOT_AUTHORIZED = 1
        const val PENDING_APPROVAL = 2
        const val MODEL_NOT_INSTALLED = 3
        const val MODEL_LOAD_FAILED = 4
        const val INVALID_REQUEST = 5
        const val CANCELLED = 6
        const val CONTEXT_OVERFLOW = 7
        const val INTERNAL = 8
        const val SERVICE_UNAVAILABLE = 9
    }

    /** Bundle keys. */
    object Key {
        // Common
        const val ERROR_CODE = "error_code"
        const val ERROR_MESSAGE = "error_message"
        const val REQUEST_ID = "request_id"
        const val MODEL = "model"

        // Status
        const val API_VERSION = "api_version"
        const val SERVICE_VERSION = "service_version"
        const val ACCESS_STATE = "access_state"
        const val LOADED_MODELS = "loaded_models"

        // Model list / model info
        const val MODELS = "models"
        const val ID = "id"
        const val NAME = "name"
        const val TYPE = "type"
        const val SIZE_BYTES = "size_bytes"
        const val CONTEXT_LENGTH = "context_length"
        const val EMBEDDING_DIM = "embedding_dim"
        const val LANGUAGES = "languages"
        const val IS_DEFAULT = "is_default"
        const val QUERY_PREFIX = "query_prefix"
        const val DOCUMENT_PREFIX = "document_prefix"
        const val DEFAULT_CHAT_MODEL = "default_chat_model"
        const val DEFAULT_EMBEDDING_MODEL = "default_embedding_model"

        // Model request
        const val URL = "url"
        const val SHA256 = "sha256"
        const val REASON = "reason"
        const val STATUS = "status"

        // Generation request
        const val MESSAGES = "messages"
        const val ROLE = "role"
        const val CONTENT = "content"
        const val PROMPT = "prompt"
        const val MAX_TOKENS = "max_tokens"
        const val TEMPERATURE = "temperature"
        const val TOP_P = "top_p"
        const val TOP_K = "top_k"
        const val MIN_P = "min_p"
        const val REPEAT_PENALTY = "repeat_penalty"
        const val SEED = "seed"
        const val STOP = "stop"
        const val JSON_MODE = "json_mode"
        const val GRAMMAR = "grammar"

        // Generation result
        const val TEXT = "text"
        const val STOP_REASON = "stop_reason"
        const val PROMPT_TOKENS = "prompt_tokens"
        const val COMPLETION_TOKENS = "completion_tokens"
        const val CACHED_TOKENS = "cached_tokens"
        const val DURATION_MS = "duration_ms"

        // Embedding request / result
        const val TEXTS = "texts"
        const val NORMALIZE = "normalize"
        const val COUNT = "count"
        const val DIMENSION = "dimension"

        // Token count
        const val TOKEN_COUNT = "token_count"
    }
}
