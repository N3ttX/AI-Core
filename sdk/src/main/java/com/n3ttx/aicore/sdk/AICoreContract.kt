package com.n3ttx.aicore.sdk

/** Constants shared by AI Core Hub and its clients. Changing a value breaks compatibility. */
object AICoreContract {
    const val API_VERSION = 3

    const val SERVICE_PACKAGE = "com.n3ttx.aicore"
    const val ACTION_BIND = "com.n3ttx.aicore.action.BIND"
    const val ACTION_REQUEST_ACCESS = "com.n3ttx.aicore.action.REQUEST_ACCESS"
    const val ACTION_OPEN_MODELS = "com.n3ttx.aicore.action.OPEN_MODELS"

    object ModelType {
        /** Text generation (llama.cpp, GGUF). */
        const val CHAT = "chat"
        /** Text embeddings (llama.cpp, GGUF). */
        const val EMBEDDING = "embedding"
        /** Image + text embeddings in a shared space, e.g. CLIP, SigLIP (ONNX). */
        const val IMAGE_TEXT = "image_text"
        /** Face detection + face embeddings, e.g. InsightFace SCRFD/ArcFace (ONNX). */
        const val FACE = "face"
        /** Any other ONNX model, used through runModel with raw tensors. */
        const val ONNX = "onnx"

        val ALL = listOf(CHAT, EMBEDDING, IMAGE_TEXT, FACE, ONNX)
    }

    object Engine {
        const val LLAMA = "llama"
        const val ONNX = "onnx"
    }

    /** Task names reported by getCapabilities. */
    object Task {
        const val CHAT = "chat"
        const val COMPLETION = "completion"
        const val JSON_MODE = "json_mode"
        const val GRAMMAR = "grammar"
        const val TEXT_EMBEDDING = "text_embedding"
        const val IMAGE_EMBEDDING = "image_embedding"
        const val MULTIMODAL_TEXT_EMBEDDING = "multimodal_text_embedding"
        const val FACE_DETECTION = "face_detection"
        const val FACE_EMBEDDING = "face_embedding"
        const val ONNX_INFERENCE = "onnx_inference"
        const val TOKENIZE = "tokenize"
        /** Chat with images in the messages (vision language models, API 3). */
        const val VISION = "vision"
    }

    object DType {
        const val FLOAT32 = "float32"
        const val FLOAT16 = "float16"
        const val INT64 = "int64"
        const val INT32 = "int32"
        const val INT8 = "int8"
        const val UINT8 = "uint8"
        const val BOOL = "bool"
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

        // Token count / tokenize
        const val TOKEN_COUNT = "token_count"
        const val TOKEN_IDS = "token_ids"

        // Model details (API 2)
        const val ENGINE = "engine"
        const val CONFIG = "config"
        const val TASKS = "tasks"
        const val DESCRIPTION = "description"
        const val INSTALLED = "installed"
        const val DOWNLOADING = "downloading"
        const val COMPONENTS = "components"
        const val DEFAULTS = "defaults"
        /** A whole model package as JSON (same format as AI Core's catalog.json entries). */
        const val PACKAGE = "package"

        // Capabilities
        const val ENGINES = "engines"
        const val MODEL_TYPES = "model_types"
        const val MAX_BATCH = "max_batch"

        // Images
        const val IMAGES = "images"
        const val BITMAP = "bitmap"
        const val FD = "fd"
        /** Longest side to decode file images to (default depends on the task). */
        const val MAX_DIMENSION = "max_dimension"
        const val WIDTH = "width"
        const val HEIGHT = "height"

        // Faces
        const val FACES = "faces"
        const val BOX = "box"
        const val BOX_NORMALIZED = "box_normalized"
        const val LANDMARKS = "landmarks"
        const val SCORE = "score"
        const val EMBEDDING = "embedding"
        const val THRESHOLD = "threshold"
        const val MIN_FACE_SIZE = "min_face_size"
        const val WITH_EMBEDDINGS = "with_embeddings"

        // Raw tensors
        const val COMPONENT = "component"
        const val INPUTS = "inputs"
        const val OUTPUTS = "outputs"
        const val DTYPE = "dtype"
        const val SHAPE = "shape"
        const val BYTES = "bytes"
        const val SHARED_MEMORY = "shared_memory"
    }
}
