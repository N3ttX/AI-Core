// JNI bridge between the Kotlin engine and llama.cpp.
//
// Text crosses the boundary as UTF-8 byte arrays, never as jstring: JNI's
// "modified UTF-8" mangles characters outside the BMP (emoji, some CJK).

#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#define TAG "AICoreEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

// Return codes of nativeGenerate, mirrored in LlamaNative.kt.
constexpr int STOP_EOS = 0;
constexpr int STOP_LENGTH = 1;
constexpr int STOP_SINK = 2;
constexpr int STOP_CANCELLED = 3;
constexpr int ERR_GENERIC = -1;
constexpr int ERR_CONTEXT_OVERFLOW = -2;

thread_local std::string g_last_error;

struct Engine {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    const llama_vocab *vocab = nullptr;
    bool embeddings = false;
    int32_t n_threads = 4;
    // Vision projector (mmproj) of a vision language model, loaded on demand by loadProjector.
    mtmd_context *mtmd = nullptr;
    // Tokens currently held in the KV cache for sequence 0, used for prefix reuse.
    std::vector<llama_token> cached;
    std::atomic<bool> cancel{false};
};

Engine *as_engine(jlong handle) { return reinterpret_cast<Engine *>(handle); }

std::string to_string(JNIEnv *env, jbyteArray bytes) {
    if (bytes == nullptr) return {};
    const jsize len = env->GetArrayLength(bytes);
    std::string out(static_cast<size_t>(len), '\0');
    if (len > 0) env->GetByteArrayRegion(bytes, 0, len, reinterpret_cast<jbyte *>(out.data()));
    return out;
}

jbyteArray to_bytes(JNIEnv *env, const std::string &s) {
    jbyteArray arr = env->NewByteArray(static_cast<jsize>(s.size()));
    if (arr != nullptr && !s.empty()) {
        env->SetByteArrayRegion(arr, 0, static_cast<jsize>(s.size()), reinterpret_cast<const jbyte *>(s.data()));
    }
    return arr;
}

std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &text, bool add_special, bool parse_special) {
    int32_t n = -llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0, add_special, parse_special);
    if (n <= 0) return {};
    std::vector<llama_token> tokens(static_cast<size_t>(n));
    n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(), n, add_special, parse_special);
    if (n < 0) return {};
    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

std::string token_piece(const llama_vocab *vocab, llama_token token) {
    char buf[256];
    int32_t n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
    if (n >= 0) return std::string(buf, static_cast<size_t>(n));
    std::string big(static_cast<size_t>(-n), '\0');
    n = llama_token_to_piece(vocab, token, big.data(), static_cast<int32_t>(big.size()), 0, false);
    return n >= 0 ? big.substr(0, static_cast<size_t>(n)) : std::string();
}

bool abort_cb(void *data) {
    return static_cast<Engine *>(data)->cancel.load();
}

void log_cb(ggml_log_level level, const char *text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        __android_log_write(ANDROID_LOG_ERROR, TAG, text);
    } else if (level == GGML_LOG_LEVEL_WARN) {
        __android_log_write(ANDROID_LOG_WARN, TAG, text);
    }
}

// Decodes tokens[from..] at positions starting at start_pos, in chunks of n_batch.
// Only the very last token requests logits. Returns llama_decode's result.
int decode_tokens(Engine *e, const std::vector<llama_token> &tokens, size_t from, llama_pos start_pos) {
    const int32_t n_batch = static_cast<int32_t>(llama_n_batch(e->ctx));
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    int rc = 0;
    size_t i = from;
    llama_pos pos = start_pos;
    while (i < tokens.size()) {
        const int32_t n = static_cast<int32_t>(std::min<size_t>(static_cast<size_t>(n_batch), tokens.size() - i));
        batch.n_tokens = n;
        for (int32_t k = 0; k < n; ++k) {
            batch.token[k] = tokens[i + k];
            batch.pos[k] = pos + k;
            batch.n_seq_id[k] = 1;
            batch.seq_id[k][0] = 0;
            batch.logits[k] = (i + k == tokens.size() - 1) ? 1 : 0;
        }
        rc = llama_decode(e->ctx, batch);
        if (rc != 0) break;
        i += static_cast<size_t>(n);
        pos += n;
        if (e->cancel.load()) {
            rc = 2;
            break;
        }
    }
    llama_batch_free(batch);
    return rc;
}

// Builds the sampler chain of a request. Returns nullptr (and sets g_last_error) for an invalid grammar.
llama_sampler *make_sampler(JNIEnv *env, Engine *e, jfloat temperature, jint top_k, jfloat top_p, jfloat min_p,
                            jfloat repeat_penalty, jint seed, jstring jgrammar) {
    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (jgrammar != nullptr) {
        const char *g = env->GetStringUTFChars(jgrammar, nullptr);
        llama_sampler *gs = llama_sampler_init_grammar(e->vocab, g, "root");
        env->ReleaseStringUTFChars(jgrammar, g);
        if (gs == nullptr) {
            llama_sampler_free(smpl);
            g_last_error = "Invalid grammar";
            return nullptr;
        }
        llama_sampler_chain_add(smpl, gs);
    }
    if (repeat_penalty > 0.0f && repeat_penalty != 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(llama_vocab_n_tokens(e->vocab), 64, repeat_penalty, 0.0f, 0.0f));
    }
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        if (top_k > 0) llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        if (top_p > 0.0f && top_p < 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        if (min_p > 0.0f) llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl,
                                llama_sampler_init_dist(seed < 0 ? LLAMA_DEFAULT_SEED : static_cast<uint32_t>(seed)));
    }
    return smpl;
}

void clear_memory(Engine *e);

// Samples up to [limit] tokens after the prompt, which ends at position [pos], and streams them to the sink.
// With [track_cache] the generated tokens are appended to e->cached (text prompts, for prefix reuse).
int sample_loop(JNIEnv *env, Engine *e, jobject sink, jmethodID on_piece, llama_sampler *smpl, int32_t limit,
                llama_pos pos, bool track_cache, int &generated) {
    int result = STOP_LENGTH;
    llama_batch batch = llama_batch_init(1, 0, 1);
    while (generated < limit) {
        if (e->cancel.load()) {
            result = STOP_CANCELLED;
            break;
        }
        const llama_token tok = llama_sampler_sample(smpl, e->ctx, -1);
        if (llama_vocab_is_eog(e->vocab, tok)) {
            result = STOP_EOS;
            break;
        }
        ++generated;

        const std::string piece = token_piece(e->vocab, tok);
        if (!piece.empty()) {
            jbyteArray arr = to_bytes(env, piece);
            const jboolean keep_going = env->CallBooleanMethod(sink, on_piece, arr);
            env->DeleteLocalRef(arr);
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                result = STOP_SINK;
                break;
            }
            if (!keep_going) {
                result = STOP_SINK;
                // The token is not decoded, so the cache stays consistent with e->cached.
                break;
            }
        }

        batch.n_tokens = 1;
        batch.token[0] = tok;
        batch.pos[0] = pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        const int rc = llama_decode(e->ctx, batch);
        if (rc != 0) {
            result = (rc == 2) ? STOP_CANCELLED : ERR_GENERIC;
            if (result == ERR_GENERIC) g_last_error = "llama_decode failed: " + std::to_string(rc);
            clear_memory(e);
            break;
        }
        ++pos;
        if (track_cache) e->cached.push_back(tok);
    }
    llama_batch_free(batch);
    return result;
}

void clear_memory(Engine *e) {
    llama_memory_t mem = llama_get_memory(e->ctx);
    if (mem != nullptr) llama_memory_clear(mem, true);
    e->cached.clear();
}

}  // namespace

extern "C" {

JNIEXPORT void JNICALL
Java_com_n3ttx_aicore_engine_LlamaNative_backendInit(JNIEnv *, jobject) {
    llama_log_set(log_cb, nullptr);
    mtmd_helper_log_set(log_cb, nullptr);
    llama_backend_init();
}

JNIEXPORT jstring JNICALL
Java_com_n3ttx_aicore_engine_LlamaNative_lastError(JNIEnv *env, jobject) {
    return env->NewStringUTF(g_last_error.c_str());
}

JNIEXPORT jlong JNICALL
Java_com_n3ttx_aicore_engine_LlamaNative_loadModel(JNIEnv *env, jobject, jstring jpath, jint n_ctx,
                                                   jint n_batch, jint n_threads, jboolean embeddings) {
    const char *path = env->GetStringUTFChars(jpath, nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(jpath, path);
    if (model == nullptr) {
        g_last_error = "Failed to load model file";
        return 0;
    }

    auto *e = new Engine();
    e->model = model;
    e->vocab = llama_model_get_vocab(model);
    e->embeddings = embeddings;
    e->n_threads = n_threads;

    llama_context_params cparams = llama_context_default_params();
    const int32_t n_ctx_train = llama_model_n_ctx_train(model);
    uint32_t ctx_size = n_ctx > 0 ? static_cast<uint32_t>(n_ctx) : 4096u;
    if (n_ctx_train > 0) ctx_size = std::min(ctx_size, static_cast<uint32_t>(n_ctx_train));
    cparams.n_ctx = ctx_size;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.abort_callback = abort_cb;
    cparams.abort_callback_data = e;
    if (embeddings) {
        // Non-causal embedding models must see the whole input in one micro-batch.
        cparams.embeddings = true;
        cparams.n_batch = ctx_size;
        cparams.n_ubatch = ctx_size;
    } else {
        cparams.n_batch = static_cast<uint32_t>(n_batch > 0 ? n_batch : 512);
    }

    e->ctx = llama_init_from_model(model, cparams);
    if (e->ctx == nullptr) {
        g_last_error = "Failed to create context (not enough memory?)";
        llama_model_free(model);
        delete e;
        return 0;
    }
    LOGI("Loaded model, n_ctx=%u embeddings=%d", llama_n_ctx(e->ctx), embeddings ? 1 : 0);
    return reinterpret_cast<jlong>(e);
}

JNIEXPORT void JNICALL
Java_com_n3ttx_aicore_engine_LlamaNative_free(JNIEnv *, jobject, jlong handle) {
    Engine *e = as_engine(handle);
    if (e == nullptr) return;
    if (e->mtmd != nullptr) mtmd_free(e->mtmd);
    if (e->ctx != nullptr) llama_free(e->ctx);
    if (e->model != nullptr) llama_model_free(e->model);
    delete e;
}

JNIEXPORT void JNICALL
Java_com_n3ttx_aicore_engine_LlamaNative_cancel(JNIEnv *, jobject, jlong handle) {
    Engine *e = as_engine(handle);
    if (e != nullptr) e->cancel.store(true);
}

// Returns "key=value" lines describing the loaded model.
JNIEXPORT jstring JNICALL
Java_com_n3ttx_aicore_engine_LlamaNative_info(JNIEnv *env, jobject, jlong handle) {
    Engine *e = as_engine(handle);
    char desc[256] = {0};
    llama_model_desc(e->model, desc, sizeof(desc));
    std::string out;
    out += "description=" + std::string(desc) + "\n";
    out += "n_ctx=" + std::to_string(llama_n_ctx(e->ctx)) + "\n";
    out += "n_ctx_train=" + std::to_string(llama_model_n_ctx_train(e->model)) + "\n";
    out += "n_embd=" + std::to_string(llama_model_n_embd_out(e->model)) + "\n";
    out += "n_params=" + std::to_string(llama_model_n_params(e->model)) + "\n";
    out += "size=" + std::to_string(llama_model_size(e->model)) + "\n";
    out += "has_chat_template=" + std::string(llama_model_chat_template(e->model, nullptr) != nullptr ? "1" : "0") + "\n";
    return env->NewStringUTF(out.c_str());
}

JNIEXPORT jbyteArray JNICALL
Java_com_n3ttx_aicore_engine_LlamaNative_applyChatTemplate(JNIEnv *env, jobject, jlong handle, jstring joverride,
                                                           jobjectArray jroles, jobjectArray jcontents,
                                                           jboolean add_assistant) {
    Engine *e = as_engine(handle);
    // An override is a built-in template name such as "chatml" or "gemma".
    std::string override_name;
    if (joverride != nullptr) {
        const char *o = env->GetStringUTFChars(joverride, nullptr);
        override_name = o;
        env->ReleaseStringUTFChars(joverride, o);
    }
    const char *tmpl = override_name.empty() ? llama_model_chat_template(e->model, nullptr) : override_name.c_str();
    if (tmpl == nullptr) {
        g_last_error = "Model has no chat template; set one in the model settings";
        return nullptr;
    }
    const jsize n = env->GetArrayLength(jroles);
    std::vector<std::string> roles(static_cast<size_t>(n));
    std::vector<std::string> contents(static_cast<size_t>(n));
    std::vector<llama_chat_message> msgs(static_cast<size_t>(n));
    size_t total = 0;
    for (jsize i = 0; i < n; ++i) {
        auto r = static_cast<jbyteArray>(env->GetObjectArrayElement(jroles, i));
        auto c = static_cast<jbyteArray>(env->GetObjectArrayElement(jcontents, i));
        roles[i] = to_string(env, r);
        contents[i] = to_string(env, c);
        env->DeleteLocalRef(r);
        env->DeleteLocalRef(c);
        total += roles[i].size() + contents[i].size();
    }
    for (jsize i = 0; i < n; ++i) msgs[i] = {roles[i].c_str(), contents[i].c_str()};

    std::vector<char> buf(total * 2 + 1024);
    int32_t len = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_assistant, buf.data(),
                                            static_cast<int32_t>(buf.size()));
    if (len > static_cast<int32_t>(buf.size())) {
        buf.resize(static_cast<size_t>(len));
        len = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_assistant, buf.data(),
                                        static_cast<int32_t>(buf.size()));
    }
    if (len < 0) {
        g_last_error = "Chat template of this model is not supported; set a built-in template in the model settings";
        return nullptr;
    }
    return to_bytes(env, std::string(buf.data(), static_cast<size_t>(len)));
}

JNIEXPORT jint JNICALL
Java_com_n3ttx_aicore_engine_LlamaNative_tokenCount(JNIEnv *env, jobject, jlong handle, jbyteArray jtext) {
    Engine *e = as_engine(handle);
    return static_cast<jint>(tokenize(e->vocab, to_string(env, jtext), true, true).size());
}

JNIEXPORT jintArray JNICALL
Java_com_n3ttx_aicore_engine_LlamaNative_tokenize(JNIEnv *env, jobject, jlong handle, jbyteArray jtext) {
    Engine *e = as_engine(handle);
    const std::vector<llama_token> tokens = tokenize(e->vocab, to_string(env, jtext), true, true);
    jintArray arr = env->NewIntArray(static_cast<jsize>(tokens.size()));
    if (arr != nullptr && !tokens.empty()) {
        env->SetIntArrayRegion(arr, 0, static_cast<jsize>(tokens.size()), reinterpret_cast<const jint *>(tokens.data()));
    }
    return arr;
}

// stats (out): [prompt tokens, generated tokens, reused cached tokens]
JNIEXPORT jint JNICALL
Java_com_n3ttx_aicore_engine_LlamaNative_generate(JNIEnv *env, jobject, jlong handle, jbyteArray jprompt,
                                                  jint max_tokens, jfloat temperature, jint top_k, jfloat top_p,
                                                  jfloat min_p, jfloat repeat_penalty, jint seed, jstring jgrammar,
                                                  jobject sink, jintArray jstats) {
    Engine *e = as_engine(handle);
    e->cancel.store(false);

    jclass sink_cls = env->GetObjectClass(sink);
    jmethodID on_piece = env->GetMethodID(sink_cls, "onPiece", "([B)Z");
    if (on_piece == nullptr) {
        g_last_error = "TokenSink.onPiece not found";
        return ERR_GENERIC;
    }

    const std::string prompt = to_string(env, jprompt);
    std::vector<llama_token> tokens = tokenize(e->vocab, prompt, true, true);
    if (tokens.empty()) {
        g_last_error = "Prompt is empty";
        return ERR_GENERIC;
    }
    const int32_t n_ctx = static_cast<int32_t>(llama_n_ctx(e->ctx));
    if (static_cast<int32_t>(tokens.size()) >= n_ctx - 4) {
        g_last_error = "Prompt has " + std::to_string(tokens.size()) + " tokens, context is " + std::to_string(n_ctx);
        return ERR_CONTEXT_OVERFLOW;
    }

    // Reuse the KV cache for the common prefix with the previous request.
    size_t n_keep = 0;
    while (n_keep < e->cached.size() && n_keep < tokens.size() && e->cached[n_keep] == tokens[n_keep]) ++n_keep;
    if (n_keep == tokens.size()) --n_keep;  // at least one token must be decoded to get logits
    llama_memory_t mem = llama_get_memory(e->ctx);
    if (mem != nullptr && !llama_memory_seq_rm(mem, 0, static_cast<llama_pos>(n_keep), -1)) {
        llama_memory_clear(mem, true);
        n_keep = 0;
    }
    e->cached.assign(tokens.begin(), tokens.begin() + static_cast<long>(n_keep));
    const int cached_reused = static_cast<int>(n_keep);

    int rc = decode_tokens(e, tokens, n_keep, static_cast<llama_pos>(n_keep));
    if (rc == 2 || e->cancel.load()) {
        clear_memory(e);
        return STOP_CANCELLED;
    }
    if (rc != 0) {
        clear_memory(e);
        g_last_error = "llama_decode failed on prompt: " + std::to_string(rc);
        return ERR_GENERIC;
    }
    e->cached = tokens;

    llama_sampler *smpl = make_sampler(env, e, temperature, top_k, top_p, min_p, repeat_penalty, seed, jgrammar);
    if (smpl == nullptr) return ERR_GENERIC;

    const int32_t room = n_ctx - static_cast<int32_t>(tokens.size());
    const int32_t limit = max_tokens > 0 ? std::min<int32_t>(max_tokens, room) : room;
    int generated = 0;
    const int result = sample_loop(env, e, sink, on_piece, smpl, limit, static_cast<llama_pos>(e->cached.size()), true, generated);
    llama_sampler_free(smpl);

    jint stats[3] = {static_cast<jint>(tokens.size()), generated, cached_reused};
    env->SetIntArrayRegion(jstats, 0, 3, stats);
    return result;
}

// Loads the vision projector (mmproj GGUF) of the model. Returns false (see lastError) if it does not fit the model.
JNIEXPORT jboolean JNICALL
Java_com_n3ttx_aicore_engine_LlamaNative_loadProjector(JNIEnv *env, jobject, jlong handle, jstring jpath) {
    Engine *e = as_engine(handle);
    if (e->mtmd != nullptr) return JNI_TRUE;
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = false;
    params.print_timings = false;
    params.warmup = false;
    params.n_threads = e->n_threads;
    params.media_marker = mtmd_default_marker();
    e->mtmd = mtmd_init_from_file(path, e->model, params);
    env->ReleaseStringUTFChars(jpath, path);
    if (e->mtmd == nullptr) {
        g_last_error = "Failed to load the vision projector";
        return JNI_FALSE;
    }
    if (!mtmd_support_vision(e->mtmd)) {
        mtmd_free(e->mtmd);
        e->mtmd = nullptr;
        g_last_error = "The projector does not support images";
        return JNI_FALSE;
    }
    LOGI("Loaded vision projector");
    return JNI_TRUE;
}

// Like generate, but the prompt contains one media marker per image. Images are RGB bytes (width * height * 3),
// dims holds width and height of each. The KV cache is not reused across such requests.
// stats (out): [prompt tokens, generated tokens, reused cached tokens]
JNIEXPORT jint JNICALL
Java_com_n3ttx_aicore_engine_LlamaNative_generateWithImages(JNIEnv *env, jobject, jlong handle, jbyteArray jprompt,
                                                            jobjectArray jimages, jintArray jdims, jint max_tokens,
                                                            jfloat temperature, jint top_k, jfloat top_p, jfloat min_p,
                                                            jfloat repeat_penalty, jint seed, jstring jgrammar,
                                                            jobject sink, jintArray jstats) {
    Engine *e = as_engine(handle);
    e->cancel.store(false);
    if (e->mtmd == nullptr) {
        g_last_error = "No vision projector loaded";
        return ERR_GENERIC;
    }
    jclass sink_cls = env->GetObjectClass(sink);
    jmethodID on_piece = env->GetMethodID(sink_cls, "onPiece", "([B)Z");
    if (on_piece == nullptr) {
        g_last_error = "TokenSink.onPiece not found";
        return ERR_GENERIC;
    }

    const jsize n_images = env->GetArrayLength(jimages);
    std::vector<jint> dims(static_cast<size_t>(n_images) * 2);
    if (env->GetArrayLength(jdims) != n_images * 2) {
        g_last_error = "Image sizes missing";
        return ERR_GENERIC;
    }
    if (n_images > 0) env->GetIntArrayRegion(jdims, 0, n_images * 2, dims.data());
    std::vector<mtmd_bitmap *> bitmaps;
    for (jsize i = 0; i < n_images; ++i) {
        auto arr = static_cast<jbyteArray>(env->GetObjectArrayElement(jimages, i));
        const uint32_t w = static_cast<uint32_t>(dims[i * 2]);
        const uint32_t h = static_cast<uint32_t>(dims[i * 2 + 1]);
        const jsize len = env->GetArrayLength(arr);
        if (static_cast<size_t>(len) != static_cast<size_t>(w) * h * 3) {
            env->DeleteLocalRef(arr);
            for (auto *b : bitmaps) mtmd_bitmap_free(b);
            g_last_error = "Image " + std::to_string(i) + " has the wrong size";
            return ERR_GENERIC;
        }
        jbyte *data = env->GetByteArrayElements(arr, nullptr);
        bitmaps.push_back(mtmd_bitmap_init(w, h, reinterpret_cast<const unsigned char *>(data)));
        env->ReleaseByteArrayElements(arr, data, JNI_ABORT);
        env->DeleteLocalRef(arr);
    }

    const std::string prompt = to_string(env, jprompt);
    mtmd_input_text text;
    text.text = prompt.c_str();
    text.text_len = prompt.size();
    text.add_special = true;
    text.parse_special = true;
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    const int32_t trc = mtmd_tokenize(e->mtmd, chunks, &text, bitmaps.data(), bitmaps.size());
    for (auto *b : bitmaps) mtmd_bitmap_free(b);
    if (trc != 0) {
        mtmd_input_chunks_free(chunks);
        g_last_error = trc == 1 ? "Number of images does not match the media markers in the prompt" : "Image could not be prepared";
        return ERR_GENERIC;
    }

    const int32_t n_ctx = static_cast<int32_t>(llama_n_ctx(e->ctx));
    const auto n_prompt = static_cast<int32_t>(mtmd_helper_get_n_tokens(chunks));
    if (n_prompt >= n_ctx - 4) {
        mtmd_input_chunks_free(chunks);
        g_last_error = "Prompt with images has " + std::to_string(n_prompt) + " tokens, context is " + std::to_string(n_ctx);
        return ERR_CONTEXT_OVERFLOW;
    }

    clear_memory(e);
    llama_pos n_past = 0;
    const int32_t erc = mtmd_helper_eval_chunks(e->mtmd, e->ctx, chunks, 0, 0,
                                                static_cast<int32_t>(llama_n_batch(e->ctx)), true, &n_past);
    mtmd_input_chunks_free(chunks);
    if (e->cancel.load()) {
        clear_memory(e);
        return STOP_CANCELLED;
    }
    if (erc != 0) {
        clear_memory(e);
        g_last_error = "Evaluating the prompt with images failed: " + std::to_string(erc);
        return ERR_GENERIC;
    }

    llama_sampler *smpl = make_sampler(env, e, temperature, top_k, top_p, min_p, repeat_penalty, seed, jgrammar);
    if (smpl == nullptr) {
        clear_memory(e);
        return ERR_GENERIC;
    }
    const int32_t room = n_ctx - n_prompt;
    const int32_t limit = max_tokens > 0 ? std::min<int32_t>(max_tokens, room) : room;
    int generated = 0;
    const int result = sample_loop(env, e, sink, on_piece, smpl, limit, n_past, false, generated);
    llama_sampler_free(smpl);
    // The cache holds image embeddings that a text prompt cannot match: start the next request from scratch.
    clear_memory(e);

    jint stats[3] = {n_prompt, generated, 0};
    env->SetIntArrayRegion(jstats, 0, 3, stats);
    return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_n3ttx_aicore_engine_LlamaNative_embed(JNIEnv *env, jobject, jlong handle, jbyteArray jtext,
                                               jboolean normalize) {
    Engine *e = as_engine(handle);
    e->cancel.store(false);
    if (!e->embeddings) {
        g_last_error = "Model was not loaded as embedding model";
        return nullptr;
    }
    std::vector<llama_token> tokens = tokenize(e->vocab, to_string(env, jtext), true, true);
    const size_t max_tokens = llama_n_batch(e->ctx);
    if (tokens.size() > max_tokens) tokens.resize(max_tokens);  // truncate overly long input
    if (tokens.empty()) {
        g_last_error = "Text is empty";
        return nullptr;
    }

    clear_memory(e);
    const int32_t n = static_cast<int32_t>(tokens.size());
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    int rc;
    if (llama_model_has_encoder(e->model) && !llama_model_has_decoder(e->model)) {
        rc = llama_encode(e->ctx, batch);
    } else {
        rc = llama_decode(e->ctx, batch);
    }
    llama_batch_free(batch);
    if (rc != 0) {
        g_last_error = "Embedding failed: " + std::to_string(rc);
        clear_memory(e);
        return nullptr;
    }

    const int32_t n_embd = llama_model_n_embd_out(e->model);
    std::vector<float> out(static_cast<size_t>(n_embd), 0.0f);
    if (llama_pooling_type(e->ctx) == LLAMA_POOLING_TYPE_NONE) {
        // Mean pooling over all token embeddings.
        for (int32_t i = 0; i < n; ++i) {
            const float *v = llama_get_embeddings_ith(e->ctx, i);
            if (v == nullptr) continue;
            for (int32_t d = 0; d < n_embd; ++d) out[d] += v[d] / static_cast<float>(n);
        }
    } else {
        const float *v = llama_get_embeddings_seq(e->ctx, 0);
        if (v == nullptr) {
            g_last_error = "No pooled embedding returned";
            clear_memory(e);
            return nullptr;
        }
        std::copy(v, v + n_embd, out.begin());
    }
    clear_memory(e);

    if (normalize) {
        double sum = 0.0;
        for (float f : out) sum += static_cast<double>(f) * f;
        const double norm = std::sqrt(sum);
        if (norm > 0.0) {
            for (float &f : out) f = static_cast<float>(f / norm);
        }
    }

    jfloatArray arr = env->NewFloatArray(n_embd);
    env->SetFloatArrayRegion(arr, 0, n_embd, out.data());
    return arr;
}

}  // extern "C"
