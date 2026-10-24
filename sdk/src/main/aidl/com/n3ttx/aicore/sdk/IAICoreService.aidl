package com.n3ttx.aicore.sdk;

import android.os.Bundle;
import com.n3ttx.aicore.sdk.IEmbeddingCallback;
import com.n3ttx.aicore.sdk.IGenerationCallback;
import com.n3ttx.aicore.sdk.IResultCallback;

/**
 * Binder interface of AI Core Hub. All requests and results are Bundles so that
 * new keys can be added without breaking older clients. Keys are listed in
 * AICoreContract. Method ids are fixed; never renumber them.
 */
interface IAICoreService {
    int getApiVersion() = 1;

    /** Access state of the caller, loaded models and service version. */
    Bundle getStatus() = 2;

    /** Installed models plus the defaults for each model type. */
    Bundle listModels() = 3;

    /** Asks the user to install a model (catalog id or custom URL). */
    Bundle requestModel(in Bundle spec) = 4;

    /** Starts a text generation. Returns the request id or an error. */
    Bundle generate(in Bundle request, IGenerationCallback callback) = 5;

    /** Starts embedding a batch of texts. Returns the request id or an error. */
    Bundle embed(in Bundle request, IEmbeddingCallback callback) = 6;

    void cancel(String requestId) = 7;

    /** Counts tokens of a text or chat for a model, without generating. */
    Bundle countTokens(in Bundle request) = 8;

    // ---- API version 2 ----

    /** Embeds images (bitmaps or file descriptors) with an image-text model. One vector per image. */
    Bundle embedImages(in Bundle request, IEmbeddingCallback callback) = 9;

    /** Detects faces (box, landmarks, score) and optionally computes face embeddings. One result per image. */
    Bundle detectFaces(in Bundle request, IResultCallback callback) = 10;

    /** Runs one component of an installed ONNX model with raw tensors. */
    Bundle runModel(in Bundle request, IResultCallback callback) = 11;

    /** Supported tasks, engines and limits of this AI Core installation. */
    Bundle getCapabilities() = 12;

    /** Models AI Core recommends and can install on request, with their install state. */
    Bundle getCatalog() = 13;

    /** Token ids of a text with a model's tokenizer. */
    Bundle tokenize(in Bundle request) = 14;
}
