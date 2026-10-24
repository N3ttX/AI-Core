package com.n3ttx.aicore.sdk;

import android.os.Bundle;
import com.n3ttx.aicore.sdk.IEmbeddingCallback;
import com.n3ttx.aicore.sdk.IGenerationCallback;

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
}
