package com.n3ttx.aicore.sdk;

import android.os.Bundle;

oneway interface IEmbeddingCallback {
    /** One vector per input text, delivered in order. */
    void onEmbedding(String requestId, int index, in float[] vector) = 1;
    void onComplete(String requestId, in Bundle result) = 2;
    void onError(String requestId, int code, String message) = 3;
}
