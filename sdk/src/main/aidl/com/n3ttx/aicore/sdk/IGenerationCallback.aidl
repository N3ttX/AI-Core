package com.n3ttx.aicore.sdk;

import android.os.Bundle;

oneway interface IGenerationCallback {
    void onToken(String requestId, String text) = 1;
    void onComplete(String requestId, in Bundle result) = 2;
    void onError(String requestId, int code, String message) = 3;
}
