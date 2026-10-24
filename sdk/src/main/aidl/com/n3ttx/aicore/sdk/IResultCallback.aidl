package com.n3ttx.aicore.sdk;

import android.os.Bundle;

/** Generic streaming result channel: one result per input item, then onComplete or onError. */
oneway interface IResultCallback {
    void onResult(String requestId, int index, in Bundle result) = 1;
    void onComplete(String requestId, in Bundle result) = 2;
    void onError(String requestId, int code, String message) = 3;
}
