package com.n3ttx.aicore.data

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** A client app asked for a model that is not installed. The user decides in the UI. */
data class ModelRequest(
    val spec: ModelSpec,
    val packageName: String,
    val appLabel: String,
    val reason: String?,
    val fromCatalog: Boolean,
    val requestedAt: Long,
) {
    fun toJson() = JSONObject().apply {
        put("spec", spec.toJson())
        put("package", packageName)
        put("label", appLabel)
        reason?.let { put("reason", it) }
        put("from_catalog", fromCatalog)
        put("requested_at", requestedAt)
    }

    companion object {
        fun fromJson(o: JSONObject) = ModelRequest(
            spec = ModelSpec.fromJson(o.getJSONObject("spec")),
            packageName = o.getString("package"),
            appLabel = o.optString("label"),
            reason = o.optStringOrNull("reason"),
            fromCatalog = o.optBoolean("from_catalog"),
            requestedAt = o.optLong("requested_at"),
        )
    }
}

class ModelRequestStore(context: Context) {
    private val file = File(context.filesDir, "model_requests.json")
    private val state = MutableStateFlow(read())
    val requests: StateFlow<List<ModelRequest>> = state.asStateFlow()

    private fun read(): List<ModelRequest> = runCatching {
        val a = JSONArray(file.readText())
        List(a.length()) { ModelRequest.fromJson(a.getJSONObject(it)) }
    }.getOrDefault(emptyList())

    @Synchronized
    private fun write(list: List<ModelRequest>) {
        file.writeText(JSONArray(list.map { it.toJson() }).toString())
        state.value = list
    }

    /** Returns false if the same app already asked for the same model. */
    @Synchronized
    fun add(request: ModelRequest): Boolean {
        if (state.value.any { it.spec.id == request.spec.id && it.packageName == request.packageName }) return false
        write(state.value + request)
        return true
    }

    @Synchronized
    fun removeModel(modelId: String) {
        write(state.value.filterNot { it.spec.id == modelId })
    }
}
