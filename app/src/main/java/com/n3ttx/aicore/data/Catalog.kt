package com.n3ttx.aicore.data

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

data class CatalogEntry(
    val id: String,
    val name: String,
    val type: String,
    val description: String,
    val url: String,
    val fileName: String,
    val sizeBytes: Long,
    val sha256: String?,
    val license: String,
    val languages: List<String>,
    val contextLength: Int,
    val recommended: Boolean,
    /** llama.cpp built-in template name to use instead of the one in the file. */
    val chatTemplate: String?,
    /** Text appended after the assistant turn start, e.g. to skip reasoning. */
    val assistantPrefix: String?,
    /** Prefix the client should add to search queries (embedding models). */
    val queryPrefix: String?,
    /** Prefix the client should add to documents (embedding models). */
    val documentPrefix: String?,
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id)
        put("name", name)
        put("type", type)
        put("description", description)
        put("url", url)
        put("file_name", fileName)
        put("size_bytes", sizeBytes)
        sha256?.let { put("sha256", it) }
        put("license", license)
        put("languages", JSONArray(languages))
        put("context_length", contextLength)
        put("recommended", recommended)
        chatTemplate?.let { put("chat_template", it) }
        assistantPrefix?.let { put("assistant_prefix", it) }
        queryPrefix?.let { put("query_prefix", it) }
        documentPrefix?.let { put("document_prefix", it) }
    }

    companion object {
        fun fromJson(o: JSONObject): CatalogEntry {
            val url = o.getString("url")
            return CatalogEntry(
                id = o.getString("id"),
                name = o.getString("name"),
                type = o.getString("type"),
                description = o.optString("description"),
                url = url,
                fileName = o.optString("file_name").ifEmpty { url.substringAfterLast('/').substringBefore('?') },
                sizeBytes = o.optLong("size_bytes"),
                sha256 = o.optStringOrNull("sha256"),
                license = o.optString("license"),
                languages = o.optJSONArray("languages")?.let { a -> List(a.length()) { a.getString(it) } }.orEmpty(),
                contextLength = o.optInt("context_length"),
                recommended = o.optBoolean("recommended"),
                chatTemplate = o.optStringOrNull("chat_template"),
                assistantPrefix = o.optStringOrNull("assistant_prefix"),
                queryPrefix = o.optStringOrNull("query_prefix"),
                documentPrefix = o.optStringOrNull("document_prefix"),
            )
        }
    }
}

fun JSONObject.optStringOrNull(key: String): String? =
    if (has(key) && !isNull(key)) getString(key).takeIf { it.isNotEmpty() } else null

/**
 * The list of recommended models. Ships with the app (assets/catalog.json) and can be
 * refreshed from a URL so the list can change without an app update.
 */
class CatalogRepository(private val context: Context, private val settings: SettingsStore) {
    private val cacheFile = File(context.filesDir, "catalog.json")
    private val state = MutableStateFlow(loadLocal())
    val entries: StateFlow<List<CatalogEntry>> = state.asStateFlow()

    fun find(id: String): CatalogEntry? = state.value.firstOrNull { it.id == id }

    private fun loadLocal(): List<CatalogEntry> {
        val cached = runCatching { if (cacheFile.exists()) parse(cacheFile.readText()) else null }.getOrNull()
        return cached ?: parse(context.assets.open("catalog.json").bufferedReader().use { it.readText() })
    }

    private fun parse(text: String): List<CatalogEntry> {
        val models = JSONObject(text).getJSONArray("models")
        return List(models.length()) { CatalogEntry.fromJson(models.getJSONObject(it)) }
    }

    /** Downloads the catalog from the configured URL. Returns an error message or null. */
    suspend fun refresh(): String? = withContext(Dispatchers.IO) {
        try {
            val conn = URL(settings.current.catalogUrl).openConnection() as HttpURLConnection
            conn.connectTimeout = 15_000
            conn.readTimeout = 15_000
            try {
                if (conn.responseCode != 200) return@withContext "Catalog server answered ${conn.responseCode}"
                val text = conn.inputStream.bufferedReader().use { it.readText() }
                val parsed = parse(text)
                cacheFile.writeText(text)
                state.value = parsed
                null
            } finally {
                conn.disconnect()
            }
        } catch (e: Exception) {
            Log.w("Catalog", "Refresh failed", e)
            e.message ?: e.javaClass.simpleName
        }
    }
}
