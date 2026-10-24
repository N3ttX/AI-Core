package com.n3ttx.aicore.data

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale

/** A model file on disk, or one that is about to be downloaded. */
data class ModelSpec(
    val id: String,
    val name: String,
    val type: String,
    val url: String?,
    val fileName: String,
    val sizeBytes: Long = 0,
    val sha256: String? = null,
    val languages: List<String> = emptyList(),
    val contextLength: Int = 0,
    val chatTemplate: String? = null,
    val assistantPrefix: String? = null,
    val queryPrefix: String? = null,
    val documentPrefix: String? = null,
    val installedAt: Long = 0,
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id)
        put("name", name)
        put("type", type)
        url?.let { put("url", it) }
        put("file_name", fileName)
        put("size_bytes", sizeBytes)
        sha256?.let { put("sha256", it) }
        put("languages", JSONArray(languages))
        put("context_length", contextLength)
        chatTemplate?.let { put("chat_template", it) }
        assistantPrefix?.let { put("assistant_prefix", it) }
        queryPrefix?.let { put("query_prefix", it) }
        documentPrefix?.let { put("document_prefix", it) }
        put("installed_at", installedAt)
    }

    companion object {
        fun fromJson(o: JSONObject) = ModelSpec(
            id = o.getString("id"),
            name = o.getString("name"),
            type = o.getString("type"),
            url = o.optStringOrNull("url"),
            fileName = o.getString("file_name"),
            sizeBytes = o.optLong("size_bytes"),
            sha256 = o.optStringOrNull("sha256"),
            languages = o.optJSONArray("languages")?.let { a -> List(a.length()) { a.getString(it) } }.orEmpty(),
            contextLength = o.optInt("context_length"),
            chatTemplate = o.optStringOrNull("chat_template"),
            assistantPrefix = o.optStringOrNull("assistant_prefix"),
            queryPrefix = o.optStringOrNull("query_prefix"),
            documentPrefix = o.optStringOrNull("document_prefix"),
            installedAt = o.optLong("installed_at"),
        )

        fun fromCatalog(e: CatalogEntry) = ModelSpec(
            id = e.id,
            name = e.name,
            type = e.type,
            url = e.url,
            fileName = e.fileName,
            sizeBytes = e.sizeBytes,
            sha256 = e.sha256,
            languages = e.languages,
            contextLength = e.contextLength,
            chatTemplate = e.chatTemplate,
            assistantPrefix = e.assistantPrefix,
            queryPrefix = e.queryPrefix,
            documentPrefix = e.documentPrefix,
        )

        /** Stable id for a model outside the catalog, derived from its file name. */
        fun customId(fileName: String): String =
            "custom-" + fileName.removeSuffix(".gguf").lowercase(Locale.ROOT).replace(Regex("[^a-z0-9.]+"), "-").trim('-')

        /** Guesses the model type from its name. */
        fun guessType(name: String): String =
            if (Regex("embed|e5|bge|minilm|gte|nomic", RegexOption.IGNORE_CASE).containsMatchIn(name)) "embedding" else "chat"
    }
}

/** Registry of installed models (filesDir/models/registry.json) and their files. */
class ModelStore(private val context: Context) {
    val modelsDir: File = File(context.filesDir, "models").apply { mkdirs() }
    private val registryFile = File(modelsDir, "registry.json")
    private val state = MutableStateFlow(read())
    val installed: StateFlow<List<ModelSpec>> = state.asStateFlow()

    fun get(id: String): ModelSpec? = state.value.firstOrNull { it.id == id }

    fun fileOf(spec: ModelSpec): File = File(modelsDir, spec.fileName)

    private fun read(): List<ModelSpec> = runCatching {
        val a = JSONArray(registryFile.readText())
        List(a.length()) { ModelSpec.fromJson(a.getJSONObject(it)) }
    }.getOrDefault(emptyList()).filter { File(modelsDir, it.fileName).exists() }

    @Synchronized
    private fun write(list: List<ModelSpec>) {
        val tmp = File(modelsDir, "registry.json.tmp")
        tmp.writeText(JSONArray(list.map { it.toJson() }).toString(2))
        tmp.renameTo(registryFile)
        state.value = list
    }

    @Synchronized
    fun add(spec: ModelSpec) {
        write(state.value.filterNot { it.id == spec.id } + spec)
    }

    @Synchronized
    fun update(id: String, transform: (ModelSpec) -> ModelSpec) {
        write(state.value.map { if (it.id == id) transform(it) else it })
    }

    @Synchronized
    fun remove(id: String) {
        val spec = get(id) ?: return
        val stillUsed = state.value.any { it.id != id && it.fileName == spec.fileName }
        if (!stillUsed) fileOf(spec).delete()
        write(state.value.filterNot { it.id == id })
    }

    fun totalBytes(): Long = state.value.sumOf { fileOf(it).length() }

    /** Copies a GGUF file picked by the user into the model folder. */
    suspend fun importFromUri(uri: Uri, type: String?): ModelSpec = withContext(Dispatchers.IO) {
        val resolver = context.contentResolver
        val displayName = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        } ?: "model-${System.currentTimeMillis()}.gguf"
        val fileName = displayName.replace(Regex("[^A-Za-z0-9._-]"), "_").let { if (it.endsWith(".gguf")) it else "$it.gguf" }
        val target = File(modelsDir, fileName)
        val tmp = File(modelsDir, "$fileName.import")
        resolver.openInputStream(uri)?.use { input ->
            tmp.outputStream().use { input.copyTo(it, 1 shl 20) }
        } ?: error("Cannot open $uri")
        require(isGguf(tmp)) { tmp.delete(); "Not a GGUF file" }
        tmp.renameTo(target)
        val name = fileName.removeSuffix(".gguf")
        val spec = ModelSpec(
            id = ModelSpec.customId(fileName),
            name = name,
            type = type ?: ModelSpec.guessType(name),
            url = null,
            fileName = fileName,
            sizeBytes = target.length(),
            installedAt = System.currentTimeMillis(),
        )
        add(spec)
        spec
    }

    companion object {
        fun isGguf(file: File): Boolean = runCatching {
            file.inputStream().use { s ->
                val magic = ByteArray(4)
                s.read(magic) == 4 && String(magic, Charsets.US_ASCII) == "GGUF"
            }
        }.getOrDefault(false)
    }
}
