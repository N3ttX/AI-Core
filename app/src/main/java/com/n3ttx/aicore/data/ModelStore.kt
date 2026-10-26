package com.n3ttx.aicore.data

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import com.n3ttx.aicore.sdk.AICoreContract
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale

/** One file of a multi-file model package (e.g. an ONNX tower or a tokenizer). */
data class ModelFile(
    /**
     * What the file is for: "vision", "text", "tokenizer", "detection", "recognition", "model", or "data"; in a llama.cpp
     * package "model" (the GGUF) and "projector" (the mmproj GGUF that lets a vision language model see images).
     */
    val role: String,
    /** Path inside the Hugging Face repo, or a full https URL. */
    val path: String,
    val sizeBytes: Long = 0,
    /** File name on disk. ONNX external data files must keep the name the .onnx file refers to. */
    val localName: String = path.substringBefore('?').substringAfterLast('/'),
) {
    fun url(repo: String?): String =
        if (path.startsWith("https://")) path
        else "https://huggingface.co/${requireNotNull(repo) { "No repo for $path" }}/resolve/main/$path"

    fun toJson(): JSONObject = JSONObject().apply {
        put("role", role)
        put("path", path)
        put("size", sizeBytes)
        put("local_name", localName)
    }

    companion object {
        fun fromJson(o: JSONObject): ModelFile {
            val path = o.getString("path")
            return ModelFile(
                role = o.optString("role", "model"),
                path = path,
                sizeBytes = o.optLong("size"),
                localName = o.optStringOrNull("local_name") ?: path.substringBefore('?').substringAfterLast('/'),
            )
        }
    }
}

/**
 * A model on disk, or one that is about to be downloaded. llama.cpp models are one GGUF file
 * ([url], [fileName]); ONNX models are a package of [files] from [repo], stored in their own folder.
 */
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
    val engine: String = AICoreContract.Engine.LLAMA,
    val repo: String? = null,
    val files: List<ModelFile> = emptyList(),
    /** Model specific settings (preprocessing, thresholds) as JSON. */
    val config: String? = null,
    val description: String? = null,
) {
    val isPackage: Boolean get() = files.isNotEmpty()

    val totalBytes: Long get() = if (isPackage) files.sumOf { it.sizeBytes } else sizeBytes

    fun configJson(): JSONObject? = config?.let { runCatching { JSONObject(it) }.getOrNull() }

    fun toJson(): JSONObject = JSONObject().apply {
        put("engine", engine)
        repo?.let { put("repo", it) }
        if (files.isNotEmpty()) put("files", JSONArray(files.map { it.toJson() }))
        config?.let { put("config", JSONObject(it)) }
        description?.let { put("description", it) }
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
            engine = o.optString("engine", AICoreContract.Engine.LLAMA),
            repo = o.optStringOrNull("repo"),
            files = o.optJSONArray("files")?.let { a -> List(a.length()) { ModelFile.fromJson(a.getJSONObject(it)) } }.orEmpty(),
            config = o.optJSONObject("config")?.toString(),
            description = o.optStringOrNull("description"),
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
            engine = e.engine,
            repo = e.repo,
            files = e.files,
            config = e.config,
            description = e.description,
        )

        /** Stable id for a model outside the catalog, derived from its file name. */
        fun customId(fileName: String): String =
            "custom-" + fileName.removeSuffix(".gguf").lowercase(Locale.ROOT).replace(Regex("[^a-z0-9.]+"), "-").trim('-')

        /** Guesses the model type from its name. */
        fun guessType(name: String): String = when {
            name.endsWith(".onnx", ignoreCase = true) -> AICoreContract.ModelType.ONNX
            Regex("embed|e5|bge|minilm|gte|nomic", RegexOption.IGNORE_CASE).containsMatchIn(name) -> AICoreContract.ModelType.EMBEDDING
            else -> AICoreContract.ModelType.CHAT
        }
    }
}

/** Registry of installed models (filesDir/models/registry.json) and their files. */
class ModelStore(private val context: Context) {
    val modelsDir: File = File(context.filesDir, "models").apply { mkdirs() }
    private val registryFile = File(modelsDir, "registry.json")
    private val state = MutableStateFlow(read())
    val installed: StateFlow<List<ModelSpec>> = state.asStateFlow()

    fun get(id: String): ModelSpec? = state.value.firstOrNull { it.id == id }

    /** The GGUF file of a llama.cpp model, or the folder of a model package. */
    fun fileOf(spec: ModelSpec): File = if (spec.isPackage) dirOf(spec) else File(modelsDir, spec.fileName)

    fun dirOf(spec: ModelSpec): File = File(modelsDir, "pkg-" + spec.id.replace(Regex("[^A-Za-z0-9._-]"), "_"))

    fun fileOf(spec: ModelSpec, file: ModelFile): File = File(dirOf(spec), file.localName)

    /** The GGUF file of a llama.cpp model: the file itself, or the "model" file of a package (vision models). */
    fun ggufOf(spec: ModelSpec): File =
        if (spec.isPackage) componentFiles(spec)[ROLE_MODEL] ?: error("${spec.name} has no model file") else fileOf(spec)

    /** The vision projector of a llama.cpp package, or null if the model cannot see images. */
    fun projectorOf(spec: ModelSpec): File? =
        if (spec.engine == AICoreContract.Engine.LLAMA && spec.isPackage) componentFiles(spec)[ROLE_PROJECTOR] else null

    /** Role to file on disk, for ONNX packages. */
    fun componentFiles(spec: ModelSpec): Map<String, File> = spec.files.associate { it.role to fileOf(spec, it) }

    /** Bytes on disk. */
    fun sizeOnDisk(spec: ModelSpec): Long =
        if (spec.isPackage) dirOf(spec).listFiles()?.sumOf { it.length() } ?: 0L else fileOf(spec).length()

    private fun read(): List<ModelSpec> = runCatching {
        val a = JSONArray(registryFile.readText())
        List(a.length()) { ModelSpec.fromJson(a.getJSONObject(it)) }
    }.getOrDefault(emptyList()).filter { fileOf(it).exists() }

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
        if (spec.isPackage) {
            dirOf(spec).deleteRecursively()
        } else {
            val stillUsed = state.value.any { it.id != id && !it.isPackage && it.fileName == spec.fileName }
            if (!stillUsed) fileOf(spec).delete()
        }
        write(state.value.filterNot { it.id == id })
    }

    fun totalBytes(): Long = state.value.sumOf { sizeOnDisk(it) }

    /** Copies a GGUF file picked by the user into the model folder. */
    suspend fun importFromUri(uri: Uri, type: String?): ModelSpec = withContext(Dispatchers.IO) {
        val resolver = context.contentResolver
        val displayName = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        } ?: "model-${System.currentTimeMillis()}.gguf"
        val clean = displayName.replace(Regex("[^A-Za-z0-9._-]"), "_")
        val isOnnx = clean.endsWith(".onnx", ignoreCase = true)
        val fileName = if (isOnnx || clean.endsWith(".gguf")) clean else "$clean.gguf"
        val name = fileName.removeSuffix(".gguf").removeSuffix(".onnx")
        val id = ModelSpec.customId(fileName)
        val onnxFile = ModelFile(role = "model", path = fileName, localName = fileName)
        val draft = ModelSpec(
            id = id, name = name, type = AICoreContract.ModelType.ONNX, url = null, fileName = fileName,
            engine = AICoreContract.Engine.ONNX, files = listOf(onnxFile),
        )
        val target = if (isOnnx) fileOf(draft, onnxFile).also { it.parentFile?.mkdirs() } else File(modelsDir, fileName)
        val tmp = File(target.parentFile, "$fileName.import")
        resolver.openInputStream(uri)?.use { input ->
            tmp.outputStream().use { input.copyTo(it, 1 shl 20) }
        } ?: error("Cannot open $uri")
        if (!isOnnx) require(isGguf(tmp)) { tmp.delete(); "Not a GGUF or ONNX file" }
        tmp.renameTo(target)
        val spec = if (isOnnx) {
            draft.copy(files = listOf(onnxFile.copy(sizeBytes = target.length())), installedAt = System.currentTimeMillis())
        } else {
            ModelSpec(
                id = id,
                name = name,
                type = type ?: ModelSpec.guessType(name),
                url = null,
                fileName = fileName,
                sizeBytes = target.length(),
                installedAt = System.currentTimeMillis(),
            )
        }
        add(spec)
        spec
    }

    companion object {
        const val ROLE_MODEL = "model"
        const val ROLE_PROJECTOR = "projector"

        fun isGguf(file: File): Boolean = runCatching {
            file.inputStream().use { s ->
                val magic = ByteArray(4)
                s.read(magic) == 4 && String(magic, Charsets.US_ASCII) == "GGUF"
            }
        }.getOrDefault(false)
    }
}
