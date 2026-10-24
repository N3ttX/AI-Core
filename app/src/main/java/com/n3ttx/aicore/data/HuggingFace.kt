package com.n3ttx.aicore.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

data class HubFile(val repo: String, val path: String, val sizeBytes: Long) {
    val downloadUrl: String
        get() = "https://huggingface.co/$repo/resolve/main/" + path.split('/').joinToString("/") { URLEncoder.encode(it, "UTF-8").replace("+", "%20") }
}

/** Minimal anonymous Hugging Face Hub client to list model files (GGUF, ONNX) of a public repository. */
object HuggingFace {
    private val REPO = Regex("^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

    /** Accepts "owner/repo" or a huggingface.co URL. */
    fun parseRepo(input: String): String? {
        val trimmed = input.trim().removePrefix("https://").removePrefix("http://").removePrefix("huggingface.co/")
        val repo = trimmed.split('/').take(2).joinToString("/")
        return repo.takeIf { REPO.matches(it) }
    }

    suspend fun listModelFiles(repo: String): List<HubFile> = withContext(Dispatchers.IO) {
        val conn = URL("https://huggingface.co/api/models/$repo/tree/main?recursive=true").openConnection() as HttpURLConnection
        conn.connectTimeout = 15_000
        conn.readTimeout = 30_000
        try {
            when (conn.responseCode) {
                200 -> Unit
                401, 403 -> error("Repository is private or gated")
                404 -> error("Repository not found")
                else -> error("Hugging Face answered ${conn.responseCode}")
            }
            val a = JSONArray(conn.inputStream.bufferedReader().use { it.readText() })
            List(a.length()) { a.getJSONObject(it) }
                .filter {
                    val path = it.optString("path")
                    it.optString("type") == "file" && (path.endsWith(".gguf", ignoreCase = true) || path.endsWith(".onnx", ignoreCase = true))
                }
                // Skip vision projectors and drafter files, which are not standalone models.
                .filterNot { Regex("mmproj|mtp-", RegexOption.IGNORE_CASE).containsMatchIn(it.optString("path")) }
                .map { HubFile(repo, it.getString("path"), it.optJSONObject("lfs")?.optLong("size") ?: it.optLong("size")) }
                .sortedBy { it.sizeBytes }
        } finally {
            conn.disconnect()
        }
    }
}
