package com.n3ttx.aicore.data

import android.content.Context
import android.content.Intent
import android.util.Log
import com.n3ttx.aicore.service.DownloadService
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

data class DownloadState(
    val spec: ModelSpec,
    val downloadedBytes: Long = 0,
    val totalBytes: Long = spec.totalBytes,
    val status: Status = Status.QUEUED,
    val error: String? = null,
) {
    enum class Status { QUEUED, RUNNING, FAILED }

    val progress: Float get() = if (totalBytes > 0) (downloadedBytes.toFloat() / totalBytes).coerceIn(0f, 1f) else 0f
}

/**
 * Queue of model downloads. The work runs inside [DownloadService] (a foreground
 * service) so it survives the app going to the background. Downloads resume
 * from a ".part" file after interruptions.
 */
class DownloadController(private val context: Context, private val store: ModelStore) {
    private val state = MutableStateFlow<Map<String, DownloadState>>(emptyMap())
    val downloads: StateFlow<Map<String, DownloadState>> = state.asStateFlow()

    @Volatile
    private var cancelled: Set<String> = emptySet()

    /** Called when a download finished successfully. */
    var onInstalled: ((ModelSpec) -> Unit)? = null

    fun enqueue(spec: ModelSpec) {
        require(spec.url != null || spec.isPackage) { "Model has no download URL" }
        if (store.get(spec.id) != null) return
        cancelled = cancelled - spec.id
        state.update { current ->
            val existing = current[spec.id]
            if (existing != null && existing.status != DownloadState.Status.FAILED) current
            else current + (spec.id to DownloadState(spec))
        }
        context.startForegroundService(Intent(context, DownloadService::class.java))
    }

    fun cancel(id: String) {
        cancelled = cancelled + id
        val spec = state.value[id]?.spec
        state.update { it - id }
        File(store.modelsDir, partName(id)).delete()
        if (spec?.isPackage == true && store.get(id) == null) store.dirOf(spec).deleteRecursively()
    }

    fun dismiss(id: String) {
        state.update { current -> if (current[id]?.status == DownloadState.Status.FAILED) current - id else current }
    }

    fun nextQueued(): DownloadState? = state.value.values.firstOrNull { it.status == DownloadState.Status.QUEUED }

    fun hasPending(): Boolean = state.value.values.any { it.status != DownloadState.Status.FAILED }

    private fun partName(id: String) = "$id.part"

    private fun set(id: String, transform: (DownloadState) -> DownloadState) {
        state.update { current -> current[id]?.let { current + (id to transform(it)) } ?: current }
    }

    /** Runs one download to completion. Called by [DownloadService]. */
    suspend fun run(item: DownloadState) = withContext(Dispatchers.IO) {
        val spec = item.spec
        set(spec.id) { it.copy(status = DownloadState.Status.RUNNING, error = null) }
        if (spec.isPackage) {
            runPackage(spec)
            return@withContext
        }
        val part = File(store.modelsDir, partName(spec.id))
        try {
            download(spec.id, requireNotNull(spec.url), part, 0L, spec.sizeBytes)
            if (spec.id in cancelled) return@withContext
            if (!ModelStore.isGguf(part)) throw IOException("Downloaded file is not a GGUF model")
            spec.sha256?.let { expected ->
                val actual = sha256(part)
                if (!actual.equals(expected, ignoreCase = true)) {
                    part.delete()
                    throw IOException("Checksum mismatch")
                }
            }
            val target = store.fileOf(spec)
            if (!part.renameTo(target)) throw IOException("Could not move file into place")
            val installed = spec.copy(sizeBytes = target.length(), installedAt = System.currentTimeMillis())
            store.add(installed)
            state.update { it - spec.id }
            onInstalled?.invoke(installed)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            if (spec.id in cancelled) return@withContext
            Log.w(TAG, "Download of ${spec.id} failed", e)
            set(spec.id) { it.copy(status = DownloadState.Status.FAILED, error = e.message ?: e.javaClass.simpleName) }
        }
    }

    /** Downloads every file of a package into its folder; finished files are kept for resuming. */
    private suspend fun runPackage(spec: ModelSpec) {
        val dir = store.dirOf(spec).apply { mkdirs() }
        try {
            val total = spec.files.sumOf { it.sizeBytes }
            set(spec.id) { it.copy(totalBytes = total) }
            var before = 0L
            for (file in spec.files) {
                if (spec.id in cancelled) return
                val target = store.fileOf(spec, file)
                if (!(target.exists() && file.sizeBytes > 0 && target.length() == file.sizeBytes)) {
                    val part = File(dir, file.localName + ".part")
                    download(spec.id, file.url(spec.repo), part, before, total)
                    if (spec.id in cancelled) return
                    if (!part.renameTo(target)) throw IOException("Could not move ${file.localName} into place")
                }
                before += target.length()
                set(spec.id) { it.copy(downloadedBytes = before) }
            }
            val installed = spec.copy(
                files = spec.files.map { it.copy(sizeBytes = store.fileOf(spec, it).length()) },
                installedAt = System.currentTimeMillis(),
            )
            store.add(installed)
            state.update { it - spec.id }
            onInstalled?.invoke(installed)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            if (spec.id in cancelled) return
            Log.w(TAG, "Download of ${spec.id} failed", e)
            set(spec.id) { it.copy(status = DownloadState.Status.FAILED, error = e.message ?: e.javaClass.simpleName) }
        }
    }

    /**
     * Downloads [url] into [part], resuming if it exists. [before] are bytes of earlier files of the
     * same package and [total] the size of the whole package, for the progress display.
     */
    private suspend fun download(id: String, url: String, part: File, before: Long, total: Long) {
        var attempt = 0
        while (true) {
            try {
                downloadOnce(id, url, part, before, total)
                return
            } catch (e: IOException) {
                if (++attempt >= 4 || id in cancelled) throw e
                Log.i(TAG, "Retrying $id after: ${e.message}")
                kotlinx.coroutines.delay(2000L * attempt)
            }
        }
    }

    private suspend fun downloadOnce(id: String, url: String, part: File, before: Long, totalHint: Long) {
        val offset = if (part.exists()) part.length() else 0L
        val conn = URL(url).openConnection() as HttpURLConnection
        conn.instanceFollowRedirects = true
        conn.connectTimeout = 20_000
        conn.readTimeout = 60_000
        conn.setRequestProperty("User-Agent", "AICoreHub/1.0")
        if (offset > 0) conn.setRequestProperty("Range", "bytes=$offset-")
        try {
            val code = conn.responseCode
            val append = when (code) {
                206 -> true
                200 -> false
                416 -> return // already complete
                401, 403 -> throw IOException("Access denied (model may be gated or private)")
                404 -> throw IOException("File not found at URL")
                else -> throw IOException("Server answered $code")
            }
            val length = conn.getHeaderField("Content-Length")?.toLongOrNull() ?: -1L
            val start = if (append) offset else 0L
            val total = if (before == 0L && length >= 0 && totalHint <= 0) start + length else maxOf(totalHint, before + start + maxOf(length, 0))
            set(id) { it.copy(downloadedBytes = before + start, totalBytes = total) }

            conn.inputStream.use { input ->
                FileOutputStream(part, append).use { out ->
                    val buf = ByteArray(1 shl 16)
                    var done = start
                    var lastReport = 0L
                    while (true) {
                        currentCoroutineContext().ensureActive()
                        if (id in cancelled) return
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        done += n
                        if (done - lastReport > (1 shl 20)) {
                            lastReport = done
                            set(id) { it.copy(downloadedBytes = before + done) }
                        }
                    }
                    set(id) { it.copy(downloadedBytes = before + done) }
                    if (length >= 0 && done < start + length) throw IOException("Connection closed early")
                }
            }
        } finally {
            conn.disconnect()
        }
    }

    private fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(1 shl 20)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    companion object {
        private const val TAG = "Downloads"
    }
}
