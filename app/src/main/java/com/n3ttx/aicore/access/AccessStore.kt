package com.n3ttx.aicore.access

import android.content.Context
import android.content.pm.PackageManager
import com.n3ttx.aicore.sdk.AICoreContract.AccessState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

data class ClientApp(
    val packageName: String,
    val label: String,
    val state: String,
    /** SHA-256 of the signing certificates seen when the user decided. */
    val certDigest: String,
    val updatedAt: Long,
    val lastUsedAt: Long = 0,
    val requestCount: Long = 0,
) {
    fun toJson() = JSONObject().apply {
        put("package", packageName)
        put("label", label)
        put("state", state)
        put("cert", certDigest)
        put("updated_at", updatedAt)
        put("last_used_at", lastUsedAt)
        put("request_count", requestCount)
    }

    companion object {
        fun fromJson(o: JSONObject) = ClientApp(
            packageName = o.getString("package"),
            label = o.optString("label"),
            state = o.getString("state"),
            certDigest = o.optString("cert"),
            updatedAt = o.optLong("updated_at"),
            lastUsedAt = o.optLong("last_used_at"),
            requestCount = o.optLong("request_count"),
        )
    }
}

/**
 * Which apps may use AI Core Hub. An approval is bound to the app's signing
 * certificate, so a different app reinstalled under the same package name
 * has to be approved again.
 */
class AccessStore(private val context: Context) {
    private val file = File(context.filesDir, "clients.json")
    private val state = MutableStateFlow(read())
    val clients: StateFlow<List<ClientApp>> = state.asStateFlow()

    private fun read(): List<ClientApp> = runCatching {
        val a = JSONArray(file.readText())
        List(a.length()) { ClientApp.fromJson(a.getJSONObject(it)) }
    }.getOrDefault(emptyList())

    @Synchronized
    private fun write(list: List<ClientApp>) {
        file.writeText(JSONArray(list.map { it.toJson() }).toString())
        state.value = list
    }

    fun get(packageName: String): ClientApp? = state.value.firstOrNull { it.packageName == packageName }

    /** Current access state of [packageName], taking certificate changes into account. */
    fun stateOf(packageName: String): String? {
        val client = get(packageName) ?: return null
        if (client.state == AccessState.APPROVED && client.certDigest != certDigest(packageName)) return null
        return client.state
    }

    @Synchronized
    fun setState(packageName: String, newState: String) {
        val now = System.currentTimeMillis()
        val existing = get(packageName)
        val updated = (existing ?: ClientApp(packageName, labelOf(packageName), newState, "", now))
            .copy(state = newState, certDigest = certDigest(packageName), label = labelOf(packageName), updatedAt = now)
        write(state.value.filterNot { it.packageName == packageName } + updated)
    }

    /** Records a first contact from an unknown app. Returns true if it is new. */
    @Synchronized
    fun markPending(packageName: String): Boolean {
        if (get(packageName) != null && stateOf(packageName) != null) return false
        setState(packageName, AccessState.PENDING)
        return true
    }

    @Synchronized
    fun recordUse(packageName: String) {
        val c = get(packageName) ?: return
        write(state.value.map {
            if (it.packageName == packageName) c.copy(lastUsedAt = System.currentTimeMillis(), requestCount = c.requestCount + 1) else it
        })
    }

    @Synchronized
    fun forget(packageName: String) {
        write(state.value.filterNot { it.packageName == packageName })
    }

    fun labelOf(packageName: String): String = runCatching {
        val pm = context.packageManager
        pm.getApplicationLabel(pm.getApplicationInfo(packageName, 0)).toString()
    }.getOrDefault(packageName)

    fun certDigest(packageName: String): String = runCatching {
        val info = context.packageManager.getPackageInfo(packageName, PackageManager.GET_SIGNING_CERTIFICATES)
        val signers = info.signingInfo?.apkContentsSigners.orEmpty()
        val md = MessageDigest.getInstance("SHA-256")
        signers.map { it.toByteArray() }.sortedBy { it.contentHashCode() }.forEach { md.update(it) }
        md.digest().joinToString("") { "%02x".format(it) }
    }.getOrDefault("")
}
