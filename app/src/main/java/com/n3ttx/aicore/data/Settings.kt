package com.n3ttx.aicore.data

import android.content.Context
import com.n3ttx.aicore.sdk.AICoreContract.ModelType
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class CoreSettings(
    val contextLength: Int = 8192,
    val threads: Int = defaultThreads(),
    /** The user set [threads] by hand; otherwise the default follows the device (and later improvements of it). */
    val threadsCustom: Boolean = false,
    /** Upper bound for the summed size of loaded models. */
    val memoryBudgetMb: Int = 8192,
    val idleUnloadMinutes: Int = 10,
    val hideThinking: Boolean = true,
    /** XNNPACK execution provider for ONNX models. Faster for some models, slower for others. */
    val useXnnpack: Boolean = false,
    /** Default model id per [ModelType]. */
    val defaults: Map<String, String> = emptyMap(),
    val catalogUrl: String = DEFAULT_CATALOG_URL,
    /** Debug: an ongoing notification that shows live which models are loaded and what they are doing. */
    val debugNotification: Boolean = false,
) {
    val defaultChatModel: String? get() = defaults[ModelType.CHAT]
    val defaultEmbeddingModel: String? get() = defaults[ModelType.EMBEDDING]

    fun withDefault(type: String, id: String?): CoreSettings =
        copy(defaults = if (id == null) defaults - type else defaults + (type to id))

    companion object {
        const val DEFAULT_CATALOG_URL =
            "https://raw.githubusercontent.com/N3ttX/AI-Core/main/app/src/main/assets/catalog.json"

        /**
         * Phones have a few efficiency cores that slow a shared computation down: 8 cores (Pixel 10 Pro: 1 + 5 fast,
         * 2 efficiency) use 6 threads, smaller CPUs half of their cores.
         */
        fun defaultThreads(): Int {
            val cores = Runtime.getRuntime().availableProcessors()
            return if (cores >= 8) cores - 2 else (cores / 2).coerceAtLeast(2)
        }
    }
}

class SettingsStore(context: Context) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val state = MutableStateFlow(read())
    val settings: StateFlow<CoreSettings> = state.asStateFlow()

    val current: CoreSettings get() = state.value

    private fun key(type: String) = "default_${type}_model"

    private fun read(): CoreSettings {
        val d = CoreSettings()
        return CoreSettings(
            contextLength = prefs.getInt("context_length", d.contextLength),
            threadsCustom = prefs.getBoolean("threads_custom", false),
            threads = if (prefs.getBoolean("threads_custom", false)) prefs.getInt("threads", d.threads) else d.threads,
            memoryBudgetMb = prefs.getInt("memory_budget_mb", d.memoryBudgetMb),
            idleUnloadMinutes = prefs.getInt("idle_unload_minutes", d.idleUnloadMinutes),
            hideThinking = prefs.getBoolean("hide_thinking", d.hideThinking),
            useXnnpack = prefs.getBoolean("use_xnnpack", d.useXnnpack),
            defaults = ModelType.ALL.mapNotNull { t -> prefs.getString(key(t), null)?.let { t to it } }.toMap(),
            catalogUrl = prefs.getString("catalog_url", null) ?: d.catalogUrl,
            debugNotification = prefs.getBoolean("debug_notification", false),
        )
    }

    @Synchronized
    fun update(transform: (CoreSettings) -> CoreSettings) {
        val s = transform(state.value)
        val edit = prefs.edit()
            .putInt("context_length", s.contextLength)
            .putInt("threads", s.threads)
            .putBoolean("threads_custom", s.threadsCustom)
            .putBoolean("debug_notification", s.debugNotification)
            .putInt("memory_budget_mb", s.memoryBudgetMb)
            .putInt("idle_unload_minutes", s.idleUnloadMinutes)
            .putBoolean("hide_thinking", s.hideThinking)
            .putBoolean("use_xnnpack", s.useXnnpack)
            .putString("catalog_url", s.catalogUrl)
        ModelType.ALL.forEach { t -> edit.putString(key(t), s.defaults[t]) }
        edit.apply()
        state.value = s
    }
}
