package com.n3ttx.aicore.data

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class CoreSettings(
    val contextLength: Int = 8192,
    val threads: Int = defaultThreads(),
    /** Upper bound for the summed size of loaded models. */
    val memoryBudgetMb: Int = 8192,
    val idleUnloadMinutes: Int = 10,
    val hideThinking: Boolean = true,
    val defaultChatModel: String? = null,
    val defaultEmbeddingModel: String? = null,
    val catalogUrl: String = DEFAULT_CATALOG_URL,
) {
    companion object {
        const val DEFAULT_CATALOG_URL =
            "https://raw.githubusercontent.com/N3ttX/AI-Core/main/app/src/main/assets/catalog.json"

        fun defaultThreads(): Int = (Runtime.getRuntime().availableProcessors() / 2).coerceIn(2, 6)
    }
}

class SettingsStore(context: Context) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val state = MutableStateFlow(read())
    val settings: StateFlow<CoreSettings> = state.asStateFlow()

    val current: CoreSettings get() = state.value

    private fun read(): CoreSettings {
        val d = CoreSettings()
        return CoreSettings(
            contextLength = prefs.getInt("context_length", d.contextLength),
            threads = prefs.getInt("threads", d.threads),
            memoryBudgetMb = prefs.getInt("memory_budget_mb", d.memoryBudgetMb),
            idleUnloadMinutes = prefs.getInt("idle_unload_minutes", d.idleUnloadMinutes),
            hideThinking = prefs.getBoolean("hide_thinking", d.hideThinking),
            defaultChatModel = prefs.getString("default_chat_model", null),
            defaultEmbeddingModel = prefs.getString("default_embedding_model", null),
            catalogUrl = prefs.getString("catalog_url", null) ?: d.catalogUrl,
        )
    }

    @Synchronized
    fun update(transform: (CoreSettings) -> CoreSettings) {
        val s = transform(state.value)
        prefs.edit()
            .putInt("context_length", s.contextLength)
            .putInt("threads", s.threads)
            .putInt("memory_budget_mb", s.memoryBudgetMb)
            .putInt("idle_unload_minutes", s.idleUnloadMinutes)
            .putBoolean("hide_thinking", s.hideThinking)
            .putString("default_chat_model", s.defaultChatModel)
            .putString("default_embedding_model", s.defaultEmbeddingModel)
            .putString("catalog_url", s.catalogUrl)
            .apply()
        state.value = s
    }
}
