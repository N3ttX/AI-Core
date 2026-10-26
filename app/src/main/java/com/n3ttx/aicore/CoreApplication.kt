package com.n3ttx.aicore

import android.app.Application
import android.content.Context
import com.n3ttx.aicore.access.AccessStore
import com.n3ttx.aicore.data.CatalogRepository
import com.n3ttx.aicore.data.DownloadController
import com.n3ttx.aicore.data.ModelRequestStore
import com.n3ttx.aicore.data.ModelSpec
import com.n3ttx.aicore.data.ModelStore
import com.n3ttx.aicore.data.SettingsStore
import com.n3ttx.aicore.runtime.ModelRuntime
import com.n3ttx.aicore.sdk.AICoreContract.ModelType
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

class CoreApplication : Application() {
    lateinit var graph: AppGraph
        private set

    override fun onCreate() {
        super.onCreate()
        graph = AppGraph(this)
    }
}

val Context.graph: AppGraph get() = (applicationContext as CoreApplication).graph

/** Process-wide singletons. */
class AppGraph(context: Context) {
    val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    val settings = SettingsStore(context)
    val catalog = CatalogRepository(context, settings)
    val store = ModelStore(context)
    val downloads = DownloadController(context, store)
    val access = AccessStore(context)
    val modelRequests = ModelRequestStore(context)
    val runtime = ModelRuntime(context, store, settings, scope)
    val notifier = Notifier(context)
    val debugNotification = com.n3ttx.aicore.runtime.DebugNotification(context, runtime, settings, scope)

    init {
        downloads.onInstalled = { spec ->
            modelRequests.removeModel(spec.id)
            ensureDefaults()
            notifier.modelInstalled(spec)
        }
        ensureDefaults()
        debugNotification.start()
    }

    /** The model used when a request does not name one. */
    fun defaultModel(type: String): ModelSpec? {
        val preferred = settings.current.defaults[type]
        return preferred?.let { store.get(it) }?.takeIf { it.type == type }
            ?: store.installed.value.firstOrNull { it.type == type }
    }

    /** Points the defaults at installed models when they are unset or were removed. */
    fun ensureDefaults() {
        settings.update { s ->
            s.copy(defaults = ModelType.ALL.mapNotNull { t -> defaultModel(t)?.let { t to it.id } }.toMap())
        }
    }

    fun removeModel(id: String) {
        scope.launch {
            runtime.unload(id)
            store.remove(id)
            ensureDefaults()
        }
    }
}
