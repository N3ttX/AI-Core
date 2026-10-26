package com.n3ttx.aicore.runtime

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import com.n3ttx.aicore.MainActivity
import com.n3ttx.aicore.R
import com.n3ttx.aicore.data.SettingsStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.sample
import kotlinx.coroutines.launch

/**
 * Debug option: an ongoing notification that shows live which models are in memory and what each one is doing for which
 * app, with the waiting requests. Updated at most twice a second, and every second while a request runs (its time).
 */
class DebugNotification(
    private val context: Context,
    private val runtime: ModelRuntime,
    private val settings: SettingsStore,
    private val scope: CoroutineScope,
) {
    private val nm = context.getSystemService(NotificationManager::class.java)

    @OptIn(kotlinx.coroutines.FlowPreview::class)
    fun start() {
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL, "Debug: live model status", NotificationManager.IMPORTANCE_LOW).apply {
                description = "Which models are loaded and what they are doing (Settings → Debug)"
                setShowBadge(false)
            }
        )
        val ticker = flow { while (true) { emit(System.currentTimeMillis()); delay(1000) } }
        scope.launch {
            combine(
                settings.settings.map { it.debugNotification }.distinctUntilChanged(),
                runtime.loadedModels, runtime.active, runtime.queuedCount, runtime.waitingBackground,
            ) { on, loaded, active, queued, waiting -> if (on) Snapshot(loaded, active, queued, waiting) else null }
                .combine(runtime.lastUsed) { s, used -> s?.also { it.lastUsed = used } }
                .combine(runtime.loading) { s, loading -> s?.also { it.loading = loading } }
                .combine(ticker) { snap, now -> snap?.let { it to now } }
                .sample(500)
                .collect { s -> if (s == null) nm.cancel(ID) else nm.notify(ID, build(s.first, s.second)) }
        }
    }

    private class Snapshot(val loaded: List<LoadedModelInfo>, val active: ActiveRequest?, val queued: Int, val waiting: Int) {
        var lastUsed: Map<String, Long> = emptyMap()
        var loading: String? = null
    }

    private fun build(s: Snapshot, now: Long): Notification {
        val totalMb = s.loaded.sumOf { it.sizeBytes } / (1024 * 1024)
        val title = if (s.loaded.isEmpty()) "AI Core Hub · no model in memory" else "AI Core Hub · ${s.loaded.size} in memory · $totalMb MB"
        val activeName = s.active?.let { a -> s.loaded.firstOrNull { it.id == a.modelId }?.name ?: a.modelId }
        val summary = s.active?.let { a -> "${a.kind}: $activeName for ${a.clientLabel} · ${secs(now - a.startedAt)}" }
            ?: s.loading?.let { "Loading $it …" } ?: "Idle"
        val lines = buildList {
            s.loaded.forEach { m ->
                val busy = s.active?.takeIf { it.modelId == m.id }
                add(
                    if (busy != null) "● ${m.name} – ${busy.kind} for ${busy.clientLabel} (${secs(now - busy.startedAt)})"
                    else "○ ${m.name} – idle ${secs(now - (s.lastUsed[m.id] ?: m.lastUsedAt))} · ${m.sizeBytes / (1024 * 1024)} MB"
                )
            }
            s.loading?.let { add("◐ $it – loading into memory …") }
            if (s.queued > 0) add("Queue: ${s.queued} waiting")
            if (s.waiting > 0) add("Background: ${s.waiting} paused for interactive requests")
        }
        val open = PendingIntent.getActivity(
            context, ID, Intent(context, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
            PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Builder(context, CHANNEL)
            .setSmallIcon(R.drawable.ic_stat_core)
            .setContentTitle(title)
            .setContentText(summary)
            .setStyle(Notification.BigTextStyle().bigText((listOf(summary) + lines).joinToString("\n")))
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setShowWhen(false)
            .setContentIntent(open)
            .build()
    }

    private fun secs(ms: Long): String {
        val s = (ms / 1000).coerceAtLeast(0)
        return if (s < 60) "${s}s" else "${s / 60}m ${s % 60}s"
    }

    private companion object {
        const val CHANNEL = "debug"
        const val ID = 4711
    }
}
