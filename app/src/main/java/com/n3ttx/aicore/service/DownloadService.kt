package com.n3ttx.aicore.service

import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import com.n3ttx.aicore.Notifier
import com.n3ttx.aicore.graph
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

/** Foreground service that works through the download queue, then stops itself. */
class DownloadService : Service() {
    private var worker: Job? = null
    private var progressJob: Job? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val g = graph
        startForeground(
            Notifier.ID_DOWNLOAD,
            g.notifier.downloadProgress("Preparing download", "", 0, true),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
        )
        if (progressJob == null) {
            progressJob = g.scope.launch {
                g.downloads.downloads.collectLatest { map ->
                    val running = map.values.firstOrNull { it.status == com.n3ttx.aicore.data.DownloadState.Status.RUNNING }
                        ?: return@collectLatest
                    val queued = map.size - 1
                    val pct = (running.progress * 100).toInt()
                    val mb = running.downloadedBytes shr 20
                    val totalMb = running.totalBytes shr 20
                    val more = if (queued > 0) " · $queued more queued" else ""
                    g.notifier.update(
                        Notifier.ID_DOWNLOAD,
                        g.notifier.downloadProgress(running.spec.name, "$mb / $totalMb MB$more", pct, running.totalBytes <= 0),
                    )
                    delay(1000) // throttle notification updates
                }
            }
        }
        if (worker?.isActive != true) {
            worker = g.scope.launch {
                while (true) {
                    val next = g.downloads.nextQueued() ?: break
                    g.downloads.run(next)
                }
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    /** Android limits data sync services to a few hours per day. */
    override fun onTimeout(startId: Int, fgsType: Int) {
        stopSelf()
    }

    override fun onDestroy() {
        worker?.cancel()
        progressJob?.cancel()
        super.onDestroy()
    }
}
