package com.n3ttx.aicore

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import com.n3ttx.aicore.data.ModelSpec

class Notifier(private val context: Context) {
    private val nm = context.getSystemService(NotificationManager::class.java)

    init {
        nm.createNotificationChannels(
            listOf(
                NotificationChannel(CHANNEL_REQUESTS, "App requests", NotificationManager.IMPORTANCE_HIGH).apply {
                    description = "Apps asking for access or for a new model"
                },
                NotificationChannel(CHANNEL_DOWNLOADS, "Model downloads", NotificationManager.IMPORTANCE_LOW),
            )
        )
    }

    private fun openApp(tab: String): PendingIntent = PendingIntent.getActivity(
        context, tab.hashCode(),
        Intent(context, MainActivity::class.java).putExtra(MainActivity.EXTRA_TAB, tab)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP),
        PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
    )

    fun accessRequested(packageName: String, label: String) {
        nm.notify(
            packageName.hashCode(),
            Notification.Builder(context, CHANNEL_REQUESTS)
                .setSmallIcon(R.drawable.ic_stat_core)
                .setContentTitle("$label wants to use AI Core Hub")
                .setContentText("Tap to allow or deny")
                .setContentIntent(openApp(MainActivity.TAB_APPS))
                .setAutoCancel(true)
                .build(),
        )
    }

    fun cancelAccessRequest(packageName: String) = nm.cancel(packageName.hashCode())

    fun modelRequested(label: String, modelName: String) {
        nm.notify(
            ID_MODEL_REQUEST,
            Notification.Builder(context, CHANNEL_REQUESTS)
                .setSmallIcon(R.drawable.ic_stat_core)
                .setContentTitle("$label asks for a model")
                .setContentText("$modelName is not installed. Tap to review.")
                .setContentIntent(openApp(MainActivity.TAB_APPS))
                .setAutoCancel(true)
                .build(),
        )
    }

    fun modelInstalled(spec: ModelSpec) {
        nm.notify(
            spec.id.hashCode(),
            Notification.Builder(context, CHANNEL_DOWNLOADS)
                .setSmallIcon(R.drawable.ic_stat_core)
                .setContentTitle("${spec.name} is ready")
                .setContentIntent(openApp(MainActivity.TAB_MODELS))
                .setAutoCancel(true)
                .build(),
        )
    }

    fun downloadProgress(title: String, text: String, progress: Int, indeterminate: Boolean): Notification =
        Notification.Builder(context, CHANNEL_DOWNLOADS)
            .setSmallIcon(R.drawable.ic_stat_core)
            .setContentTitle(title)
            .setContentText(text)
            .setProgress(100, progress, indeterminate)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setContentIntent(openApp(MainActivity.TAB_MODELS))
            .build()

    fun update(id: Int, notification: Notification) = nm.notify(id, notification)

    companion object {
        const val CHANNEL_REQUESTS = "requests"
        const val CHANNEL_DOWNLOADS = "downloads"
        const val ID_MODEL_REQUEST = 2
        const val ID_DOWNLOAD = 1
    }
}
