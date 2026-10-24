package com.n3ttx.aicore

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.mutableStateOf
import com.n3ttx.aicore.sdk.AICoreContract
import com.n3ttx.aicore.ui.AICoreTheme
import com.n3ttx.aicore.ui.MainScreen

class MainActivity : ComponentActivity() {
    private val tab = mutableStateOf(TAB_MODELS)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        handleIntent(intent)
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
        }
        setContent {
            AICoreTheme {
                MainScreen(tab = tab.value, onTabChange = { tab.value = it })
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: Intent?) {
        intent ?: return
        when {
            intent.action == AICoreContract.ACTION_OPEN_MODELS -> tab.value = TAB_MODELS
            intent.hasExtra(EXTRA_TAB) -> tab.value = intent.getStringExtra(EXTRA_TAB) ?: TAB_MODELS
        }
    }

    companion object {
        const val EXTRA_TAB = "tab"
        const val TAB_MODELS = "models"
        const val TAB_APPS = "apps"
        const val TAB_PLAYGROUND = "playground"
        const val TAB_SETTINGS = "settings"
    }
}
