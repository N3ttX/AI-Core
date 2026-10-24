package com.n3ttx.aicore

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.n3ttx.aicore.ui.AICoreTheme
import com.n3ttx.aicore.ui.HomeScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AICoreTheme {
                HomeScreen()
            }
        }
    }
}
