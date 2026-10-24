package com.n3ttx.aicore.ui

import androidx.annotation.DrawableRes
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Badge
import androidx.compose.material3.BadgedBox
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.n3ttx.aicore.MainActivity
import com.n3ttx.aicore.R
import com.n3ttx.aicore.graph
import com.n3ttx.aicore.sdk.AICoreContract.AccessState

private data class Tab(val id: String, val label: String, @DrawableRes val icon: Int)

private val tabs = listOf(
    Tab(MainActivity.TAB_MODELS, "Models", R.drawable.ic_tab_models),
    Tab(MainActivity.TAB_APPS, "Apps", R.drawable.ic_tab_apps),
    Tab(MainActivity.TAB_PLAYGROUND, "Playground", R.drawable.ic_tab_playground),
    Tab(MainActivity.TAB_SETTINGS, "Settings", R.drawable.ic_tab_settings),
)

@Composable
fun MainScreen(tab: String, onTabChange: (String) -> Unit) {
    val g = LocalContext.current.graph
    val clients by g.access.clients.collectAsStateWithLifecycle()
    val requests by g.modelRequests.requests.collectAsStateWithLifecycle()
    val appsBadge = clients.count { it.state == AccessState.PENDING } + requests.size

    Scaffold(
        bottomBar = {
            NavigationBar {
                tabs.forEach { t ->
                    NavigationBarItem(
                        selected = tab == t.id,
                        onClick = { onTabChange(t.id) },
                        icon = {
                            BadgedBox(badge = { if (t.id == MainActivity.TAB_APPS && appsBadge > 0) Badge { Text("$appsBadge") } }) {
                                Icon(painterResource(t.icon), contentDescription = null)
                            }
                        },
                        label = { Text(t.label) },
                    )
                }
            }
        },
    ) { padding ->
        Box(Modifier.padding(padding)) {
            when (tab) {
                MainActivity.TAB_APPS -> AppsScreen()
                MainActivity.TAB_PLAYGROUND -> PlaygroundScreen()
                MainActivity.TAB_SETTINGS -> SettingsScreen()
                else -> ModelsScreen()
            }
        }
    }
}
