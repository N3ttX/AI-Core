package com.n3ttx.aicore.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.n3ttx.aicore.BuildConfig
import com.n3ttx.aicore.graph
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

@Composable
fun SettingsScreen() {
    val g = LocalContext.current.graph
    val s by g.settings.settings.collectAsStateWithLifecycle()
    var catalogUrl by remember { mutableStateOf(s.catalogUrl) }
    val cores = Runtime.getRuntime().availableProcessors()

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text("Settings", style = MaterialTheme.typography.headlineSmall)
        Hint("Changes apply the next time a model is loaded.")

        SectionTitle("Performance")
        SliderSetting("Context length", "${s.contextLength} tokens", s.contextLength.toFloat(), 1024f..32768f, 30) {
            g.settings.update { st -> st.copy(contextLength = (it / 1024).roundToInt().coerceAtLeast(1) * 1024) }
        }
        SliderSetting("CPU threads", "${s.threads} of $cores", s.threads.toFloat(), 1f..cores.toFloat(), cores - 2) {
            g.settings.update { st -> st.copy(threads = it.roundToInt()) }
        }

        SectionTitle("Memory")
        SliderSetting("Memory budget for models", "${s.memoryBudgetMb / 1024.0} GB", s.memoryBudgetMb.toFloat(), 2048f..14336f, 23) {
            g.settings.update { st -> st.copy(memoryBudgetMb = ((it / 512).roundToInt() * 512)) }
        }
        SliderSetting(
            "Unload idle models after",
            if (s.idleUnloadMinutes == 0) "never" else "${s.idleUnloadMinutes} min",
            s.idleUnloadMinutes.toFloat(), 0f..60f, 11,
        ) {
            g.settings.update { st -> st.copy(idleUnloadMinutes = (it / 5).roundToInt() * 5) }
        }
        OutlinedButton(onClick = { g.scope.launch { g.runtime.unloadAll() } }) { Text("Unload all models now") }

        SectionTitle("Output")
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Hide reasoning")
                Hint("Removes <think>…</think> blocks of reasoning models from the output.")
            }
            Switch(checked = s.hideThinking, onCheckedChange = { v -> g.settings.update { it.copy(hideThinking = v) } })
        }

        SectionTitle("Catalog")
        OutlinedTextField(
            value = catalogUrl,
            onValueChange = { catalogUrl = it },
            label = { Text("Catalog URL") },
            modifier = Modifier.fillMaxWidth(),
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { g.settings.update { it.copy(catalogUrl = catalogUrl.trim()) } }) { Text("Save") }
            OutlinedButton(onClick = {
                catalogUrl = com.n3ttx.aicore.data.CoreSettings.DEFAULT_CATALOG_URL
                g.settings.update { it.copy(catalogUrl = catalogUrl) }
            }) { Text("Reset") }
        }

        SectionTitle("Privacy")
        Hint(
            "AI Core Hub has no analytics, no ads and no Google Play Services. It only connects to the internet " +
                "to download models and the catalog. Prompts and results never leave this device.",
        )
        Hint("Version ${BuildConfig.VERSION_NAME}", Modifier.padding(top = 16.dp))
    }
}

@Composable
private fun SliderSetting(
    title: String,
    value: String,
    current: Float,
    range: ClosedFloatingPointRange<Float>,
    steps: Int,
    onChange: (Float) -> Unit,
) {
    var pos by remember(current) { mutableStateOf(current) }
    Column {
        Row {
            Text(title, Modifier.weight(1f))
            Text(value, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary)
        }
        Slider(
            value = pos,
            onValueChange = { pos = it },
            onValueChangeFinished = { onChange(pos) },
            valueRange = range,
            steps = steps.coerceAtLeast(0),
        )
    }
}
