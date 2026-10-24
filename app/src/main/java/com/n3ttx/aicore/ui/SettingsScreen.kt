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

        SectionCard("Performance") {
            SliderSetting("Context length", "${s.contextLength} tokens", s.contextLength.toFloat(), 1024f..32768f, 30) {
                g.settings.update { st -> st.copy(contextLength = (it / 1024).roundToInt().coerceAtLeast(1) * 1024) }
            }
            RowDivider()
            SliderSetting("CPU threads", "${s.threads} of $cores", s.threads.toFloat(), 1f..cores.toFloat(), cores - 2) {
                g.settings.update { st -> st.copy(threads = it.roundToInt()) }
            }
            RowDivider()
            SwitchSetting(
                "XNNPACK for ONNX models",
                "Faster image and face models on some devices. Turn off if a model fails to load.",
                s.useXnnpack,
            ) { v -> g.settings.update { it.copy(useXnnpack = v) } }
        }

        SectionCard("Memory") {
            SliderSetting("Memory budget for models", "${s.memoryBudgetMb / 1024.0} GB", s.memoryBudgetMb.toFloat(), 2048f..14336f, 23) {
                g.settings.update { st -> st.copy(memoryBudgetMb = ((it / 512).roundToInt() * 512)) }
            }
            RowDivider()
            SliderSetting(
                "Unload idle models after",
                if (s.idleUnloadMinutes == 0) "never" else "${s.idleUnloadMinutes} min",
                s.idleUnloadMinutes.toFloat(), 0f..60f, 11,
            ) {
                g.settings.update { st -> st.copy(idleUnloadMinutes = (it / 5).roundToInt() * 5) }
            }
            RowDivider()
            OutlinedButton(onClick = { g.scope.launch { g.runtime.unloadAll() } }, modifier = Modifier.padding(16.dp)) {
                Text("Unload all models now")
            }
        }

        SectionCard("Output") {
            SwitchSetting("Hide reasoning", "Removes <think>…</think> blocks of reasoning models from the output.", s.hideThinking) { v ->
                g.settings.update { it.copy(hideThinking = v) }
            }
        }

        SectionCard("Catalog") {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
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
            }
        }

        SectionCard("About") {
            Hint(
                "AI Core Hub has no analytics, no ads and no Google Play Services. It only connects to the internet " +
                    "to download models and the catalog. Prompts, photos and results never leave this device.",
                Modifier.padding(16.dp),
            )
            RowDivider()
            InfoRow("Version", BuildConfig.VERSION_NAME)
            RowDivider()
            InfoRow("Engines", "llama.cpp · ONNX Runtime")
        }
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
    Column(Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
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

@Composable
private fun SwitchSetting(title: String, description: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(Modifier.padding(horizontal = 16.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f).padding(end = 12.dp)) {
            Text(title)
            Hint(description)
        }
        Switch(checked = checked, onCheckedChange = onChange)
    }
}
