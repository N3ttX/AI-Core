@file:OptIn(ExperimentalMaterial3Api::class)

package com.n3ttx.aicore.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.n3ttx.aicore.data.ModelSpec
import com.n3ttx.aicore.graph
import com.n3ttx.aicore.sdk.AICoreClient
import com.n3ttx.aicore.sdk.ChatMessage
import com.n3ttx.aicore.sdk.GenerationEvent
import com.n3ttx.aicore.sdk.GenerationOptions
import com.n3ttx.aicore.sdk.GenerationRequest
import com.n3ttx.aicore.sdk.AICoreContract.ModelType
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.util.Locale
import kotlin.math.sqrt

/**
 * Tries the installed models through the same client SDK and binder service
 * that other apps use, so it also tests the IPC path end to end.
 */
@Composable
fun PlaygroundScreen() {
    val context = LocalContext.current
    val client = remember { AICoreClient(context) }
    DisposableEffect(Unit) { onDispose { client.close() } }
    var mode by remember { mutableStateOf(0) }

    Column(Modifier.fillMaxSize().padding(horizontal = 16.dp)) {
        Text("Playground", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.padding(top = 16.dp))
        SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth().padding(vertical = 8.dp)) {
            listOf("Chat", "Embeddings").forEachIndexed { i, label ->
                SegmentedButton(selected = mode == i, onClick = { mode = i }, shape = SegmentedButtonDefaults.itemShape(i, 2)) {
                    Text(label)
                }
            }
        }
        if (mode == 0) ChatPlayground(client) else EmbeddingPlayground(client)
    }
}

@Composable
private fun ModelPicker(type: String, selected: String?, onSelect: (String) -> Unit) {
    val g = LocalContext.current.graph
    val installed by g.store.installed.collectAsStateWithLifecycle()
    val options = installed.filter { it.type == type }
    val current = options.firstOrNull { it.id == selected } ?: g.defaultModel(type)
    var open by remember { mutableStateOf(false) }
    LaunchedEffect(current?.id) { current?.id?.let(onSelect) }
    Box {
        OutlinedButton(onClick = { open = true }, enabled = options.isNotEmpty()) {
            Text(current?.name ?: "No ${typeLabel(type).lowercase(Locale.ROOT)} model installed")
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            options.forEach { m: ModelSpec ->
                DropdownMenuItem(text = { Text(m.name) }, onClick = { onSelect(m.id); open = false })
            }
        }
    }
}

@Composable
private fun ColumnScopeSpacer() = Box(Modifier.padding(4.dp))

@Composable
private fun ChatPlayground(client: AICoreClient) {
    val scope = rememberCoroutineScope()
    val messages = remember { mutableStateListOf<ChatMessage>() }
    var input by remember { mutableStateOf("") }
    var system by remember { mutableStateOf("You are a helpful assistant. Answer concisely.") }
    var model by remember { mutableStateOf<String?>(null) }
    var json by remember { mutableStateOf(false) }
    var streaming by remember { mutableStateOf("") }
    var job by remember { mutableStateOf<Job?>(null) }
    var stats by remember { mutableStateOf<String?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    val listState = rememberLazyListState()

    Column(Modifier.fillMaxSize()) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ModelPicker(ModelType.CHAT, model) { model = it }
            FilterChip(selected = json, onClick = { json = !json }, label = { Text("JSON") })
            TextButton(onClick = { messages.clear(); stats = null; error = null }) { Text("Clear") }
        }
        OutlinedTextField(
            value = system, onValueChange = { system = it },
            label = { Text("System prompt") }, maxLines = 3,
            modifier = Modifier.fillMaxWidth(),
        )
        LazyColumn(Modifier.weight(1f).fillMaxWidth(), state = listState, verticalArrangement = Arrangement.spacedBy(6.dp)) {
            itemsIndexed(messages) { _, m -> MessageBubble(m.role, m.content) }
            if (job != null) item { MessageBubble("assistant", streaming.ifEmpty { "…" }) }
            stats?.let { item { Hint(it) } }
            error?.let { item { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) } }
        }
        LaunchedEffect(messages.size, streaming.length) {
            val count = listState.layoutInfo.totalItemsCount
            if (count > 0) listState.scrollToItem(count - 1)
        }
        Row(Modifier.padding(vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(
                value = input, onValueChange = { input = it },
                placeholder = { Text("Message") }, maxLines = 4,
                modifier = Modifier.weight(1f),
            )
            if (job == null) {
                Button(
                    enabled = input.isNotBlank() && model != null,
                    onClick = {
                        messages.add(ChatMessage.user(input.trim()))
                        input = ""
                        error = null
                        stats = null
                        streaming = ""
                        val history = listOf(ChatMessage.system(system)) + messages
                        job = scope.launch {
                            try {
                                client.generate(
                                    GenerationRequest(history, model = model, options = GenerationOptions(maxTokens = 1024, jsonMode = json)),
                                ).collect { ev ->
                                    when (ev) {
                                        is GenerationEvent.Token -> streaming += ev.text
                                        is GenerationEvent.Completed -> {
                                            val r = ev.result
                                            messages.add(ChatMessage.assistant(r.text))
                                            val secs = r.durationMs / 1000.0
                                            stats = String.format(
                                                Locale.ROOT, "%d prompt tokens (%d cached) · %d generated · %.1f s · %.1f tok/s · %s",
                                                r.promptTokens, r.cachedTokens, r.completionTokens, secs,
                                                if (secs > 0) r.completionTokens / secs else 0.0, r.stopReason,
                                            )
                                        }
                                    }
                                }
                            } catch (e: kotlinx.coroutines.CancellationException) {
                                if (streaming.isNotEmpty()) messages.add(ChatMessage.assistant(streaming))
                                throw e
                            } catch (e: Exception) {
                                error = e.message
                            } finally {
                                job = null
                            }
                        }
                    },
                    modifier = Modifier.padding(start = 8.dp),
                ) { Text("Send") }
            } else {
                OutlinedButton(onClick = { job?.cancel() }, modifier = Modifier.padding(start = 8.dp)) { Text("Stop") }
            }
        }
    }
}

@Composable
private fun MessageBubble(role: String, text: String) {
    val mine = role == "user"
    Row(Modifier.fillMaxWidth(), horizontalArrangement = if (mine) Arrangement.End else Arrangement.Start) {
        Card(
            colors = if (mine) CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
            else CardDefaults.cardColors(),
            modifier = Modifier.fillMaxWidth(0.85f),
        ) {
            Text(text, Modifier.padding(10.dp), style = MaterialTheme.typography.bodyMedium)
        }
    }
}

@Composable
private fun EmbeddingPlayground(client: AICoreClient) {
    val scope = rememberCoroutineScope()
    var model by remember { mutableStateOf<String?>(null) }
    var a by remember { mutableStateOf("The cat sits on the mat.") }
    var b by remember { mutableStateOf("Eine Katze liegt auf dem Teppich.") }
    var c by remember { mutableStateOf("Quarterly revenue grew by 12 percent.") }
    var result by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        ModelPicker(ModelType.EMBEDDING, model) { model = it }
        Hint("Compares the meaning of three texts. Try the same sentence in different languages.")
        OutlinedTextField(a, { a = it }, label = { Text("Text A") }, modifier = Modifier.fillMaxWidth())
        OutlinedTextField(b, { b = it }, label = { Text("Text B") }, modifier = Modifier.fillMaxWidth())
        OutlinedTextField(c, { c = it }, label = { Text("Text C") }, modifier = Modifier.fillMaxWidth())
        Button(enabled = !busy && model != null, onClick = {
            busy = true
            scope.launch {
                result = try {
                    val start = System.currentTimeMillis()
                    val r = client.embed(listOf(a, b, c), model = model)
                    val v = r.vectors
                    String.format(
                        Locale.ROOT,
                        "Dimension %d · %d ms\nA ↔ B: %.3f\nA ↔ C: %.3f\nB ↔ C: %.3f",
                        r.dimension, System.currentTimeMillis() - start,
                        cosine(v[0], v[1]), cosine(v[0], v[2]), cosine(v[1], v[2]),
                    )
                } catch (e: Exception) {
                    "Error: ${e.message}"
                }
                busy = false
            }
        }) { Text(if (busy) "Embedding…" else "Compare") }
        result?.let { Text(it, style = MaterialTheme.typography.bodyLarge) }
        ColumnScopeSpacer()
    }
}

private fun cosine(x: FloatArray, y: FloatArray): Double {
    if (x.size != y.size || x.isEmpty()) return 0.0
    var dot = 0.0
    var nx = 0.0
    var ny = 0.0
    for (i in x.indices) {
        dot += x[i] * y[i]
        nx += x[i] * x[i]
        ny += y[i] * y[i]
    }
    return if (nx == 0.0 || ny == 0.0) 0.0 else dot / (sqrt(nx) * sqrt(ny))
}
