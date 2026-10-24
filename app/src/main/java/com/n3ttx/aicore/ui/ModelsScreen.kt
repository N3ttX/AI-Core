@file:OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)

package com.n3ttx.aicore.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.n3ttx.aicore.AppGraph
import com.n3ttx.aicore.data.CatalogEntry
import com.n3ttx.aicore.data.DownloadState
import com.n3ttx.aicore.data.HubFile
import com.n3ttx.aicore.data.HuggingFace
import com.n3ttx.aicore.data.ModelSpec
import com.n3ttx.aicore.graph
import com.n3ttx.aicore.sdk.AICoreContract.ModelType
import kotlinx.coroutines.launch

@Composable
fun ModelsScreen() {
    val g = LocalContext.current.graph
    val installed by g.store.installed.collectAsStateWithLifecycle()
    val catalog by g.catalog.entries.collectAsStateWithLifecycle()
    val downloads by g.downloads.downloads.collectAsStateWithLifecycle()
    val loaded by g.runtime.loadedModels.collectAsStateWithLifecycle()
    val active by g.runtime.active.collectAsStateWithLifecycle()
    val settings by g.settings.settings.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()
    var typeFilter by remember { mutableStateOf<String?>(null) }
    var showAdd by remember { mutableStateOf(false) }
    var editing by remember { mutableStateOf<ModelSpec?>(null) }
    var removing by remember { mutableStateOf<ModelSpec?>(null) }
    var message by remember { mutableStateOf<String?>(null) }

    LazyColumn(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        item {
            Row(Modifier.fillMaxWidth().padding(top = 16.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("AI Core Hub", style = MaterialTheme.typography.headlineSmall)
                    Hint("${installed.size} models · ${formatBytes(installed.sumOf { g.store.fileOf(it).length() })} on disk")
                }
                Button(onClick = { showAdd = true }) { Text("Add model") }
            }
        }

        if (loaded.isNotEmpty()) {
            item { SectionTitle("In memory") }
            items(loaded, key = { "loaded-" + it.id }) { m ->
                Card(Modifier.fillMaxWidth()) {
                    Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(m.name, style = MaterialTheme.typography.bodyLarge)
                            val busy = active?.takeIf { it.modelId == m.id }
                            Hint(if (busy != null) "Working for ${busy.clientLabel} (${busy.kind})" else "Idle · ${formatBytes(m.sizeBytes)}")
                        }
                        TextButton(onClick = { scope.launch { g.runtime.unload(m.id) } }) { Text("Unload") }
                    }
                }
            }
        }

        if (downloads.isNotEmpty()) {
            item { SectionTitle("Downloads") }
            items(downloads.values.toList(), key = { "dl-" + it.spec.id }) { d -> DownloadCard(g, d) }
        }

        item { SectionTitle("Installed") }
        if (installed.isEmpty()) {
            item { Hint("No models yet. Download one from the catalog below or add your own.") }
        }
        items(installed, key = { "inst-" + it.id }) { m ->
            val isDefault = m.id == settings.defaultChatModel || m.id == settings.defaultEmbeddingModel
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(m.name, style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                        if (isDefault) AssistChip(onClick = {}, label = { Text("Default") })
                    }
                    Hint("${typeLabel(m.type)} · ${formatBytes(m.sizeBytes)} · id: ${m.id}")
                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        if (!isDefault) TextButton(onClick = {
                            g.settings.update {
                                if (m.type == ModelType.EMBEDDING) it.copy(defaultEmbeddingModel = m.id) else it.copy(defaultChatModel = m.id)
                            }
                        }) { Text("Make default") }
                        TextButton(onClick = { editing = m }) { Text("Settings") }
                        TextButton(onClick = { removing = m }) { Text("Remove") }
                    }
                }
            }
        }

        item {
            Row(verticalAlignment = Alignment.CenterVertically) {
                SectionTitle("Catalog", Modifier.weight(1f))
                TextButton(onClick = { scope.launch { message = g.catalog.refresh() ?: "Catalog updated" } }) { Text("Refresh") }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(selected = typeFilter == null, onClick = { typeFilter = null }, label = { Text("All") })
                FilterChip(selected = typeFilter == ModelType.CHAT, onClick = { typeFilter = ModelType.CHAT }, label = { Text("Text") })
                FilterChip(selected = typeFilter == ModelType.EMBEDDING, onClick = { typeFilter = ModelType.EMBEDDING }, label = { Text("Embeddings") })
            }
        }
        val visible = catalog.filter { typeFilter == null || it.type == typeFilter }
            .sortedWith(compareByDescending<CatalogEntry> { it.recommended }.thenBy { it.sizeBytes })
        items(visible, key = { "cat-" + it.id }) { e ->
            CatalogCard(
                entry = e,
                installed = installed.any { it.id == e.id },
                downloading = downloads.containsKey(e.id),
                onDownload = { g.downloads.enqueue(ModelSpec.fromCatalog(e)) },
            )
        }
        item { Spacer(Modifier.padding(bottom = 24.dp)) }
    }

    if (showAdd) AddModelDialog(g, onDismiss = { showAdd = false }, onMessage = { message = it })
    editing?.let { m -> ModelSettingsDialog(g, m, onDismiss = { editing = null }) }
    removing?.let { m ->
        AlertDialog(
            onDismissRequest = { removing = null },
            title = { Text("Remove ${m.name}?") },
            text = { Text("The file (${formatBytes(m.sizeBytes)}) is deleted. Apps that use this model will need it downloaded again.") },
            confirmButton = { TextButton(onClick = { g.removeModel(m.id); removing = null }) { Text("Remove") } },
            dismissButton = { TextButton(onClick = { removing = null }) { Text("Cancel") } },
        )
    }
    message?.let { msg ->
        AlertDialog(
            onDismissRequest = { message = null },
            text = { Text(msg) },
            confirmButton = { TextButton(onClick = { message = null }) { Text("OK") } },
        )
    }
}

fun typeLabel(type: String) = if (type == ModelType.EMBEDDING) "Embeddings" else "Text generation"

@Composable
private fun DownloadCard(g: AppGraph, d: DownloadState) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(d.spec.name, style = MaterialTheme.typography.titleSmall)
            when (d.status) {
                DownloadState.Status.FAILED -> {
                    Text("Failed: ${d.error}", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                    Row {
                        TextButton(onClick = { g.downloads.enqueue(d.spec) }) { Text("Retry") }
                        TextButton(onClick = { g.downloads.cancel(d.spec.id) }) { Text("Discard") }
                    }
                }
                else -> {
                    if (d.status == DownloadState.Status.QUEUED) {
                        Hint("Queued")
                    } else {
                        LinearProgressIndicator(progress = { d.progress }, modifier = Modifier.fillMaxWidth())
                        Hint("${formatBytes(d.downloadedBytes)} of ${formatBytes(d.totalBytes)}")
                    }
                    TextButton(onClick = { g.downloads.cancel(d.spec.id) }) { Text("Cancel") }
                }
            }
        }
    }
}

@Composable
private fun CatalogCard(entry: CatalogEntry, installed: Boolean, downloading: Boolean, onDownload: () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(entry.name, style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                if (entry.recommended) AssistChip(onClick = {}, label = { Text("Recommended") })
            }
            Text(entry.description, style = MaterialTheme.typography.bodyMedium)
            Hint(
                listOf(
                    typeLabel(entry.type),
                    "~" + formatBytes(entry.sizeBytes),
                    entry.languages.joinToString(", "),
                    entry.license,
                ).filter { it.isNotBlank() }.joinToString(" · "),
            )
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                when {
                    installed -> Hint("Installed", Modifier.padding(8.dp))
                    downloading -> Hint("Downloading…", Modifier.padding(8.dp))
                    else -> OutlinedButton(onClick = onDownload) { Text("Download") }
                }
            }
        }
    }
}

@Composable
private fun TypeSelector(type: String, onChange: (String) -> Unit) {
    SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
        listOf(ModelType.CHAT to "Text", ModelType.EMBEDDING to "Embeddings").forEachIndexed { i, (t, label) ->
            SegmentedButton(
                selected = type == t,
                onClick = { onChange(t) },
                shape = SegmentedButtonDefaults.itemShape(i, 2),
            ) { Text(label) }
        }
    }
}

@Composable
private fun AddModelDialog(g: AppGraph, onDismiss: () -> Unit, onMessage: (String) -> Unit) {
    val scope = rememberCoroutineScope()
    var mode by remember { mutableStateOf(0) }
    var repoInput by remember { mutableStateOf("") }
    var files by remember { mutableStateOf<List<HubFile>>(emptyList()) }
    var loading by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    var url by remember { mutableStateOf("") }
    var name by remember { mutableStateOf("") }
    var type by remember { mutableStateOf(ModelType.CHAT) }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            onDismiss()
            // The dialog's scope ends with the dialog, so copy on the app scope.
            g.scope.launch {
                onMessage(
                    runCatching { g.store.importFromUri(uri, type) }
                        .fold({ g.ensureDefaults(); "Imported ${it.name}" }, { "Import failed: ${it.message}" }),
                )
            }
        }
    }

    fun enqueue(spec: ModelSpec) {
        if (g.store.get(spec.id) != null) {
            onMessage("${spec.name} is already installed")
        } else {
            g.downloads.enqueue(spec)
        }
        onDismiss()
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Add model") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                    listOf("Hugging Face", "URL", "File").forEachIndexed { i, label ->
                        SegmentedButton(selected = mode == i, onClick = { mode = i }, shape = SegmentedButtonDefaults.itemShape(i, 3)) {
                            Text(label)
                        }
                    }
                }
                TypeSelector(type) { type = it }
                when (mode) {
                    0 -> {
                        OutlinedTextField(
                            value = repoInput,
                            onValueChange = { repoInput = it },
                            label = { Text("Repository, e.g. unsloth/Qwen3.5-4B-GGUF") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth(),
                        )
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            OutlinedButton(enabled = !loading, onClick = {
                                val repo = HuggingFace.parseRepo(repoInput)
                                if (repo == null) {
                                    error = "Enter owner/repository"
                                } else {
                                    loading = true
                                    error = null
                                    scope.launch {
                                        runCatching { HuggingFace.listGgufFiles(repo) }
                                            .onSuccess { files = it; if (it.isEmpty()) error = "No GGUF files in this repository" }
                                            .onFailure { error = it.message }
                                        loading = false
                                    }
                                }
                            }) { Text("List files") }
                            if (loading) CircularProgressIndicator(Modifier.padding(start = 12.dp).size(24.dp))
                        }
                        LazyColumn(Modifier.heightIn(max = 280.dp)) {
                            items(files, key = { it.path }) { f ->
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Column(Modifier.weight(1f)) {
                                        Text(f.path.substringAfterLast('/'), style = MaterialTheme.typography.bodyMedium)
                                        Hint(formatBytes(f.sizeBytes))
                                    }
                                    TextButton(onClick = {
                                        val fileName = f.path.substringAfterLast('/')
                                        enqueue(
                                            ModelSpec(
                                                id = ModelSpec.customId(fileName),
                                                name = fileName.removeSuffix(".gguf"),
                                                type = type,
                                                url = f.downloadUrl,
                                                fileName = fileName,
                                                sizeBytes = f.sizeBytes,
                                            ),
                                        )
                                    }) { Text("Get") }
                                }
                            }
                        }
                    }
                    1 -> {
                        OutlinedTextField(
                            value = url, onValueChange = { url = it },
                            label = { Text("Direct https link to a .gguf file") },
                            modifier = Modifier.fillMaxWidth(),
                        )
                        OutlinedTextField(
                            value = name, onValueChange = { name = it },
                            label = { Text("Name (optional)") }, singleLine = true,
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }
                    else -> Hint("Pick a .gguf file from your device. It is copied into AI Core Hub's private storage.")
                }
                error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
            }
        },
        confirmButton = {
            when (mode) {
                1 -> TextButton(onClick = {
                    val clean = url.trim()
                    val fileName = clean.substringBefore('?').substringAfterLast('/')
                    if (!clean.startsWith("https://") || !fileName.endsWith(".gguf", ignoreCase = true)) {
                        error = "Needs an https link ending in .gguf"
                    } else {
                        enqueue(
                            ModelSpec(
                                id = ModelSpec.customId(fileName),
                                name = name.ifBlank { fileName.removeSuffix(".gguf") },
                                type = type,
                                url = clean,
                                fileName = fileName.replace(Regex("[^A-Za-z0-9._-]"), "_"),
                            ),
                        )
                    }
                }) { Text("Download") }
                2 -> TextButton(onClick = { picker.launch(arrayOf("*/*")) }) { Text("Choose file") }
                else -> {}
            }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Close") } },
    )
}

private val BUILT_IN_TEMPLATES = listOf(
    "", "chatml", "llama3", "gemma", "phi4", "mistral-v7", "deepseek3", "granite", "command-r", "zephyr", "vicuna",
)

@Composable
private fun ModelSettingsDialog(g: AppGraph, m: ModelSpec, onDismiss: () -> Unit) {
    var name by remember { mutableStateOf(m.name) }
    var type by remember { mutableStateOf(m.type) }
    var template by remember { mutableStateOf(m.chatTemplate.orEmpty()) }
    var prefix by remember { mutableStateOf(m.assistantPrefix.orEmpty()) }
    var queryPrefix by remember { mutableStateOf(m.queryPrefix.orEmpty()) }
    var docPrefix by remember { mutableStateOf(m.documentPrefix.orEmpty()) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Model settings") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(name, { name = it }, label = { Text("Name") }, singleLine = true, modifier = Modifier.fillMaxWidth())
                TypeSelector(type) { type = it }
                if (type == ModelType.CHAT) {
                    Hint("Chat template: empty uses the template stored in the file.")
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        BUILT_IN_TEMPLATES.forEach { t ->
                            FilterChip(selected = template == t, onClick = { template = t }, label = { Text(t.ifEmpty { "from file" }) })
                        }
                    }
                    OutlinedTextField(
                        prefix, { prefix = it },
                        label = { Text("Assistant prefix (advanced)") },
                        modifier = Modifier.fillMaxWidth(),
                    )
                } else {
                    OutlinedTextField(queryPrefix, { queryPrefix = it }, label = { Text("Query prefix") }, modifier = Modifier.fillMaxWidth())
                    OutlinedTextField(docPrefix, { docPrefix = it }, label = { Text("Document prefix") }, modifier = Modifier.fillMaxWidth())
                }
            }
        },
        confirmButton = {
            TextButton(onClick = {
                g.store.update(m.id) {
                    it.copy(
                        name = name.ifBlank { m.name },
                        type = type,
                        chatTemplate = template.ifBlank { null },
                        assistantPrefix = prefix.ifEmpty { null },
                        queryPrefix = queryPrefix.ifEmpty { null },
                        documentPrefix = docPrefix.ifEmpty { null },
                    )
                }
                // The load options depend on the type, so drop the loaded instance.
                if (type != m.type) g.scope.launch { g.runtime.unload(m.id) }
                g.ensureDefaults()
                onDismiss()
            }) { Text("Save") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}
