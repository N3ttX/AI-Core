@file:OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)

package com.n3ttx.aicore.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.horizontalScroll
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
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
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.n3ttx.aicore.AppGraph
import com.n3ttx.aicore.data.CatalogEntry
import com.n3ttx.aicore.data.DownloadState
import com.n3ttx.aicore.data.HubFile
import com.n3ttx.aicore.data.HuggingFace
import com.n3ttx.aicore.data.ModelFile
import com.n3ttx.aicore.data.ModelSpec
import com.n3ttx.aicore.graph
import com.n3ttx.aicore.sdk.AICoreContract
import com.n3ttx.aicore.sdk.AICoreContract.AccessState
import com.n3ttx.aicore.sdk.AICoreContract.ModelType
import kotlinx.coroutines.launch

private val TYPE_FILTERS = listOf(
    null to "All",
    ModelType.CHAT to "Text",
    ModelType.EMBEDDING to "Embeddings",
    ModelType.IMAGE_TEXT to "Images",
    ModelType.FACE to "Faces",
)

@Composable
fun ModelsScreen() {
    val g = LocalContext.current.graph
    val installed by g.store.installed.collectAsStateWithLifecycle()
    val catalog by g.catalog.entries.collectAsStateWithLifecycle()
    val downloads by g.downloads.downloads.collectAsStateWithLifecycle()
    val loaded by g.runtime.loadedModels.collectAsStateWithLifecycle()
    val active by g.runtime.active.collectAsStateWithLifecycle()
    val queued by g.runtime.queuedCount.collectAsStateWithLifecycle()
    val clients by g.access.clients.collectAsStateWithLifecycle()
    val settings by g.settings.settings.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()
    var typeFilter by remember { mutableStateOf<String?>(null) }
    var showAdd by remember { mutableStateOf(false) }
    var editing by remember { mutableStateOf<ModelSpec?>(null) }
    var removing by remember { mutableStateOf<ModelSpec?>(null) }
    var message by remember { mutableStateOf<String?>(null) }

    LazyColumn(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item {
            Row(Modifier.fillMaxWidth().padding(top = 16.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("AI Core Hub", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.weight(1f))
                Button(onClick = { showAdd = true }) { Text("Add model") }
            }
        }

        item {
            SectionCard("Overview") {
                InfoRow("Installed models", "${installed.size}")
                RowDivider()
                InfoRow("Storage used", formatBytes(installed.sumOf { g.store.sizeOnDisk(it) }))
                RowDivider()
                InfoRow("In memory", if (loaded.isEmpty()) "none" else "${loaded.size} · ${formatBytes(loaded.sumOf { it.sizeBytes })}")
                RowDivider()
                InfoRow("Working", active?.let { "${it.kind} for ${it.clientLabel}" + if (queued > 0) " · $queued queued" else "" } ?: "idle")
                RowDivider()
                InfoRow("Allowed apps", "${clients.count { it.state == AccessState.APPROVED }}")
            }
        }

        if (loaded.isNotEmpty()) {
            item {
                SectionCard("In memory") {
                    loaded.forEachIndexed { i, m ->
                        if (i > 0) RowDivider()
                        Row(Modifier.padding(start = 16.dp, end = 8.dp, top = 6.dp, bottom = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Text(m.name, style = MaterialTheme.typography.bodyLarge)
                                val busy = active?.takeIf { it.modelId == m.id }
                                Hint(if (busy != null) "Working for ${busy.clientLabel}" else "Idle · ${formatBytes(m.sizeBytes)}")
                            }
                            TextButton(onClick = { scope.launch { g.runtime.unload(m.id) } }) { Text("Unload") }
                        }
                    }
                }
            }
        }

        if (downloads.isNotEmpty()) {
            item {
                SectionCard("Downloads") {
                    downloads.values.forEachIndexed { i, d ->
                        if (i > 0) RowDivider()
                        DownloadRow(g, d)
                    }
                }
            }
        }

        item {
            SectionCard("Installed") {
                if (installed.isEmpty()) {
                    Hint("No models yet. Download one from the catalog below or add your own.", Modifier.padding(16.dp))
                }
                installed.forEachIndexed { i, m ->
                    if (i > 0) RowDivider()
                    val isDefault = settings.defaults[m.type] == m.id
                    Column(Modifier.padding(start = 16.dp, end = 8.dp, top = 10.dp, bottom = 4.dp)) {
                        Text(m.name, style = MaterialTheme.typography.titleSmall)
                        TagRow {
                            Tag(typeLabel(m.type))
                            if (isDefault) Tag("Default", highlighted = true)
                            Hint(formatBytes(g.store.sizeOnDisk(m)))
                        }
                        Hint("id: ${m.id}", Modifier.padding(top = 2.dp))
                        Row {
                            if (!isDefault) TextButton(onClick = { g.settings.update { it.withDefault(m.type, m.id) } }) { Text("Make default") }
                            TextButton(onClick = { editing = m }) { Text("Settings") }
                            TextButton(onClick = { removing = m }) { Text("Remove") }
                        }
                    }
                }
            }
        }

        item {
            SectionCard("Catalog", trailing = {
                TextButton(onClick = { scope.launch { message = g.catalog.refresh() ?: "Catalog updated" } }) { Text("Refresh") }
            }) {
                Row(
                    Modifier.horizontalScroll(rememberScrollState()).padding(horizontal = 12.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    TYPE_FILTERS.forEach { (type, label) ->
                        FilterChip(selected = typeFilter == type, onClick = { typeFilter = type }, label = { Text(label) })
                    }
                }
                val visible = catalog.filter { typeFilter == null || it.type == typeFilter }
                    .sortedWith(compareBy<CatalogEntry> { ModelType.ALL.indexOf(it.type) }.thenByDescending { it.recommended }.thenBy { it.sizeBytes })
                visible.forEach { e ->
                    HorizontalDivider()
                    CatalogRow(
                        entry = e,
                        installed = installed.any { it.id == e.id },
                        downloading = downloads.containsKey(e.id),
                        onDownload = { g.downloads.enqueue(ModelSpec.fromCatalog(e)) },
                    )
                }
            }
        }
        item { Spacer(Modifier.padding(bottom = 24.dp)) }
    }

    if (showAdd) AddModelDialog(g, onDismiss = { showAdd = false }, onMessage = { message = it })
    editing?.let { m -> ModelSettingsDialog(g, m, onDismiss = { editing = null }) }
    removing?.let { m ->
        AlertDialog(
            onDismissRequest = { removing = null },
            title = { Text("Remove ${m.name}?") },
            text = { Text("The files (${formatBytes(g.store.sizeOnDisk(m))}) are deleted. Apps that use this model will need it downloaded again.") },
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

@Composable
private fun DownloadRow(g: AppGraph, d: DownloadState) {
    Column(Modifier.padding(horizontal = 16.dp, vertical = 10.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(d.spec.name, style = MaterialTheme.typography.bodyLarge)
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

@Composable
private fun CatalogRow(entry: CatalogEntry, installed: Boolean, downloading: Boolean, onDownload: () -> Unit) {
    Column(Modifier.padding(horizontal = 16.dp, vertical = 12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(entry.name, style = MaterialTheme.typography.titleSmall)
        TagRow {
            Tag(typeLabel(entry.type))
            if (entry.recommended) Tag("Recommended", highlighted = true)
        }
        Text(entry.description, style = MaterialTheme.typography.bodyMedium)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Hint(
                listOf("~" + formatBytes(entry.sizeBytes), entry.languages.joinToString(", "), entry.license)
                    .filter { it.isNotBlank() }.joinToString(" · "),
                Modifier.weight(1f),
            )
            when {
                installed -> Hint("Installed", Modifier.padding(8.dp))
                downloading -> Hint("Downloading…", Modifier.padding(8.dp))
                else -> OutlinedButton(onClick = onDownload) { Text("Download") }
            }
        }
    }
}

/** Equal-height segmented buttons: no check mark, single-line labels. */
@Composable
private fun Segments(labels: List<String>, selected: Int, onSelect: (Int) -> Unit) {
    SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
        labels.forEachIndexed { i, label ->
            SegmentedButton(
                selected = selected == i,
                onClick = { onSelect(i) },
                shape = SegmentedButtonDefaults.itemShape(i, labels.size),
                icon = {},
            ) {
                Text(label, maxLines = 1, overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.labelLarge)
            }
        }
    }
}

private val LLAMA_TYPES = listOf(ModelType.CHAT to "Text", ModelType.EMBEDDING to "Embeddings")

@Composable
private fun TypeSelector(type: String, onChange: (String) -> Unit) {
    Segments(LLAMA_TYPES.map { it.second }, LLAMA_TYPES.indexOfFirst { it.first == type }.coerceAtLeast(0)) {
        onChange(LLAMA_TYPES[it].first)
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

    fun specFor(fileName: String, fileUrl: String, size: Long, displayName: String): ModelSpec =
        if (fileName.endsWith(".onnx", ignoreCase = true)) {
            ModelSpec(
                id = ModelSpec.customId(fileName),
                name = displayName,
                type = ModelType.ONNX,
                url = null,
                fileName = fileName,
                engine = AICoreContract.Engine.ONNX,
                files = listOf(ModelFile(role = "model", path = fileUrl, sizeBytes = size, localName = fileName)),
            )
        } else {
            ModelSpec(
                id = ModelSpec.customId(fileName),
                name = displayName,
                type = type,
                url = fileUrl,
                fileName = fileName.replace(Regex("[^A-Za-z0-9._-]"), "_"),
                sizeBytes = size,
            )
        }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Add model") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Segments(listOf("Hugging Face", "Link", "File"), mode) { mode = it }
                Hint("GGUF files run with llama.cpp, ONNX files can be used by apps through runModel.")
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
                                        runCatching { HuggingFace.listModelFiles(repo) }
                                            .onSuccess { files = it; if (it.isEmpty()) error = "No GGUF or ONNX files in this repository" }
                                            .onFailure { error = it.message }
                                        loading = false
                                    }
                                }
                            }) { Text("List files") }
                            if (loading) CircularProgressIndicator(Modifier.padding(start = 12.dp).size(24.dp))
                        }
                        if (files.isNotEmpty()) {
                            SectionCard {
                                LazyColumn(Modifier.heightIn(max = 260.dp)) {
                                    items(files, key = { it.path }) { f ->
                                        if (f != files.first()) RowDivider()
                                        Row(Modifier.padding(start = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                                            Column(Modifier.weight(1f)) {
                                                Text(f.path.substringAfterLast('/'), style = MaterialTheme.typography.bodyMedium)
                                                Hint(formatBytes(f.sizeBytes))
                                            }
                                            TextButton(onClick = {
                                                val fileName = f.path.substringAfterLast('/')
                                                enqueue(specFor(fileName, f.downloadUrl, f.sizeBytes, fileName.substringBeforeLast('.')))
                                            }) { Text("Get") }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    1 -> {
                        OutlinedTextField(
                            value = url, onValueChange = { url = it },
                            label = { Text("Direct https link to a .gguf or .onnx file") },
                            modifier = Modifier.fillMaxWidth(),
                        )
                        OutlinedTextField(
                            value = name, onValueChange = { name = it },
                            label = { Text("Name (optional)") }, singleLine = true,
                            modifier = Modifier.fillMaxWidth(),
                        )
                    }
                    else -> Hint("Pick a .gguf or .onnx file from your device. It is copied into AI Core Hub's private storage.")
                }
                error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
            }
        },
        confirmButton = {
            when (mode) {
                1 -> TextButton(onClick = {
                    val clean = url.trim()
                    val fileName = clean.substringBefore('?').substringAfterLast('/')
                    val known = fileName.endsWith(".gguf", ignoreCase = true) || fileName.endsWith(".onnx", ignoreCase = true)
                    if (!clean.startsWith("https://") || !known) {
                        error = "Needs an https link ending in .gguf or .onnx"
                    } else {
                        enqueue(specFor(fileName, clean, 0, name.ifBlank { fileName.substringBeforeLast('.') }))
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
    val isLlama = m.engine == AICoreContract.Engine.LLAMA

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Model settings") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(name, { name = it }, label = { Text("Name") }, singleLine = true, modifier = Modifier.fillMaxWidth())
                if (!isLlama) {
                    SectionCard {
                        InfoRow("Type", typeLabel(m.type))
                        RowDivider()
                        InfoRow("Engine", "ONNX Runtime")
                        m.files.forEach { f ->
                            RowDivider()
                            InfoRow(f.role, formatBytes(f.sizeBytes))
                        }
                    }
                } else {
                    TypeSelector(type) { type = it }
                    if (type == ModelType.CHAT) {
                        Hint("Chat template: \"from file\" uses the template stored in the model.")
                        FlowRow(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                            BUILT_IN_TEMPLATES.forEach { t ->
                                FilterChip(selected = template == t, onClick = { template = t }, label = { Text(t.ifEmpty { "from file" }) })
                            }
                        }
                        OutlinedTextField(prefix, { prefix = it }, label = { Text("Assistant prefix (advanced)") }, modifier = Modifier.fillMaxWidth())
                    } else {
                        OutlinedTextField(queryPrefix, { queryPrefix = it }, label = { Text("Query prefix") }, modifier = Modifier.fillMaxWidth())
                        OutlinedTextField(docPrefix, { docPrefix = it }, label = { Text("Document prefix") }, modifier = Modifier.fillMaxWidth())
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = {
                g.store.update(m.id) {
                    it.copy(
                        name = name.ifBlank { m.name },
                        type = if (isLlama) type else m.type,
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
