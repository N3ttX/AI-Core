package com.n3ttx.aicore.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.n3ttx.aicore.graph
import com.n3ttx.aicore.sdk.AICoreContract.AccessState
import java.text.DateFormat
import java.util.Date

@Composable
fun AppsScreen() {
    val g = LocalContext.current.graph
    val clients by g.access.clients.collectAsStateWithLifecycle()
    val requests by g.modelRequests.requests.collectAsStateWithLifecycle()
    val downloads by g.downloads.downloads.collectAsStateWithLifecycle()

    LazyColumn(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        item {
            Text("Apps", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.padding(top = 16.dp))
            Hint("Apps that use AI Core Hub through its service. New apps have to be allowed once.")
        }

        if (requests.isNotEmpty()) {
            item { SectionTitle("Model requests") }
            items(requests, key = { it.packageName + "/" + it.spec.id }) { r ->
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text("${r.appLabel} asks for ${r.spec.name}", style = MaterialTheme.typography.titleSmall)
                        r.reason?.let { Text("“$it”", style = MaterialTheme.typography.bodyMedium) }
                        Hint(
                            (if (r.fromCatalog) "From the catalog" else "Custom model: ${r.spec.url}") +
                                (if (r.spec.sizeBytes > 0) " · ~${formatBytes(r.spec.sizeBytes)}" else ""),
                        )
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp, androidx.compose.ui.Alignment.End)) {
                            OutlinedButton(onClick = { g.modelRequests.removeModel(r.spec.id) }) { Text("Reject") }
                            Button(
                                enabled = !downloads.containsKey(r.spec.id),
                                onClick = { g.downloads.enqueue(r.spec) },
                            ) { Text(if (downloads.containsKey(r.spec.id)) "Downloading" else "Download") }
                        }
                    }
                }
            }
        }

        item { SectionTitle("Connected apps") }
        if (clients.isEmpty()) {
            item { Hint("No app has connected yet.") }
        }
        items(clients.sortedBy { it.state != AccessState.PENDING }, key = { it.packageName }) { c ->
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(c.label, style = MaterialTheme.typography.titleSmall)
                    Hint(c.packageName)
                    val stateText = when (c.state) {
                        AccessState.APPROVED -> "Allowed"
                        AccessState.DENIED -> "Denied"
                        else -> "Waiting for your decision"
                    }
                    val used = if (c.lastUsedAt > 0) " · ${c.requestCount} requests, last ${DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT).format(Date(c.lastUsedAt))}" else ""
                    Text(stateText + used, style = MaterialTheme.typography.bodySmall)
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp, androidx.compose.ui.Alignment.End)) {
                        TextButton(onClick = { g.access.forget(c.packageName) }) { Text("Forget") }
                        if (c.state != AccessState.DENIED) {
                            TextButton(onClick = {
                                g.access.setState(c.packageName, AccessState.DENIED)
                                g.notifier.cancelAccessRequest(c.packageName)
                            }) { Text("Deny") }
                        }
                        if (c.state != AccessState.APPROVED) {
                            Button(onClick = {
                                g.access.setState(c.packageName, AccessState.APPROVED)
                                g.notifier.cancelAccessRequest(c.packageName)
                            }) { Text("Allow") }
                        }
                    }
                }
            }
        }
    }
}
