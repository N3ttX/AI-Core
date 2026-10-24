package com.n3ttx.aicore.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
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
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item {
            Column(Modifier.padding(top = 16.dp)) {
                Text("Apps", style = MaterialTheme.typography.headlineSmall)
                Hint("Apps that use AI Core Hub through its service. New apps have to be allowed once.")
            }
        }

        if (requests.isNotEmpty()) {
            item {
                SectionCard("Model requests") {
                    requests.forEachIndexed { i, r ->
                        if (i > 0) RowDivider()
                        Column(Modifier.padding(horizontal = 16.dp, vertical = 10.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            Text("${r.appLabel} asks for ${r.spec.name}", style = MaterialTheme.typography.titleSmall)
                            TagRow {
                                Tag(typeLabel(r.spec.type))
                                if (r.spec.totalBytes > 0) Hint("~${formatBytes(r.spec.totalBytes)}")
                            }
                            r.reason?.let { Text("“$it”", style = MaterialTheme.typography.bodyMedium) }
                            Hint(
                                if (r.fromCatalog) "From the catalog"
                                else "Custom model from " + (r.spec.repo ?: r.spec.url ?: "unknown source"),
                            )
                            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End)) {
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
        }

        item {
            SectionCard("Connected apps") {
                if (clients.isEmpty()) Hint("No app has connected yet.", Modifier.padding(16.dp))
                clients.sortedBy { it.state != AccessState.PENDING }.forEachIndexed { i, c ->
                    if (i > 0) RowDivider()
                    Column(Modifier.padding(start = 16.dp, end = 8.dp, top = 10.dp, bottom = 4.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(c.label, style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                            when (c.state) {
                                AccessState.APPROVED -> Tag("Allowed", highlighted = true)
                                AccessState.DENIED -> Tag("Denied")
                                else -> Tag("Waiting", highlighted = true)
                            }
                        }
                        Hint(c.packageName)
                        if (c.lastUsedAt > 0) {
                            Hint(
                                "${c.requestCount} requests · last " +
                                    DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT).format(Date(c.lastUsedAt)),
                            )
                        }
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp, Alignment.End)) {
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
        item { Spacer(Modifier.padding(bottom = 24.dp)) }
    }
}
