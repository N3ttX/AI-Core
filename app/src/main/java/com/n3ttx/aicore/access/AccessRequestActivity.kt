package com.n3ttx.aicore.access

import android.app.Activity
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.n3ttx.aicore.graph
import com.n3ttx.aicore.sdk.AICoreContract.AccessState
import com.n3ttx.aicore.ui.AICoreTheme

/**
 * Asks the user whether an app may use AI Core Hub. Started by a client with
 * startActivityForResult(ACTION_REQUEST_ACCESS); the result is RESULT_OK if allowed.
 */
class AccessRequestActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val g = graph
        // The system tells us who is asking; an intent extra could be forged.
        val pkg = callingActivity?.packageName
        if (pkg == null || pkg == packageName) {
            setResult(Activity.RESULT_CANCELED)
            finish()
            return
        }
        if (g.access.stateOf(pkg) == AccessState.APPROVED) {
            setResult(Activity.RESULT_OK)
            finish()
            return
        }
        val label = g.access.labelOf(pkg)

        fun decide(approved: Boolean) {
            g.access.setState(pkg, if (approved) AccessState.APPROVED else AccessState.DENIED)
            g.notifier.cancelAccessRequest(pkg)
            setResult(if (approved) Activity.RESULT_OK else Activity.RESULT_CANCELED)
            finish()
        }

        setContent {
            AICoreTheme {
                Surface(shape = MaterialTheme.shapes.extraLarge, tonalElevation = 6.dp) {
                    Column(Modifier.padding(24.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                        Text("Allow $label to use AI Core Hub?", style = MaterialTheme.typography.titleLarge)
                        Text(
                            "$label ($pkg) will be able to run text generation and embeddings with your " +
                                "installed models and ask you to install new ones. Everything stays on this device.",
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp, androidx.compose.ui.Alignment.End)) {
                            OutlinedButton(onClick = { decide(false) }) { Text("Deny") }
                            Button(onClick = { decide(true) }) { Text("Allow") }
                        }
                    }
                }
            }
        }
    }
}
