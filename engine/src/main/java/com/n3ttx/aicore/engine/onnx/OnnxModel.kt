package com.n3ttx.aicore.engine.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.n3ttx.aicore.engine.onnx.tokenizer.BpeTokenizer
import java.io.Closeable
import java.io.File

data class OnnxOptions(val threads: Int = 4, val xnnpack: Boolean = false)

/**
 * A model package run with ONNX Runtime: one or more ONNX files ("components", e.g. a vision and
 * a text tower) plus optional extra files such as a tokenizer. Sessions are created lazily per
 * component, so an app that only searches never loads the vision tower.
 */
class OnnxModel(
    /** Component role (e.g. "vision", "text", "tokenizer") to file on disk. */
    val files: Map<String, File>,
    val options: OnnxOptions,
) : Closeable {
    val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessions = HashMap<String, OrtSession>()
    private var tokenizer: BpeTokenizer? = null
    private val lock = Any()

    private fun sessionOptions() = OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        setIntraOpNumThreads(options.threads.coerceIn(1, 8))
        setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
        setMemoryPatternOptimization(true)
        if (options.xnnpack) {
            runCatching { addXnnpack(mapOf("intra_op_num_threads" to options.threads.coerceIn(1, 8).toString())) }
        }
    }

    fun hasComponent(role: String) = files.containsKey(role)

    fun session(role: String): OrtSession = synchronized(lock) {
        sessions[role] ?: run {
            val file = files[role] ?: throw IllegalArgumentException("Model has no component '$role'")
            env.createSession(file.absolutePath, sessionOptions()).also { sessions[role] = it }
        }
    }

    fun tokenizer(role: String = "tokenizer"): BpeTokenizer = synchronized(lock) {
        tokenizer ?: run {
            val file = files[role] ?: throw IllegalArgumentException("Model has no tokenizer")
            BpeTokenizer.load(file, File(file.parentFile, file.name + ".cache")).also { tokenizer = it }
        }
    }

    /** Frees one component's session, e.g. the vision tower after indexing. */
    fun release(role: String) = synchronized(lock) {
        sessions.remove(role)?.close()
    }

    /** Runs a component with the given inputs. The caller closes the result. */
    fun run(role: String, inputs: Map<String, OnnxTensor>, outputs: Set<String>? = null): OrtSession.Result {
        val s = session(role)
        return if (outputs.isNullOrEmpty()) s.run(inputs) else s.run(inputs, outputs)
    }

    override fun close() = synchronized(lock) {
        sessions.values.forEach { it.close() }
        sessions.clear()
        tokenizer = null
    }
}
