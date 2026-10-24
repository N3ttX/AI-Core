package com.n3ttx.aicore.runtime

/**
 * Post-processes streamed model output: hides a leading `<think>…</think>`
 * block and cuts the text at the first stop sequence. Text that might be the
 * start of a stop sequence is held back until it is clear.
 */
class OutputFilter(
    stops: List<String>,
    private val hideThinking: Boolean,
    private val emit: (String) -> Boolean,
) {
    private val stops = stops.filter { it.isNotEmpty() }
    private val pending = StringBuilder()
    private val output = StringBuilder()
    private var phase = if (hideThinking) Phase.START else Phase.TEXT

    var stoppedBySequence = false
        private set

    val text: String get() = output.toString()

    private enum class Phase { START, THINKING, TEXT }

    /** Returns false when generation should stop. */
    fun push(chunk: String): Boolean {
        pending.append(chunk)
        if (phase == Phase.START) {
            val trimmed = pending.trimStart()
            when {
                trimmed.startsWith(THINK_OPEN) -> {
                    phase = Phase.THINKING
                    pending.setLength(0)
                    pending.append(trimmed.substring(THINK_OPEN.length))
                }
                THINK_OPEN.startsWith(trimmed) -> return true
                else -> phase = Phase.TEXT
            }
        }
        if (phase == Phase.THINKING) {
            val end = pending.indexOf(THINK_CLOSE)
            if (end < 0) {
                if (pending.length > THINK_CLOSE.length) pending.delete(0, pending.length - THINK_CLOSE.length)
                return true
            }
            val rest = pending.substring(end + THINK_CLOSE.length).trimStart()
            pending.setLength(0)
            pending.append(rest)
            phase = Phase.TEXT
        }
        return flushText(final = false)
    }

    /** Emits whatever is still held back. */
    fun finish() {
        if (phase == Phase.TEXT && !stoppedBySequence) flushText(final = true)
        if (phase == Phase.START && pending.isNotEmpty()) {
            phase = Phase.TEXT
            flushText(final = true)
        }
    }

    private fun flushText(final: Boolean): Boolean {
        var stopAt = -1
        for (s in stops) {
            val i = pending.indexOf(s)
            if (i >= 0 && (stopAt < 0 || i < stopAt)) stopAt = i
        }
        if (stopAt >= 0) {
            stoppedBySequence = true
            send(pending.substring(0, stopAt))
            pending.setLength(0)
            return false
        }
        val hold = if (final) 0 else holdBack()
        val ready = pending.length - hold
        if (ready <= 0) return true
        val out = pending.substring(0, ready)
        pending.delete(0, ready)
        return send(out)
    }

    /** Length of the longest suffix of pending that is a proper prefix of a stop sequence. */
    private fun holdBack(): Int {
        var best = 0
        for (s in stops) {
            val max = minOf(s.length - 1, pending.length)
            for (len in max downTo 1) {
                if (len <= best) break
                if (pending.matchesAt(pending.length - len, s, len)) {
                    best = len
                    break
                }
            }
        }
        return best
    }

    private fun send(s: String): Boolean {
        if (s.isEmpty()) return true
        output.append(s)
        return emit(s)
    }

    companion object {
        private const val THINK_OPEN = "<think>"
        private const val THINK_CLOSE = "</think>"
    }
}

private fun StringBuilder.matchesAt(offset: Int, other: String, len: Int): Boolean {
    for (i in 0 until len) if (this[offset + i] != other[i]) return false
    return true
}
