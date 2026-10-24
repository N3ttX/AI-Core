package com.n3ttx.aicore.engine.onnx.tokenizer

/**
 * Open-addressing hash map from Long to Long without boxing. Used for BPE merge tables
 * that can have more than half a million entries.
 */
class LongLongMap(expected: Int) {
    private var capacity = Integer.highestOneBit((expected * 2).coerceAtLeast(16)) shl 1
    private var keys = LongArray(capacity)
    private var values = LongArray(capacity)
    private var used = BooleanArray(capacity)
    var size = 0
        private set

    private fun mix(key: Long): Int {
        var h = key * -0x61c8864680b583ebL
        h = h xor (h ushr 32)
        return h.toInt()
    }

    fun put(key: Long, value: Long) {
        if ((size + 1) * 2 > capacity) grow()
        var i = mix(key) and (capacity - 1)
        while (used[i]) {
            if (keys[i] == key) {
                values[i] = value
                return
            }
            i = (i + 1) and (capacity - 1)
        }
        used[i] = true
        keys[i] = key
        values[i] = value
        size++
    }

    fun get(key: Long, default: Long = Long.MIN_VALUE): Long {
        var i = mix(key) and (capacity - 1)
        while (used[i]) {
            if (keys[i] == key) return values[i]
            i = (i + 1) and (capacity - 1)
        }
        return default
    }

    private fun grow() {
        val oldKeys = keys
        val oldValues = values
        val oldUsed = used
        capacity = capacity shl 1
        keys = LongArray(capacity)
        values = LongArray(capacity)
        used = BooleanArray(capacity)
        size = 0
        for (i in oldKeys.indices) if (oldUsed[i]) put(oldKeys[i], oldValues[i])
    }

    fun forEach(block: (Long, Long) -> Unit) {
        for (i in keys.indices) if (used[i]) block(keys[i], values[i])
    }
}
