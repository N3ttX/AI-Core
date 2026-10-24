package com.n3ttx.aicore.engine.onnx.tokenizer

import java.io.Reader

/**
 * Minimal pull-style JSON reader. `tokenizer.json` files of multilingual models are 30+ MB,
 * so building a full object tree would waste hundreds of MB of heap. This reader streams
 * through the document and lets the caller pick only what it needs.
 */
class JsonStreamReader(reader: Reader) {
    enum class Token { BEGIN_OBJECT, END_OBJECT, BEGIN_ARRAY, END_ARRAY, NAME, STRING, NUMBER, BOOLEAN, NULL, END }

    private val input = if (reader is java.io.BufferedReader) reader else reader.buffered(1 shl 16)
    private var peeked = -2
    private val sb = StringBuilder()

    /** Stack of containers: true = object, false = array. Used to tell names from values. */
    private val stack = ArrayList<Boolean>()
    private var expectName = false
    private var cached: Token? = null

    private fun read(): Int {
        if (peeked != -2) {
            val c = peeked
            peeked = -2
            return c
        }
        return input.read()
    }

    private fun peekChar(): Int {
        if (peeked == -2) peeked = input.read()
        return peeked
    }

    private fun skipWs(): Int {
        while (true) {
            val c = peekChar()
            if (c == ' '.code || c == '\n'.code || c == '\r'.code || c == '\t'.code || c == ','.code || c == ':'.code) {
                read()
            } else {
                return c
            }
        }
    }

    fun peek(): Token {
        cached?.let { return it }
        val c = skipWs()
        val t = when {
            c == -1 -> Token.END
            c == '{'.code -> Token.BEGIN_OBJECT
            c == '}'.code -> Token.END_OBJECT
            c == '['.code -> Token.BEGIN_ARRAY
            c == ']'.code -> Token.END_ARRAY
            c == '"'.code -> if (expectName) Token.NAME else Token.STRING
            c == 't'.code || c == 'f'.code -> Token.BOOLEAN
            c == 'n'.code -> Token.NULL
            else -> Token.NUMBER
        }
        cached = t
        return t
    }

    private fun consume(expected: Token) {
        val t = peek()
        check(t == expected) { "Expected $expected but was $t" }
        cached = null
    }

    fun beginObject() {
        consume(Token.BEGIN_OBJECT); read()
        stack.add(true); expectName = true
    }

    fun endObject() {
        consume(Token.END_OBJECT); read()
        stack.removeAt(stack.lastIndex); afterValue()
    }

    fun beginArray() {
        consume(Token.BEGIN_ARRAY); read()
        stack.add(false); expectName = false
    }

    fun endArray() {
        consume(Token.END_ARRAY); read()
        stack.removeAt(stack.lastIndex); afterValue()
    }

    fun hasNext(): Boolean {
        val t = peek()
        return t != Token.END_OBJECT && t != Token.END_ARRAY && t != Token.END
    }

    private fun afterValue() {
        expectName = stack.isNotEmpty() && stack.last()
    }

    fun nextName(): String {
        consume(Token.NAME)
        val s = readQuoted()
        expectName = false
        return s
    }

    fun nextString(): String {
        consume(Token.STRING)
        val s = readQuoted()
        afterValue()
        return s
    }

    fun nextLong(): Long = readLiteral().toDouble().toLong()

    fun nextInt(): Int = nextLong().toInt()

    fun nextDouble(): Double = readLiteral().toDouble()

    fun nextBoolean(): Boolean = readLiteral() == "true"

    fun nextNull() {
        readLiteral()
    }

    /** Returns a string for STRING tokens, the literal text for numbers/booleans and null for null. */
    fun nextStringOrNull(): String? = when (peek()) {
        Token.STRING -> nextString()
        Token.NULL -> { nextNull(); null }
        else -> readLiteral()
    }

    private fun readLiteral(): String {
        val t = peek()
        check(t == Token.NUMBER || t == Token.BOOLEAN || t == Token.NULL) { "Expected literal but was $t" }
        cached = null
        sb.setLength(0)
        while (true) {
            val c = peekChar()
            if (c == -1 || c == ','.code || c == '}'.code || c == ']'.code || c == ' '.code ||
                c == '\n'.code || c == '\r'.code || c == '\t'.code
            ) break
            sb.append(read().toChar())
        }
        afterValue()
        return sb.toString()
    }

    private fun readQuoted(): String {
        check(read() == '"'.code)
        sb.setLength(0)
        while (true) {
            val c = read()
            when {
                c == -1 -> error("Unterminated string")
                c == '"'.code -> return sb.toString()
                c == '\\'.code -> {
                    when (val e = read()) {
                        'n'.code -> sb.append('\n')
                        't'.code -> sb.append('\t')
                        'r'.code -> sb.append('\r')
                        'b'.code -> sb.append('\b')
                        'f'.code -> sb.append('\u000C')
                        'u'.code -> {
                            var v = 0
                            repeat(4) { v = (v shl 4) or Character.digit(read(), 16) }
                            sb.append(v.toChar())
                        }
                        else -> sb.append(e.toChar())
                    }
                }
                else -> sb.append(c.toChar())
            }
        }
    }

    /** Skips the next value, including nested objects and arrays. */
    fun skipValue() {
        when (peek()) {
            Token.BEGIN_OBJECT -> {
                beginObject()
                while (hasNext()) { nextName(); skipValue() }
                endObject()
            }
            Token.BEGIN_ARRAY -> {
                beginArray()
                while (hasNext()) skipValue()
                endArray()
            }
            Token.STRING -> nextString()
            Token.NAME -> { nextName(); skipValue() }
            else -> readLiteral()
        }
    }
}
