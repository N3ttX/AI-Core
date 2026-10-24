// Ported unchanged from the Lumen gallery app (app.lumen.photos.ai.tokenizer) so that text
// embeddings computed here match the ones Lumen computes itself.
package com.n3ttx.aicore.engine.onnx.tokenizer

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.text.Normalizer
import java.util.Locale

/**
 * Pure Kotlin implementation of the two BPE flavours used by the supported image/text models:
 *
 *  * [Style.CLIP] – OpenAI CLIP / MobileCLIP: lower-casing, regex pre-tokenisation, GPT-2 byte-level
 *    alphabet and an end-of-word suffix (`</w>`).
 *  * [Style.SENTENCEPIECE] – Gemma tokenizer used by SigLIP 2: spaces become `▁`, the whole text is one
 *    BPE word and unknown characters fall back to `<0xXX>` byte tokens. Handles 100+ languages.
 *
 * Both are loaded from a Hugging Face `tokenizer.json` and cached in a compact binary format.
 */
class BpeTokenizer private constructor(
    val style: Style,
    private val vocab: HashMap<String, Int>,
    /** key = (leftId shl 32 | rightId), value = (rank shl 32 | mergedId) */
    private val merges: LongLongMap,
    private val unkId: Int,
    private val bosId: Int,
    private val eosId: Int,
    private val suffix: String,
) {
    enum class Style { CLIP, SENTENCEPIECE }

    data class Config(
        val maxLength: Int,
        val padId: Int = 0,
        val lowercase: Boolean = true,
    )

    val vocabSize: Int get() = vocab.size

    private val wordCache = object : LinkedHashMap<String, IntArray>(256, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, IntArray>?) = size > 2048
    }

    fun encode(text: String, config: Config): LongArray {
        val ids = when (style) {
            Style.CLIP -> encodeClip(text)
            Style.SENTENCEPIECE -> encodeSentencePiece(if (config.lowercase) text.lowercase(Locale.ROOT) else text)
        }
        val out = LongArray(config.maxLength) { config.padId.toLong() }
        var n = 0
        when (style) {
            Style.CLIP -> {
                out[n++] = bosId.toLong()
                for (id in ids) {
                    if (n >= config.maxLength - 1) break
                    out[n++] = id.toLong()
                }
                out[n] = eosId.toLong()
            }
            Style.SENTENCEPIECE -> {
                for (id in ids) {
                    if (n >= config.maxLength - 1) break
                    out[n++] = id.toLong()
                }
                out[n] = eosId.toLong()
            }
        }
        return out
    }

    /** Returns the raw token ids without special tokens or padding (used in tests). */
    fun tokenize(text: String): IntArray = when (style) {
        Style.CLIP -> encodeClip(text)
        Style.SENTENCEPIECE -> encodeSentencePiece(text)
    }

    private fun encodeClip(text: String): IntArray {
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFC)
            .replace(WHITESPACE, " ")
            .lowercase(Locale.ROOT)
        val result = ArrayList<Int>()
        val m = CLIP_PATTERN.matcher(normalized)
        while (m.find()) {
            val piece = m.group()
            val cached = synchronized(wordCache) { wordCache[piece] }
            val ids = cached ?: bpeClipWord(piece).also { synchronized(wordCache) { wordCache[piece] = it } }
            ids.forEach { result.add(it) }
        }
        return result.toIntArray()
    }

    private fun bpeClipWord(piece: String): IntArray {
        val bytes = piece.toByteArray(Charsets.UTF_8)
        val symbols = ArrayList<Int>(bytes.size)
        for ((i, b) in bytes.withIndex()) {
            var s = BYTE_ENCODER[b.toInt() and 0xFF].toString()
            if (i == bytes.lastIndex) s += suffix
            symbols.add(vocab[s] ?: unkId)
        }
        return merge(symbols)
    }

    private fun encodeSentencePiece(text: String): IntArray {
        val normalized = text.replace(' ', '▁')
        val symbols = ArrayList<Int>(normalized.length)
        var i = 0
        while (i < normalized.length) {
            val cp = normalized.codePointAt(i)
            val s = String(Character.toChars(cp))
            val id = vocab[s]
            if (id != null) {
                symbols.add(id)
            } else {
                for (b in s.toByteArray(Charsets.UTF_8)) {
                    symbols.add(vocab[String.format(Locale.ROOT, "<0x%02X>", b.toInt() and 0xFF)] ?: unkId)
                }
            }
            i += Character.charCount(cp)
        }
        return merge(symbols)
    }

    /** Classic BPE: repeatedly merge the adjacent pair with the lowest rank (leftmost on ties). */
    private fun merge(symbols: ArrayList<Int>): IntArray {
        while (symbols.size > 1) {
            var bestRank = Long.MAX_VALUE
            var bestIndex = -1
            var bestId = -1
            for (i in 0 until symbols.size - 1) {
                val v = merges.get(pairKey(symbols[i], symbols[i + 1]))
                if (v != Long.MIN_VALUE) {
                    val rank = v ushr 32
                    if (rank < bestRank) {
                        bestRank = rank
                        bestIndex = i
                        bestId = (v and 0xFFFFFFFFL).toInt()
                    }
                }
            }
            if (bestIndex < 0) break
            symbols[bestIndex] = bestId
            symbols.removeAt(bestIndex + 1)
        }
        return symbols.toIntArray()
    }

    fun writeCache(file: File) {
        val tmp = File(file.parentFile, file.name + ".tmp")
        DataOutputStream(BufferedOutputStream(FileOutputStream(tmp), 1 shl 16)).use { out ->
            out.writeInt(CACHE_MAGIC)
            out.writeInt(style.ordinal)
            out.writeInt(unkId)
            out.writeInt(bosId)
            out.writeInt(eosId)
            out.writeUTF(suffix)
            out.writeInt(vocab.size)
            for ((k, v) in vocab) {
                out.writeUTF(k)
                out.writeInt(v)
            }
            out.writeInt(merges.size)
            merges.forEach { k, v ->
                out.writeLong(k)
                out.writeLong(v)
            }
        }
        tmp.renameTo(file)
    }

    companion object {
        private const val CACHE_MAGIC = 0x4C554D31 // "LUM1"
        private val WHITESPACE = Regex("\\s+")
        private val CLIP_PATTERN = java.util.regex.Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"
        )

        /** GPT-2 byte to printable unicode character table. */
        private val BYTE_ENCODER: CharArray = CharArray(256).also { table ->
            val bs = ArrayList<Int>()
            bs.addAll('!'.code..'~'.code)
            bs.addAll('¡'.code..'¬'.code)
            bs.addAll('®'.code..'ÿ'.code)
            val cs = ArrayList(bs)
            var n = 0
            for (b in 0 until 256) {
                if (b !in bs) {
                    bs.add(b)
                    cs.add(256 + n)
                    n++
                }
            }
            for (i in bs.indices) table[bs[i]] = cs[i].toChar()
        }

        private fun pairKey(a: Int, b: Int): Long = (a.toLong() shl 32) or (b.toLong() and 0xFFFFFFFFL)

        /** Loads from the binary cache if present, otherwise parses `tokenizer.json` and writes the cache. */
        fun load(tokenizerJson: File, cacheFile: File): BpeTokenizer {
            if (cacheFile.exists() && cacheFile.lastModified() >= tokenizerJson.lastModified()) {
                runCatching { return readCache(cacheFile) }
            }
            val tokenizer = parse(tokenizerJson)
            runCatching { tokenizer.writeCache(cacheFile) }
            return tokenizer
        }

        fun readCache(file: File): BpeTokenizer {
            DataInputStream(BufferedInputStream(FileInputStream(file), 1 shl 16)).use { input ->
                check(input.readInt() == CACHE_MAGIC) { "Bad cache" }
                val style = Style.entries[input.readInt()]
                val unk = input.readInt()
                val bos = input.readInt()
                val eos = input.readInt()
                val suffix = input.readUTF()
                val vocabSize = input.readInt()
                val vocab = HashMap<String, Int>(vocabSize * 2)
                repeat(vocabSize) { vocab[input.readUTF()] = input.readInt() }
                val mergeCount = input.readInt()
                val merges = LongLongMap(mergeCount)
                repeat(mergeCount) { merges.put(input.readLong(), input.readLong()) }
                return BpeTokenizer(style, vocab, merges, unk, bos, eos, suffix)
            }
        }

        fun parse(file: File): BpeTokenizer =
            InputStreamReader(FileInputStream(file), Charsets.UTF_8).use { parse(JsonStreamReader(it)) }

        fun parse(reader: JsonStreamReader): BpeTokenizer {
            val vocab = HashMap<String, Int>()
            val added = HashMap<String, Int>()
            val rawMerges = ArrayList<Pair<String, String>>()
            var suffix: String? = null
            var byteFallback = false
            var unkToken: String? = null

            reader.beginObject()
            while (reader.hasNext()) {
                when (reader.nextName()) {
                    "added_tokens" -> {
                        reader.beginArray()
                        while (reader.hasNext()) {
                            var id = -1
                            var content: String? = null
                            reader.beginObject()
                            while (reader.hasNext()) {
                                when (reader.nextName()) {
                                    "id" -> id = reader.nextInt()
                                    "content" -> content = reader.nextString()
                                    else -> reader.skipValue()
                                }
                            }
                            reader.endObject()
                            if (content != null && id >= 0) added[content] = id
                        }
                        reader.endArray()
                    }
                    "model" -> {
                        reader.beginObject()
                        while (reader.hasNext()) {
                            when (reader.nextName()) {
                                "end_of_word_suffix" -> suffix = reader.nextStringOrNull()
                                "byte_fallback" -> byteFallback = reader.nextStringOrNull() == "true"
                                "unk_token" -> unkToken = reader.nextStringOrNull()
                                "vocab" -> {
                                    reader.beginObject()
                                    while (reader.hasNext()) {
                                        val token = reader.nextName()
                                        vocab[token] = reader.nextInt()
                                    }
                                    reader.endObject()
                                }
                                "merges" -> {
                                    reader.beginArray()
                                    while (reader.hasNext()) {
                                        if (reader.peek() == JsonStreamReader.Token.BEGIN_ARRAY) {
                                            reader.beginArray()
                                            val a = reader.nextString()
                                            val b = reader.nextString()
                                            while (reader.hasNext()) reader.skipValue()
                                            reader.endArray()
                                            rawMerges.add(a to b)
                                        } else {
                                            val s = reader.nextString()
                                            val sp = s.indexOf(' ', startIndex = 1)
                                            if (sp > 0) rawMerges.add(s.substring(0, sp) to s.substring(sp + 1))
                                        }
                                    }
                                    reader.endArray()
                                }
                                else -> reader.skipValue()
                            }
                        }
                        reader.endObject()
                    }
                    else -> reader.skipValue()
                }
            }
            reader.endObject()

            for ((k, v) in added) vocab.putIfAbsent(k, v)

            val merges = LongLongMap(rawMerges.size)
            rawMerges.forEachIndexed { rank, (a, b) ->
                val ia = vocab[a] ?: return@forEachIndexed
                val ib = vocab[b] ?: return@forEachIndexed
                val merged = vocab[a + b] ?: return@forEachIndexed
                val key = pairKey(ia, ib)
                // Keep the first (lowest) rank if a pair is listed twice.
                if (merges.get(key) == Long.MIN_VALUE) {
                    merges.put(key, (rank.toLong() shl 32) or merged.toLong())
                }
            }

            val clipStyle = !suffix.isNullOrEmpty()
            val style = if (clipStyle) Style.CLIP else Style.SENTENCEPIECE
            val unk = unkToken?.let { vocab[it] } ?: 0
            val bos = vocab["<|startoftext|>"] ?: vocab["<bos>"] ?: vocab["<s>"] ?: 0
            val eos = vocab["<|endoftext|>"] ?: vocab["<eos>"] ?: vocab["</s>"] ?: 1
            check(clipStyle || byteFallback || vocab.isNotEmpty()) { "Unsupported tokenizer" }
            return BpeTokenizer(style, vocab, merges, unk, bos, eos, suffix ?: "")
        }
    }
}
