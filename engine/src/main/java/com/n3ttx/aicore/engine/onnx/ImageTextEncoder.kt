package com.n3ttx.aicore.engine.onnx

import ai.onnxruntime.OnnxTensor
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import com.n3ttx.aicore.engine.onnx.tokenizer.BpeTokenizer
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.LongBuffer
import kotlin.math.sqrt

/** Preprocessing of a CLIP/SigLIP style model, read from the model's config. */
data class ImageTextConfig(
    val imageSize: Int,
    /** "squash" (SigLIP) or "center_crop" (CLIP). */
    val resize: String,
    val mean: FloatArray,
    val std: FloatArray,
    val maxTextLength: Int,
    val padId: Int,
    val lowercase: Boolean,
) {
    companion object {
        fun from(json: JSONObject?): ImageTextConfig {
            val j = json ?: JSONObject()
            fun floats(key: String, default: Float) =
                j.optJSONArray(key)?.let { a -> FloatArray(3) { a.optDouble(it, default.toDouble()).toFloat() } }
                    ?: floatArrayOf(default, default, default)
            return ImageTextConfig(
                imageSize = j.optInt("image_size", 224),
                resize = j.optString("resize", "center_crop"),
                mean = floats("mean", 0f),
                std = floats("std", 1f),
                maxTextLength = j.optInt("max_text_length", 77),
                padId = j.optInt("pad_id", 0),
                lowercase = j.optBoolean("lowercase", true),
            )
        }
    }
}

/**
 * Image and text embeddings with a two-tower model (CLIP, MobileCLIP, SigLIP). Mirrors the
 * preprocessing of the Lumen gallery app exactly, so vectors from both are interchangeable.
 */
class ImageTextEncoder(private val model: OnnxModel, val config: ImageTextConfig) {
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)

    /** Scales/crops a bitmap to the model's input resolution. */
    private fun prepare(source: Bitmap): Bitmap {
        val size = config.imageSize
        val out = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        val src = if (source.config == Bitmap.Config.HARDWARE) source.copy(Bitmap.Config.ARGB_8888, false) else source
        val w = src.width.toFloat()
        val h = src.height.toFloat()
        if (config.resize == "squash") {
            canvas.drawBitmap(src, null, RectF(0f, 0f, size.toFloat(), size.toFloat()), paint)
        } else {
            val side = minOf(w, h)
            val left = ((w - side) / 2f).toInt()
            val top = ((h - side) / 2f).toInt()
            canvas.drawBitmap(
                src,
                Rect(left, top, left + side.toInt(), top + side.toInt()),
                RectF(0f, 0f, size.toFloat(), size.toFloat()),
                paint,
            )
        }
        if (src !== source) src.recycle()
        return out
    }

    /** Returns an L2-normalized image embedding. */
    fun embedImage(bitmap: Bitmap): FloatArray {
        val size = config.imageSize
        val prepared = if (bitmap.width == size && bitmap.height == size && bitmap.config == Bitmap.Config.ARGB_8888) bitmap
        else prepare(bitmap)
        val plane = size * size
        val ints = IntArray(plane)
        prepared.getPixels(ints, 0, size, 0, 0, size, size)
        if (prepared !== bitmap) prepared.recycle()

        val buffer = ByteBuffer.allocateDirect(plane * 3 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        val mean = config.mean
        val std = config.std
        val sR = 1f / (255f * std[0]); val sG = 1f / (255f * std[1]); val sB = 1f / (255f * std[2])
        val oR = mean[0] / std[0]; val oG = mean[1] / std[1]; val oB = mean[2] / std[2]
        for (i in 0 until plane) {
            val c = ints[i]
            buffer.put(i, ((c shr 16) and 0xFF) * sR - oR)
            buffer.put(plane + i, ((c shr 8) and 0xFF) * sG - oG)
            buffer.put(2 * plane + i, (c and 0xFF) * sB - oB)
        }
        buffer.rewind()
        val session = model.session(VISION)
        val inputName = session.inputNames.first()
        OnnxTensor.createTensor(model.env, buffer, longArrayOf(1, 3, size.toLong(), size.toLong())).use { tensor ->
            val outputName = pickOutput(session.outputNames, "image_embeds")
            model.run(VISION, mapOf(inputName to tensor), setOf(outputName)).use { result ->
                return normalize(readFloats(result.get(outputName).get() as OnnxTensor))
            }
        }
    }

    fun tokenize(text: String): LongArray =
        model.tokenizer().encode(
            text.trim(),
            BpeTokenizer.Config(maxLength = config.maxTextLength, padId = config.padId, lowercase = config.lowercase),
        )

    /** Returns an L2-normalized text embedding in the same space as [embedImage]. */
    fun embedText(text: String): FloatArray {
        val ids = tokenize(text)
        val session = model.session(TEXT)
        val inputName = session.inputNames.firstOrNull { it.contains("input_ids") } ?: session.inputNames.first()
        OnnxTensor.createTensor(model.env, LongBuffer.wrap(ids), longArrayOf(1, ids.size.toLong())).use { tensor ->
            val inputs = HashMap<String, OnnxTensor>()
            inputs[inputName] = tensor
            var mask: OnnxTensor? = null
            if (session.inputNames.contains("attention_mask")) {
                mask = OnnxTensor.createTensor(model.env, LongBuffer.wrap(LongArray(ids.size) { 1 }), longArrayOf(1, ids.size.toLong()))
                inputs["attention_mask"] = mask
            }
            try {
                val outputName = pickOutput(session.outputNames, "text_embeds")
                return model.run(TEXT, inputs, setOf(outputName)).use { result ->
                    normalize(readFloats(result.get(outputName).get() as OnnxTensor))
                }
            } finally {
                mask?.close()
            }
        }
    }

    private fun pickOutput(names: Set<String>, preferred: String): String =
        names.firstOrNull { it == preferred } ?: names.firstOrNull { it == "pooler_output" }
            ?: names.firstOrNull { it.endsWith("embeds") } ?: names.last()

    companion object {
        const val VISION = "vision"
        const val TEXT = "text"

        fun readFloats(tensor: OnnxTensor): FloatArray {
            val fb = tensor.floatBuffer
            val arr = FloatArray(fb.remaining())
            fb.get(arr)
            return arr
        }

        fun normalize(v: FloatArray): FloatArray {
            var sum = 0.0
            for (x in v) sum += x * x
            val inv = if (sum > 0) (1.0 / sqrt(sum)).toFloat() else 0f
            for (i in v.indices) v[i] *= inv
            return v
        }
    }
}
