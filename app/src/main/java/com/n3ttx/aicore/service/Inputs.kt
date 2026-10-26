package com.n3ttx.aicore.service

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.os.Bundle
import android.os.ParcelFileDescriptor
import com.n3ttx.aicore.sdk.AICoreContract.DType
import com.n3ttx.aicore.sdk.AICoreContract.Key
import com.n3ttx.aicore.sdk.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * An image received from a client. File descriptors and bitmaps are taken out of the request
 * while the binder call is still running, because the request's parcel is recycled afterwards.
 */
sealed class ImageSource : AutoCloseable {
    abstract fun decode(maxDimension: Int): Bitmap

    class OfBitmap(private val bitmap: Bitmap) : ImageSource() {
        override fun decode(maxDimension: Int): Bitmap {
            val b = if (bitmap.config == Bitmap.Config.HARDWARE) bitmap.copy(Bitmap.Config.ARGB_8888, false) else bitmap
            val longest = max(b.width, b.height)
            if (maxDimension <= 0 || longest <= maxDimension) return b
            val scale = maxDimension.toFloat() / longest
            return Bitmap.createScaledBitmap(b, (b.width * scale).roundToInt().coerceAtLeast(1), (b.height * scale).roundToInt().coerceAtLeast(1), true)
        }

        override fun close() = Unit
    }

    class OfFile(private val fd: ParcelFileDescriptor) : ImageSource() {
        override fun decode(maxDimension: Int): Bitmap {
            val bytes = ParcelFileDescriptor.AutoCloseInputStream(fd).use { it.readBytes() }
            // ImageDecoder handles HEIF/AVIF and applies the EXIF orientation.
            return ImageDecoder.decodeBitmap(ImageDecoder.createSource(ByteBuffer.wrap(bytes))) { decoder, info, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                val w = info.size.width
                val h = info.size.height
                val longest = max(w, h)
                if (maxDimension in 1 until longest) {
                    val scale = maxDimension.toFloat() / longest
                    decoder.setTargetSize((w * scale).roundToInt().coerceAtLeast(1), (h * scale).roundToInt().coerceAtLeast(1))
                }
            }
        }

        override fun close() {
            runCatching { fd.close() }
        }
    }

    companion object {
        @Suppress("DEPRECATION")
        fun listFrom(request: Bundle): List<ImageSource> =
            request.getParcelableArrayList<Bundle>(Key.IMAGES).orEmpty().map { b ->
                b.getParcelable<Bitmap>(Key.BITMAP)?.let { OfBitmap(it) }
                    ?: b.getParcelable<ParcelFileDescriptor>(Key.FD)?.let { OfFile(it) }
                    ?: throw IllegalArgumentException("Image needs a bitmap or a file descriptor")
            }
    }
}

/** RGB bytes of the bitmap for a vision language model (alpha is dropped). */
fun Bitmap.toRgbImage(): com.n3ttx.aicore.engine.RgbImage {
    val soft = if (config == Bitmap.Config.HARDWARE) copy(Bitmap.Config.ARGB_8888, false) else this
    val pixels = IntArray(soft.width * soft.height)
    soft.getPixels(pixels, 0, soft.width, 0, 0, soft.width, soft.height)
    val rgb = ByteArray(pixels.size * 3)
    for (i in pixels.indices) {
        val p = pixels[i]
        rgb[i * 3] = (p shr 16).toByte()
        rgb[i * 3 + 1] = (p shr 8).toByte()
        rgb[i * 3 + 2] = p.toByte()
    }
    return com.n3ttx.aicore.engine.RgbImage(soft.width, soft.height, rgb)
}

/** Conversion between SDK tensors (little-endian bytes) and ONNX Runtime tensors. */
object Tensors {
    private fun javaType(dtype: String): OnnxJavaType = when (dtype) {
        DType.FLOAT32 -> OnnxJavaType.FLOAT
        DType.FLOAT16 -> OnnxJavaType.FLOAT16
        DType.INT64 -> OnnxJavaType.INT64
        DType.INT32 -> OnnxJavaType.INT32
        DType.INT8 -> OnnxJavaType.INT8
        DType.UINT8 -> OnnxJavaType.UINT8
        DType.BOOL -> OnnxJavaType.BOOL
        else -> throw IllegalArgumentException("Unsupported tensor type $dtype")
    }

    private fun dtypeOf(type: OnnxJavaType): String = when (type) {
        OnnxJavaType.FLOAT -> DType.FLOAT32
        OnnxJavaType.FLOAT16 -> DType.FLOAT16
        OnnxJavaType.INT64 -> DType.INT64
        OnnxJavaType.INT32 -> DType.INT32
        OnnxJavaType.INT8 -> DType.INT8
        OnnxJavaType.UINT8 -> DType.UINT8
        OnnxJavaType.BOOL -> DType.BOOL
        else -> throw IllegalArgumentException("Unsupported output type $type")
    }

    fun toOnnx(env: OrtEnvironment, t: Tensor): OnnxTensor {
        val buf = ByteBuffer.allocateDirect(t.data.size).order(ByteOrder.LITTLE_ENDIAN)
        buf.put(t.data)
        buf.rewind()
        return OnnxTensor.createTensor(env, buf, t.shape, javaType(t.dtype))
    }

    fun fromOnnx(name: String, t: OnnxTensor): Tensor {
        val bb = t.byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
        val bytes = ByteArray(bb.remaining()).also { bb.get(it) }
        return Tensor(name, dtypeOf(t.info.type), t.info.shape, bytes)
    }
}
