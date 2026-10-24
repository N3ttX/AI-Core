package com.n3ttx.aicore.sdk

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.net.Uri
import android.os.Bundle
import android.os.ParcelFileDescriptor
import com.n3ttx.aicore.sdk.AICoreContract.Key
import java.io.File

/**
 * An image for [AICoreClient.embedImages] and [AICoreClient.detectFaces]. A file descriptor lets
 * AI Core decode the image itself (no storage permission needed on its side); a bitmap is useful
 * when the app already has a decoded thumbnail.
 */
sealed class ImageInput {
    abstract fun toBundle(): Bundle

    class OfBitmap(val bitmap: Bitmap) : ImageInput() {
        override fun toBundle() = Bundle().apply {
            // Hardware bitmaps cannot be read from another process.
            val b = if (bitmap.config == Bitmap.Config.HARDWARE) bitmap.copy(Bitmap.Config.ARGB_8888, false) else bitmap
            putParcelable(Key.BITMAP, b)
        }
    }

    /** AI Core reads the encoded image (JPEG, PNG, HEIF, AVIF, WebP …) from [fd]. The caller closes it afterwards. */
    class OfFile(val fd: ParcelFileDescriptor) : ImageInput() {
        override fun toBundle() = Bundle().apply { putParcelable(Key.FD, fd) }
    }

    companion object {
        fun of(bitmap: Bitmap): ImageInput = OfBitmap(bitmap)

        fun of(file: File): ImageInput = OfFile(ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY))

        /** Opens a content:// or file:// uri (e.g. from MediaStore). */
        fun of(context: Context, uri: Uri): ImageInput =
            OfFile(requireNotNull(context.contentResolver.openFileDescriptor(uri, "r")) { "Cannot open $uri" })
    }
}

data class Face(
    /** Bounding box in pixels of the analysed image ([FaceResult.width] × [FaceResult.height]). */
    val box: RectF,
    /** Bounding box relative to the image size, 0..1. Use this to map onto your own image. */
    val boxNormalized: RectF,
    /** Eyes, nose tip, mouth corners as x0,y0,…,x4,y4 in pixels of the analysed image. */
    val landmarks: FloatArray,
    val score: Float,
    /** L2-normalized face embedding, empty if not requested. Compare with cosine similarity. */
    val embedding: FloatArray,
)

data class FaceResult(val width: Int, val height: Int, val faces: List<Face>) {
    companion object {
        @Suppress("DEPRECATION")
        fun fromBundle(b: Bundle): FaceResult = FaceResult(
            width = b.getInt(Key.WIDTH),
            height = b.getInt(Key.HEIGHT),
            faces = b.getParcelableArrayList<Bundle>(Key.FACES).orEmpty().map { f ->
                val box = f.getFloatArray(Key.BOX) ?: FloatArray(4)
                val n = f.getFloatArray(Key.BOX_NORMALIZED) ?: FloatArray(4)
                Face(
                    box = RectF(box[0], box[1], box[2], box[3]),
                    boxNormalized = RectF(n[0], n[1], n[2], n[3]),
                    landmarks = f.getFloatArray(Key.LANDMARKS) ?: FloatArray(0),
                    score = f.getFloat(Key.SCORE),
                    embedding = f.getFloatArray(Key.EMBEDDING) ?: FloatArray(0),
                )
            },
        )
    }
}

data class Capabilities(
    val apiVersion: Int,
    val serviceVersion: String,
    val tasks: List<String>,
    val engines: List<String>,
    val modelTypes: List<String>,
    val maxBatch: Int,
) {
    fun supports(task: String) = task in tasks
}

/** A model from AI Core's catalog that can be requested with [AICoreClient.requestModel]. */
data class CatalogModel(
    val id: String,
    val name: String,
    val type: String,
    val engine: String,
    val description: String,
    val sizeBytes: Long,
    val languages: List<String>,
    val installed: Boolean,
    val downloading: Boolean,
    /** Model specific settings as JSON, e.g. embedding size or face match threshold. */
    val config: String?,
)
