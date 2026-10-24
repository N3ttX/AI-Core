package com.n3ttx.aicore.engine.onnx

import ai.onnxruntime.OnnxTensor
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

data class DetectedFace(
    /** Box in pixels of the analysed bitmap. */
    val box: RectF,
    /** 5 landmarks (eyes, nose, mouth corners) as x0,y0,x1,y1,… in pixels. */
    val landmarks: FloatArray,
    val score: Float,
)

/**
 * SCRFD face detection + ArcFace embeddings (InsightFace models in Immich's ONNX packaging).
 * Same pipeline as the Lumen gallery app: letterbox 640, anchors at strides 8/16/32, NMS 0.4 and
 * 5-point similarity alignment to the 112×112 ArcFace template.
 */
class FaceAnalyzer(private val model: OnnxModel) {
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)

    fun detect(source: Bitmap, threshold: Float = 0.55f, minSizePx: Float = 20f): List<DetectedFace> {
        val bitmap = if (source.config == Bitmap.Config.HARDWARE) source.copy(Bitmap.Config.ARGB_8888, false) else source
        val scale = SIZE.toFloat() / max(bitmap.width, bitmap.height)
        val canvasBitmap = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
        Canvas(canvasBitmap).apply {
            drawColor(Color.BLACK)
            drawBitmap(bitmap, null, RectF(0f, 0f, bitmap.width * scale, bitmap.height * scale), paint)
        }
        if (bitmap !== source) bitmap.recycle()
        val pixels = IntArray(SIZE * SIZE)
        canvasBitmap.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        canvasBitmap.recycle()
        val plane = SIZE * SIZE
        val buffer = ByteBuffer.allocateDirect(3 * plane * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        for (i in 0 until plane) {
            val c = pixels[i]
            buffer.put(i, (((c shr 16) and 0xFF) - 127.5f) / 128f)
            buffer.put(plane + i, (((c shr 8) and 0xFF) - 127.5f) / 128f)
            buffer.put(2 * plane + i, ((c and 0xFF) - 127.5f) / 128f)
        }
        buffer.rewind()

        val detector = model.session(DETECTION)
        val candidates = ArrayList<DetectedFace>()
        OnnxTensor.createTensor(model.env, buffer, longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())).use { input ->
            detector.run(mapOf(detector.inputNames.first() to input)).use { result ->
                val outputs = (0 until result.size()).map { (result.get(it) as OnnxTensor).floatBuffer }
                STRIDES.forEachIndexed { level, stride ->
                    val scores = outputs[level]
                    val boxes = outputs[level + 3]
                    val kps = outputs[level + 6]
                    val grid = SIZE / stride
                    val count = grid * grid * 2
                    for (j in 0 until count) {
                        val score = scores.get(j)
                        if (score < threshold) continue
                        val cell = j / 2
                        val cx = (cell % grid) * stride.toFloat()
                        val cy = (cell / grid) * stride.toFloat()
                        val box = RectF(
                            (cx - boxes.get(j * 4) * stride) / scale,
                            (cy - boxes.get(j * 4 + 1) * stride) / scale,
                            (cx + boxes.get(j * 4 + 2) * stride) / scale,
                            (cy + boxes.get(j * 4 + 3) * stride) / scale,
                        )
                        if (box.width() < minSizePx || box.height() < minSizePx) continue
                        val lm = FloatArray(10) { k ->
                            val v = kps.get(j * 10 + k) * stride
                            (if (k % 2 == 0) cx + v else cy + v) / scale
                        }
                        candidates += DetectedFace(box, lm, score)
                    }
                }
            }
        }
        return nms(candidates.sortedByDescending { it.score }, 0.4f)
    }

    private fun nms(sorted: List<DetectedFace>, iouThreshold: Float): List<DetectedFace> {
        val keep = ArrayList<DetectedFace>()
        for (d in sorted) {
            if (keep.none { iou(it.box, d.box) > iouThreshold }) keep += d
        }
        return keep
    }

    private fun iou(a: RectF, b: RectF): Float {
        val ix = max(0f, min(a.right, b.right) - max(a.left, b.left))
        val iy = max(0f, min(a.bottom, b.bottom) - max(a.top, b.top))
        val inter = ix * iy
        return inter / (a.width() * a.height() + b.width() * b.height() - inter)
    }

    /** Aligns the face to the ArcFace 112×112 template and returns an L2-normalized embedding. */
    fun embed(source: Bitmap, landmarks: FloatArray): FloatArray {
        val bitmap = if (source.config == Bitmap.Config.HARDWARE) source.copy(Bitmap.Config.ARGB_8888, false) else source
        val aligned = align(bitmap, landmarks)
        if (bitmap !== source) bitmap.recycle()
        val pixels = IntArray(112 * 112)
        aligned.getPixels(pixels, 0, 112, 0, 0, 112, 112)
        aligned.recycle()
        val plane = 112 * 112
        val buffer = ByteBuffer.allocateDirect(3 * plane * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        for (i in 0 until plane) {
            val c = pixels[i]
            buffer.put(i, (((c shr 16) and 0xFF) - 127.5f) / 127.5f)
            buffer.put(plane + i, (((c shr 8) and 0xFF) - 127.5f) / 127.5f)
            buffer.put(2 * plane + i, ((c and 0xFF) - 127.5f) / 127.5f)
        }
        buffer.rewind()
        val recognizer = model.session(RECOGNITION)
        OnnxTensor.createTensor(model.env, buffer, longArrayOf(1, 3, 112, 112)).use { input ->
            recognizer.run(mapOf(recognizer.inputNames.first() to input)).use { result ->
                return ImageTextEncoder.normalize(ImageTextEncoder.readFloats(result.get(0) as OnnxTensor))
            }
        }
    }

    /** Least-squares similarity transform (rotation, uniform scale, translation) onto the template. */
    private fun align(bitmap: Bitmap, lm: FloatArray): Bitmap {
        var sx = 0f; var sy = 0f; var dx = 0f; var dy = 0f
        for (i in 0 until 5) {
            sx += lm[i * 2]; sy += lm[i * 2 + 1]
            dx += TEMPLATE[i * 2]; dy += TEMPLATE[i * 2 + 1]
        }
        sx /= 5; sy /= 5; dx /= 5; dy /= 5
        var num1 = 0f; var num2 = 0f; var den = 0f
        for (i in 0 until 5) {
            val px = lm[i * 2] - sx; val py = lm[i * 2 + 1] - sy
            val qx = TEMPLATE[i * 2] - dx; val qy = TEMPLATE[i * 2 + 1] - dy
            num1 += px * qx + py * qy
            num2 += px * qy - py * qx
            den += px * px + py * py
        }
        val a = num1 / den
        val b = num2 / den
        val tx = dx - (a * sx - b * sy)
        val ty = dy - (b * sx + a * sy)
        val matrix = Matrix().apply { setValues(floatArrayOf(a, -b, tx, b, a, ty, 0f, 0f, 1f)) }
        val out = Bitmap.createBitmap(112, 112, Bitmap.Config.ARGB_8888)
        Canvas(out).apply {
            drawColor(Color.BLACK)
            drawBitmap(bitmap, matrix, paint)
        }
        return out
    }

    companion object {
        const val DETECTION = "detection"
        const val RECOGNITION = "recognition"
        const val SIZE = 640
        private val STRIDES = intArrayOf(8, 16, 32)
        private val TEMPLATE = floatArrayOf(
            38.2946f, 51.6963f, 73.5318f, 51.5014f, 56.0252f, 71.7366f, 41.5493f, 92.3655f, 70.7299f, 92.2041f,
        )
    }
}
