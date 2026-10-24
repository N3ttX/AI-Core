package com.n3ttx.aicore.sdk

import android.os.Bundle
import android.os.SharedMemory
import android.system.OsConstants
import com.n3ttx.aicore.sdk.AICoreContract.DType
import com.n3ttx.aicore.sdk.AICoreContract.Key
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * A raw tensor for [AICoreClient.runModel]. [data] holds the values in little-endian byte order.
 * Large tensors cross the process boundary as shared memory, small ones inline.
 */
class Tensor(val name: String, val dtype: String, val shape: LongArray, val data: ByteArray) {
    val elementCount: Long get() = shape.fold(1L) { a, b -> a * b }

    private fun buffer(): ByteBuffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)

    fun floats(): FloatArray {
        require(dtype == DType.FLOAT32) { "Tensor $name is $dtype, not float32" }
        return FloatArray(data.size / 4).also { buffer().asFloatBuffer().get(it) }
    }

    fun longs(): LongArray {
        require(dtype == DType.INT64) { "Tensor $name is $dtype, not int64" }
        return LongArray(data.size / 8).also { buffer().asLongBuffer().get(it) }
    }

    fun ints(): IntArray {
        require(dtype == DType.INT32) { "Tensor $name is $dtype, not int32" }
        return IntArray(data.size / 4).also { buffer().asIntBuffer().get(it) }
    }

    fun toBundle(): Bundle = Bundle().apply {
        putString(Key.NAME, name)
        putString(Key.DTYPE, dtype)
        putLongArray(Key.SHAPE, shape)
        if (data.size <= INLINE_LIMIT) {
            putByteArray(Key.BYTES, data)
        } else {
            val shm = SharedMemory.create("tensor-$name", data.size)
            val map = shm.mapReadWrite()
            map.put(data)
            SharedMemory.unmap(map)
            shm.setProtect(OsConstants.PROT_READ)
            putParcelable(Key.SHARED_MEMORY, shm)
        }
    }

    override fun toString() = "Tensor($name, $dtype, ${shape.contentToString()})"

    companion object {
        /** Tensors above this size are sent through shared memory. */
        const val INLINE_LIMIT = 64 * 1024

        fun ofFloats(name: String, shape: LongArray, values: FloatArray) = Tensor(
            name, DType.FLOAT32, shape,
            ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN).also { it.asFloatBuffer().put(values) }.array(),
        )

        fun ofLongs(name: String, shape: LongArray, values: LongArray) = Tensor(
            name, DType.INT64, shape,
            ByteBuffer.allocate(values.size * 8).order(ByteOrder.LITTLE_ENDIAN).also { it.asLongBuffer().put(values) }.array(),
        )

        fun ofInts(name: String, shape: LongArray, values: IntArray) = Tensor(
            name, DType.INT32, shape,
            ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN).also { it.asIntBuffer().put(values) }.array(),
        )

        fun ofBytes(name: String, dtype: String, shape: LongArray, values: ByteArray) = Tensor(name, dtype, shape, values)

        @Suppress("DEPRECATION")
        fun fromBundle(b: Bundle): Tensor {
            val bytes = b.getByteArray(Key.BYTES) ?: b.getParcelable<SharedMemory>(Key.SHARED_MEMORY)?.use { shm ->
                val map = shm.mapReadOnly()
                try {
                    ByteArray(map.remaining()).also { map.get(it) }
                } finally {
                    SharedMemory.unmap(map)
                }
            } ?: ByteArray(0)
            return Tensor(
                name = b.getString(Key.NAME).orEmpty(),
                dtype = b.getString(Key.DTYPE) ?: DType.FLOAT32,
                shape = b.getLongArray(Key.SHAPE) ?: longArrayOf(),
                data = bytes,
            )
        }
    }
}
