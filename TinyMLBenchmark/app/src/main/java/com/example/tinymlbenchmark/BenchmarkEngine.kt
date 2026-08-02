package com.example.tinymlbenchmark

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Debug
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import java.io.FileInputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.ArrayList
import kotlin.random.Random

class BenchmarkEngine(private val context: Context) {

    companion object {
        init {
            try {
                // Force Android JNI Linker to load TensorFlow Lite native symbols
                System.loadLibrary("tensorflowlite_jni")
                System.loadLibrary("tensorflowlite_flex_jni")
                Log.i("TINYML_BENCHMARK", "Native TensorFlow Lite & Flex JNI Libraries successfully loaded.")
            } catch (e: UnsatisfiedLinkError) {
                Log.e("TINYML_BENCHMARK", "Failed to load native TFLite binaries: ${e.message}")
            }
        }
    }

    private val TAG = "TINYML_BENCHMARK"
    private var cpuAccessBlocked = false
    private var lastCpu: Long = 0
    private var lastIdle: Long = 0

    private fun getCpuUsage(): Double {
        if (cpuAccessBlocked) return 0.0

        return try {
            val reader = RandomAccessFile("/proc/stat", "r")
            val load = reader.readLine() ?: return 0.0
            reader.close()

            val toks = load.split(" +".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
            val idle = toks[4].toLong()
            val cpu = toks[1].toLong() + toks[2].toLong() + toks[3].toLong() +
                    toks[6].toLong() + toks[7].toLong() + toks[8].toLong()

            if (lastCpu == 0L && lastIdle == 0L) {
                lastCpu = cpu
                lastIdle = idle
                return 0.0
            }

            val totalDelta = (cpu + idle) - (lastCpu + lastIdle)
            val result = if (totalDelta <= 0L) 0.0 else (cpu - lastCpu).toDouble() / totalDelta * 100.0

            lastCpu = cpu
            lastIdle = idle

            result
        } catch (e: Exception) {
            if (!cpuAccessBlocked) {
                Log.w(TAG, "CPU usage reading blocked (Android 8+ security): ${e.message}")
                cpuAccessBlocked = true
            }
            0.0
        }
    }

    private fun getRamUsage(): Long {
        val memoryInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(memoryInfo)
        return memoryInfo.totalPss.toLong() // In KB
    }

    private fun getBatteryLevel(): Float {
        return try {
            val batteryStatus: Intent? = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            val level = batteryStatus?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
            val scale = batteryStatus?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1

            if (scale <= 0) 0f else (level / scale.toFloat()) * 100f
        } catch (e: Exception) {
            Log.e(TAG, "Error getting battery level: ${e.message}")
            0f
        }
    }

    // Fixed: Keep FileChannel open during mapping so memory buffer remains valid
    private fun loadModelFile(modelPath: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // Fixed: Removed the colliding addDelegate parameter
    fun runBenchmark(
        modelName: String,
        useFlexDelegate: Boolean,
        inputShape: IntArray,
        durationMinutes: Int = 1
    ): String {
        val resultBuilder = StringBuilder()
        resultBuilder.append("MODEL: $modelName\n")

        Log.i(TAG, "==================================================")
        Log.i(TAG, "STARTING BENCHMARK FOR MODEL: $modelName ($durationMinutes min)")

        val initialBattery = getBatteryLevel()
        val initialRam = getRamUsage()

        val options = Interpreter.Options()
        var flexDelegate: FlexDelegate? = null

        if (useFlexDelegate) {
            try {
                flexDelegate = FlexDelegate()
                options.addDelegate(flexDelegate)
                Log.i(TAG, "Flex Delegate successfully attached.")
            } catch (e: Throwable) {
                Log.e(TAG, "Error initializing Flex Delegate: ${e.message}")
                return "Error initializing Flex Delegate: ${e.message}\n"
            }
        }

        var interpreter: Interpreter? = null
        try {
            val modelBuffer = loadModelFile(modelName)
            interpreter = Interpreter(modelBuffer, options)

            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)

            val inputBuffer = ByteBuffer.allocateDirect(inputTensor.numBytes())
                .order(ByteOrder.nativeOrder())
            val outputBuffer = ByteBuffer.allocateDirect(outputTensor.numBytes())
                .order(ByteOrder.nativeOrder())

            for (i in 0 until inputTensor.numBytes()) {
                inputBuffer.put(i, 1.toByte())
            }

            // Warmup iterations
            Log.i(TAG, "Running warmup iterations...")
            for (i in 1..20) {
                inputBuffer.rewind()
                outputBuffer.rewind()
                interpreter.run(inputBuffer, outputBuffer)
            }

            // Timed Benchmark loop
            val startTimeGlobal = System.currentTimeMillis()
            val endTimeGlobal = startTimeGlobal + (durationMinutes * 60 * 1000L)

            val maxLatencySamples = 100_000
            val latencies = ArrayList<Long>(maxLatencySamples)
            val cpuSamples = mutableListOf<Double>()
            var iterations = 0

            Log.i(TAG, "Executing profiling loop...")
            while (System.currentTimeMillis() < endTimeGlobal) {
                inputBuffer.rewind()
                outputBuffer.rewind()

                val start = System.nanoTime()
                interpreter.run(inputBuffer, outputBuffer)
                val end = System.nanoTime()

                val latency = end - start
                if (latencies.size < maxLatencySamples) {
                    latencies.add(latency)
                } else {
                    val randomIndex = Random.nextInt(iterations + 1)
                    if (randomIndex < maxLatencySamples) {
                        latencies[randomIndex] = latency
                    }
                }
                iterations++

                // Periodically sample CPU without sleeping inside the loop
                if (iterations % 2000 == 0) {
                    val cpu = getCpuUsage()
                    if (cpu > 0.0) cpuSamples.add(cpu)
                }
            }

            val finalRam = getRamUsage()
            val finalBattery = getBatteryLevel()

            latencies.sort()
            val trimSize = (latencies.size * 0.05).toInt()
            var trimmedSum = 0L
            for (i in trimSize until (latencies.size - trimSize)) {
                trimmedSum += latencies[i]
            }
            val validSamples = latencies.size - (2 * trimSize)
            val meanMs = if (validSamples > 0) (trimmedSum.toDouble() / validSamples) / 1_000_000.0 else 0.0

            val avgCpu = if (cpuSamples.isNotEmpty()) cpuSamples.average() else 0.0
            val cpuString = if (cpuAccessBlocked && avgCpu == 0.0) "Blocked" else String.format(java.util.Locale.ROOT, "%.1f%%", avgCpu)

            resultBuilder.append(String.format(java.util.Locale.ROOT, "Latency: %.3f ms\n", meanMs))
            resultBuilder.append("Avg CPU: $cpuString\n")
            resultBuilder.append(String.format(java.util.Locale.ROOT, "RAM (Total App): %.2f MB\n", finalRam / 1024.0))
            resultBuilder.append(String.format(java.util.Locale.ROOT, "Model Footprint: %.2f MB\n", (finalRam - initialRam) / 1024.0))
            resultBuilder.append(String.format(java.util.Locale.ROOT, "Battery Drop: %.4f%%\n", initialBattery - finalBattery))
            resultBuilder.append("Iterations: $iterations\n")
            resultBuilder.append("----------------------------\n")

            Log.i(TAG, "BENCHMARK COMPLETE: $modelName")
            return resultBuilder.toString()

        } catch (e: Exception) {
            Log.e(TAG, "CRITICAL ERROR Benchmarking $modelName: ${e.message}", e)
            return "Error benchmarking $modelName: ${e.message}\n"
        } finally {
            interpreter?.close()
            flexDelegate?.close()
        }
    }
}