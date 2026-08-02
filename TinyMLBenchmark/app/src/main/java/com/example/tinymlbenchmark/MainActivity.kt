package com.example.tinymlbenchmark

import android.os.Bundle
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.util.Locale
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val statusText = findViewById<TextView>(R.id.statusText)
        val timerText = findViewById<TextView>(R.id.timerText)
        val progressBar = findViewById<ProgressBar>(R.id.progressBar)
        val resultsText = findViewById<TextView>(R.id.resultsText)
        val benchmarkEngine = BenchmarkEngine(this)

        // Run off the main UI thread to keep app responsive
        thread {
            // Give OS 2 seconds to reach background idle state
            Thread.sleep(2000)

            val fullResults = StringBuilder()

            fun runAndShow(model: String, flex: Boolean, shape: IntArray) {
                val durationMin = 10
                val totalMs = durationMin * 60 * 1000L
                val startTime = System.currentTimeMillis()

                runOnUiThread { 
                    statusText.text = "Running $model..."
                    progressBar.progress = 0
                    progressBar.visibility = View.VISIBLE
                }

                // Start a timer thread for the UI
                val timerThread = thread {
                    try {
                        while (System.currentTimeMillis() - startTime < totalMs) {
                            val elapsedMs = System.currentTimeMillis() - startTime
                            val remainingMs = totalMs - elapsedMs
                            val progress = ((elapsedMs.toDouble() / totalMs) * 100).toInt()
                            
                            val seconds = (remainingMs / 1000) % 60
                            val minutes = (remainingMs / (1000 * 60)) % 60
                            
                            runOnUiThread {
                                timerText.text = String.format(Locale.US, "Time remaining: %02d:%02d", minutes, seconds)
                                progressBar.progress = progress
                            }
                            Thread.sleep(1000)
                        }
                    } catch (e: InterruptedException) {
                        // Stopped
                    }
                }

                val result = benchmarkEngine.runBenchmark(
                    model,
                    flex,
                    shape,
                    durationMinutes = durationMin,
                )
                
                timerThread.interrupt()
                fullResults.append(result).append("\n")
                
                runOnUiThread { 
                    resultsText.text = fullResults.toString()
                    timerText.text = "Completed: $model"
                }
            }

            // Model 1: DS-CNN + GRU (Uses Flex Delegate)
            runAndShow(
                model = "m1_ds_cnn_gru.tflite",
                flex = true,
                shape = intArrayOf(1, 50, 11)
            )

            // Model 2: DeepConvLSTM (Uses Flex Delegate)
            runAndShow(
                model = "m2_deepconvlstm.tflite",
                flex = true,
                shape = intArrayOf(1, 50, 11)
            )

            // Model 3: Pool-Free 1D-CNN (Standard Built-In INT8 Micro-kernels)
            runAndShow(
                model = "m3_poolfree_1dcnn.tflite",
                flex = false,
                shape = intArrayOf(1, 50, 11)
            )

            runOnUiThread { 
                statusText.text = "Benchmarks Complete!"
                progressBar.visibility = android.view.View.GONE
                timerText.text = "All benchmarks finished"
            }
        }
    }
}