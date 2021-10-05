/*
 * *
 *  * Created by Rafsan Ahmad on 10/5/21, 1:08 PM
 *  * Copyright (c) 2021 . All rights reserved.
 *
 */

package org.tensorflow.lite.examples.digitclassifier

import android.annotation.SuppressLint
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.divyanshu.draw.widget.DrawView
import com.google.firebase.analytics.ktx.analytics
import com.google.firebase.ktx.Firebase
import com.google.firebase.ml.modeldownloader.CustomModel
import com.google.firebase.ml.modeldownloader.CustomModelDownloadConditions
import com.google.firebase.ml.modeldownloader.DownloadType
import com.google.firebase.ml.modeldownloader.FirebaseModelDownloader
import com.google.firebase.perf.FirebasePerformance
import com.google.firebase.perf.metrics.Trace
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.channels.FileChannel


class MainActivity : AppCompatActivity() {

    private var drawView: DrawView? = null
    private var clearButton: Button? = null
    private var yesButton: Button? = null
    private var predictedTextView: TextView? = null
    private var digitClassifier = DigitClassifier(this)
    private val firebasePerformance = FirebasePerformance.getInstance()
    private lateinit var downloadTrace: Trace

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.tfe_dc_activity_main)

        // Setup view instances
        drawView = findViewById(R.id.draw_view)
        drawView?.setStrokeWidth(70.0f)
        drawView?.setColor(Color.WHITE)
        drawView?.setBackgroundColor(Color.BLACK)
        clearButton = findViewById(R.id.clear_button)
        yesButton = findViewById(R.id.yes_button)
        predictedTextView = findViewById(R.id.predicted_text)

        // Setup clear drawing button
        clearButton?.setOnClickListener {
            drawView?.clearCanvas()
            predictedTextView?.text = getString(R.string.tfe_dc_prediction_text_placeholder)
        }

        // Setup YES button
        yesButton?.setOnClickListener {
            Firebase.analytics.logEvent("correct_inference", null)
        }

        // Setup classification trigger so that it classify after every stroke drew
        drawView?.setOnTouchListener { _, event ->
            // As we have interrupted DrawView's touch event,
            // we first need to pass touch events through to the instance for the drawing to show up
            drawView?.onTouchEvent(event)

            // Then if user finished a touch event, run classification
            if (event.action == MotionEvent.ACTION_UP) {
                classifyDrawing()
            }

            true
        }
        setupDigitClassifier()
    }

//    private fun setupDigitClassifier() {
//        digitClassifier.initialize(loadModelFile())
//    }

    override fun onDestroy() {
        digitClassifier.close()
        super.onDestroy()
    }

    private fun classifyDrawing() {
        val bitmap = drawView?.getBitmap()

        if ((bitmap != null) && (digitClassifier.isInitialized)) {
            // Add these lines to create and start the trace
            val classifyTrace = firebasePerformance.newTrace("classify")
            classifyTrace.start()
            digitClassifier
                .classifyAsync(bitmap)
                .addOnSuccessListener { resultText ->
                    // Add this line to stop the trace on success
                    classifyTrace.stop()
                    predictedTextView?.text = resultText
                }
                .addOnFailureListener { e ->
                    predictedTextView?.text = getString(
                        R.string.tfe_dc_classification_error_message,
                        e.localizedMessage
                    )
                    Log.e(TAG, "Error classifying drawing.", e)
                }
        }
    }

    @Throws(IOException::class)
    private fun loadModelFile(): ByteBuffer {
        val fileDescriptor = assets.openFd(MainActivity.MODEL_FILE)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun setupDigitClassifier() {
        // Add these lines to create and start the trace
        downloadTrace = firebasePerformance.newTrace("download_model")
        downloadTrace.start()
        downloadModel("mnist_v1")
    }

    private fun downloadModel(modelName: String) {
        val conditions = CustomModelDownloadConditions.Builder()
            .requireWifi()  // Also possible: .requireCharging() and .requireDeviceIdle()
            .build()
        FirebaseModelDownloader.getInstance()
            .getModel(
                modelName, DownloadType.LOCAL_MODEL_UPDATE_IN_BACKGROUND,
                conditions
            )
            .addOnSuccessListener { model: CustomModel? ->
                // Download complete. Depending on your app, you could enable the ML
                // feature, or switch from the local model to the remote model, etc.

                // The CustomModel object contains the local path of the model file,
                // which you can use to instantiate a TensorFlow Lite interpreter.
                val modelFile = model?.file
                if (modelFile != null) {
                    showToast("Downloaded remote model: $model")
                    //var interpreter = Interpreter(modelFile)
                    digitClassifier.initializeInterpreter(modelFile)
                    downloadTrace.stop()
                } else {
                    showToast("Failed to get model file.")
                }
            }
            .addOnFailureListener {
                showToast("Exception occurred in downloading model.")
            }
    }

    private fun showToast(text: String) {
        Toast.makeText(
            this,
            text,
            Toast.LENGTH_LONG
        ).show()
    }

    companion object {
        private const val TAG = "MainActivity"

        private const val MODEL_FILE = "mnist.tflite"
    }
}
