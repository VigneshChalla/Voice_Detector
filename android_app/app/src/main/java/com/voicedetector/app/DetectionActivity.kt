package com.voicedetector.app

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.*
import java.util.concurrent.TimeUnit

class DetectionActivity : AppCompatActivity() {

    companion object {
        private const val PERMISSION_REQUEST_CODE = 100
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    }

    private lateinit var recordButton: Button
    private lateinit var analyzeButton: Button
    private lateinit var resultText: TextView
    private lateinit var riskText: TextView
    private lateinit var scoreText: TextView
    private lateinit var statusText: TextView

    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordingThread: Thread? = null
    private var tempFile: File? = null

    private var serverUrl: String = ""
    private var apiKey: String = ""
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    private val handler = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_detection)

        serverUrl = intent.getStringExtra("server_url") ?: ""
        apiKey = intent.getStringExtra("api_key") ?: ""

        recordButton = findViewById(R.id.recordButton)
        analyzeButton = findViewById(R.id.analyzeButton)
        resultText = findViewById(R.id.resultText)
        riskText = findViewById(R.id.riskText)
        scoreText = findViewById(R.id.scoreText)
        statusText = findViewById(R.id.statusText)

        analyzeButton.isEnabled = false
        statusText.text = "Server: $serverUrl"

        checkPermission()

        recordButton.setOnClickListener {
            if (isRecording) {
                stopRecording()
            } else {
                startRecording()
            }
        }

        analyzeButton.setOnClickListener {
            analyzeAudio()
        }

        testConnection()
    }

    private fun checkPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                PERMISSION_REQUEST_CODE
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Microphone permission granted", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Microphone permission required", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun testConnection() {
        val request = Request.Builder()
            .url("$serverUrl/api/v1/health")
            .addHeader("X-API-Key", apiKey)
            .get()
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                handler.post {
                    statusText.text = "Connection failed: ${e.message}"
                    statusText.setTextColor(0xFFFF4444.toInt())
                }
            }

            override fun onResponse(call: Call, response: Response) {
                handler.post {
                    if (response.isSuccessful) {
                        statusText.text = "Connected ✓"
                        statusText.setTextColor(0xFF00FF88.toInt())
                    } else {
                        statusText.text = "Server error: ${response.code}"
                        statusText.setTextColor(0xFFFF4444.toInt())
                    }
                }
            }
        })
    }

    private fun startRecording() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            checkPermission()
            return
        }

        val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            )
        } catch (e: SecurityException) {
            Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show()
            return
        }

        tempFile = File(cacheDir, "recording.pcm")

        audioRecord?.startRecording()
        isRecording = true

        recordButton.text = "⏹ Stop Recording"
        recordButton.setBackgroundColor(0xFFCC0000.toInt())
        analyzeButton.isEnabled = false
        resultText.text = ""
        riskText.text = ""
        scoreText.text = ""

        recordingThread = Thread {
            try {
                val outputStream = FileOutputStream(tempFile!!)
                val buffer = ByteArray(bufferSize)

                while (isRecording) {
                    val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                    if (read > 0) {
                        outputStream.write(buffer, 0, read)
                    }
                }

                outputStream.flush()
                outputStream.close()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }.apply { start() }
    }

    private fun stopRecording() {
        isRecording = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null

        recordButton.text = "🎤 Start Recording"
        recordButton.setBackgroundColor(0xFF00D4FF.toInt())
        analyzeButton.isEnabled = true
    }

    private fun analyzeAudio() {
        val file = tempFile ?: return
        if (!file.exists() || file.length()().toInt() == 0) {
            Toast.makeText(this, "No audio recorded", Toast.LENGTH_SHORT).show()
            return
        }

        analyzeButton.isEnabled = false
        analyzeButton.text = "Analyzing..."
        resultText.text = "Processing..."

        // Convert PCM to WAV
        val wavFile = File(cacheDir, "recording.wav")
        pcmToWav(file, wavFile, SAMPLE_RATE, 1, 16)

        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "file", "recording.wav",
                wavFile.readBody().toRequestBody("audio/wav".toMediaType())
            )
            .addFormDataPart("caller_id", "android_user")
            .addFormDataPart("call_type", "regular_call")
            .build()

        val request = Request.Builder()
            .url("$serverUrl/api/v1/detect")
            .addHeader("X-API-Key", apiKey)
            .post(requestBody)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                handler.post {
                    resultText.text = "Error: ${e.message}"
                    resultText.setTextColor(0xFFFF4444.toInt())
                    analyzeButton.isEnabled = true
                    analyzeButton.text = "🔍 Analyze Voice"
                }
            }

            override fun onResponse(call: Call, response: Response) {
                val body = response.body?.string() ?: ""
                handler.post {
                    try {
                        val json = com.google.gson.JsonParser.parseString(body).asJsonObject
                        val synthProb = json.get("synthetic_probability").asDouble
                        val riskLevel = json.get("risk_level").asString
                        val isSynthetic = json.get("is_synthetic").asBoolean
                        val recommendation = json.get("recommendation").asString

                        val percentage = (synthProb * 100).toInt()

                        if (isSynthetic) {
                            resultText.text = "🚨 SYNTHETIC DETECTED"
                            resultText.setTextColor(0xFFFF4444.toInt())
                        } else {
                            resultText.text = "✅ GENUINE VOICE"
                            resultText.setTextColor(0xFF00FF88.toInt())
                        }

                        scoreText.text = "Score: $percentage%"
                        riskText.text = "Risk: $riskLevel"
                        riskText.setTextColor(when (riskLevel) {
                            "HIGH" -> 0xFFFF4444.toInt()
                            "MEDIUM" -> 0xFFFFAA00.toInt()
                            else -> 0xFF00FF88.toInt()
                        })

                        Toast.makeText(
                            this@DetectionActivity,
                            recommendation,
                            Toast.LENGTH_LONG
                        ).show()
                    } catch (e: Exception) {
                        resultText.text = "Parse error"
                        resultText.setTextColor(0xFFFF4444.toInt())
                    }

                    analyzeButton.isEnabled = true
                    analyzeButton.text = "🔍 Analyze Voice"
                }
            }
        })
    }

    private fun File.readBody(): RequestBody {
        return this.readBytes().toRequestBody("audio/wav".toMediaType())
    }

    private fun pcmToWav(pcmFile: File, wavFile: File, sampleRate: Int, channels: Int, bitsPerSample: Int) {
        val pcmData = pcmFile.readBytes()
        val totalDataLen = pcmData.size + 36
        val byteRate = sampleRate * channels * bitsPerSample / 8

        val output = FileOutputStream(wavFile)
        val header = ByteArray(44)

        // RIFF header
        header[0] = 'R'.code.toByte()
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()
        writeInt(header, 4, totalDataLen)
        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()

        // fmt chunk
        header[12] = 'f'.code.toByte()
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()
        writeInt(header, 16, 16)
        writeShort(header, 20, 1)
        writeShort(header, 22, channels)
        writeInt(header, 24, sampleRate)
        writeInt(header, 28, byteRate)
        writeShort(header, 32, channels * bitsPerSample / 8)
        writeShort(header, 34, bitsPerSample)

        // data chunk
        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()
        writeInt(header, 40, pcmData.size)

        output.write(header)
        output.write(pcmData)
        output.close()
    }

    private fun writeInt(header: ByteArray, offset: Int, value: Int) {
        header[offset] = (value and 0xFF).toByte()
        header[offset + 1] = ((value shr 8) and 0xFF).toByte()
        header[offset + 2] = ((value shr 16) and 0xFF).toByte()
        header[offset + 3] = ((value shr 24) and 0xFF).toByte()
    }

    private fun writeShort(header: ByteArray, offset: Int, value: Int) {
        header[offset] = (value and 0xFF).toByte()
        header[offset + 1] = ((value shr 8) and 0xFF).toByte()
    }
}
