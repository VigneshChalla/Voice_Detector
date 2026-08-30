package com.voicedetector.app

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.edit

class MainActivity : AppCompatActivity() {

    private lateinit var serverInput: EditText
    private lateinit var apiKeyInput: EditText
    private lateinit var connectButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        serverInput = findViewById(R.id.serverInput)
        apiKeyInput = findViewById(R.id.apiKeyInput)
        connectButton = findViewById(R.id.connectButton)

        val prefs = getSharedPreferences("voice_detector", MODE_PRIVATE)
        serverInput.setText(prefs.getString("server_url", ""))
        apiKeyInput.setText(prefs.getString("api_key", "vd_dev_key_2024"))

        connectButton.setOnClickListener {
            val server = serverInput.text.toString().trim()
            val apiKey = apiKeyInput.text.toString().trim()

            if (server.isEmpty()) {
                serverInput.error = "Enter server IP"
                return@setOnClickListener
            }

            val url = if (server.startsWith("http")) server else "http://$server:8000"

            prefs.edit {
                putString("server_url", url)
                putString("api_key", apiKey)
            }

            val intent = Intent(this, DetectionActivity::class.java).apply {
                putExtra("server_url", url)
                putExtra("api_key", apiKey)
            }
            startActivity(intent)
        }
    }
}
