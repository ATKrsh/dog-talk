package com.example.dogtalk.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Base64
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.dogtalk.model.DogAnalysis
import com.example.dogtalk.network.WebSocketManager
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

class DogTalkViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "DogTalkVM"
        private const val AUDIO_SAMPLE_RATE = 16000
        private const val JPEG_QUALITY = 60
        private const val FRAME_INTERVAL_MS = 300L // ~3 FPS to server
    }

    val webSocketManager = WebSocketManager()

    // Connection
    val connectionState = webSocketManager.connectionState
    val latestAnalysis = webSocketManager.latestAnalysis
    val latencyMs = webSocketManager.latencyMs

    // UI state
    private val _serverIp = MutableStateFlow("10.0.2.2")
    val serverIp: StateFlow<String> = _serverIp.asStateFlow()

    private val _serverPort = MutableStateFlow(8765)
    val serverPort: StateFlow<Int> = _serverPort.asStateFlow()

    private val _isAnalyzing = MutableStateFlow(false)
    val isAnalyzing: StateFlow<Boolean> = _isAnalyzing.asStateFlow()

    private val _audioEnabled = MutableStateFlow(true)
    val audioEnabled: StateFlow<Boolean> = _audioEnabled.asStateFlow()

    private val _analysisHistory = MutableStateFlow<List<DogAnalysis>>(emptyList())
    val analysisHistory: StateFlow<List<DogAnalysis>> = _analysisHistory.asStateFlow()

    // Audio recording
    private var audioRecord: AudioRecord? = null
    private var audioJob: Job? = null
    private var lastAudioBase64: String? = null

    // Frame throttling
    private var lastFrameTime = 0L

    fun updateServerIp(ip: String) {
        _serverIp.value = ip
    }

    fun updateServerPort(port: Int) {
        _serverPort.value = port
    }

    fun connect() {
        webSocketManager.connect(_serverIp.value, _serverPort.value)
    }

    fun disconnect() {
        stopAnalysis()
        webSocketManager.disconnect()
    }

    fun startAnalysis() {
        _isAnalyzing.value = true
        if (_audioEnabled.value) {
            startAudioCapture()
        }
    }

    fun stopAnalysis() {
        _isAnalyzing.value = false
        stopAudioCapture()
    }

    fun toggleAudio() {
        _audioEnabled.value = !_audioEnabled.value
        if (_audioEnabled.value && _isAnalyzing.value) {
            startAudioCapture()
        } else {
            stopAudioCapture()
        }
    }

    /**
     * Called by CameraX ImageAnalysis analyzer for each frame.
     */
    fun onImageCaptured(imageProxy: ImageProxy) {
        if (!_isAnalyzing.value) {
            imageProxy.close()
            return
        }

        // Throttle frame rate
        val now = System.currentTimeMillis()
        if (now - lastFrameTime < FRAME_INTERVAL_MS) {
            imageProxy.close()
            return
        }
        lastFrameTime = now

        viewModelScope.launch(Dispatchers.Default) {
            try {
                val base64 = imageProxyToBase64(imageProxy)
                if (base64 != null) {
                    webSocketManager.sendFrame(base64, lastAudioBase64)

                    // Save to history
                    latestAnalysis.value?.let { analysis ->
                        if (analysis.dog_detected) {
                            val history = _analysisHistory.value.toMutableList()
                            history.add(0, analysis)
                            if (history.size > 50) history.removeAt(history.lastIndex)
                            _analysisHistory.value = history
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Frame processing error: ${e.message}")
            } finally {
                imageProxy.close()
            }
        }
    }

    private fun imageProxyToBase64(imageProxy: ImageProxy): String? {
        return try {
            val bitmap = imageProxy.toBitmap()
            val out = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
            Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP)
        } catch (e: Exception) {
            Log.e(TAG, "Image conversion error: ${e.message}")
            null
        }
    }

    @Suppress("MissingPermission")
    private fun startAudioCapture() {
        stopAudioCapture()

        try {
            val bufferSize = AudioRecord.getMinBufferSize(
                AUDIO_SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )

            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                AUDIO_SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize * 2
            )

            audioRecord?.startRecording()

            audioJob = viewModelScope.launch(Dispatchers.IO) {
                val buffer = ShortArray(AUDIO_SAMPLE_RATE) // 1 second of audio
                while (isActive && _audioEnabled.value) {
                    val readSize = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                    if (readSize > 0) {
                        // Convert to bytes and base64
                        val byteBuffer = ByteBuffer.allocate(readSize * 2).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                        for (i in 0 until readSize) {
                            byteBuffer.putShort(buffer[i])
                        }
                        lastAudioBase64 = Base64.encodeToString(
                            byteBuffer.array(), Base64.NO_WRAP
                        )
                    }
                }
            }

            Log.i(TAG, "Audio capture started")
        } catch (e: Exception) {
            Log.e(TAG, "Audio capture error: ${e.message}")
        }
    }

    private fun stopAudioCapture() {
        audioJob?.cancel()
        audioJob = null
        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (_: Exception) {}
        audioRecord = null
        lastAudioBase64 = null
    }

    override fun onCleared() {
        super.onCleared()
        webSocketManager.destroy()
        stopAudioCapture()
    }
}
