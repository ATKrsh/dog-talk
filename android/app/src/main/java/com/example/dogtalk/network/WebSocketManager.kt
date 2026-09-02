package com.example.dogtalk.network

import android.util.Log
import com.example.dogtalk.model.DogAnalysis
import com.google.gson.Gson
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import okhttp3.*
import java.util.concurrent.TimeUnit

/**
 * Manages WebSocket connection to the Dog Talk AI server.
 * Handles connection, reconnection, sending frames, and receiving analysis results.
 */
class WebSocketManager {

    companion object {
        private const val TAG = "WebSocketManager"
        private const val RECONNECT_DELAY_MS = 3000L
        private const val MAX_RECONNECT_ATTEMPTS = 10
    }

    enum class ConnectionState {
        DISCONNECTED, CONNECTING, CONNECTED, RECONNECTING, ERROR
    }

    private val gson = Gson()
    private var webSocket: WebSocket? = null
    private var client: OkHttpClient? = null
    private var serverUrl: String = ""
    private var reconnectAttempts = 0
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    // State flows
    private val _connectionState = MutableStateFlow(ConnectionState.DISCONNECTED)
    val connectionState: StateFlow<ConnectionState> = _connectionState.asStateFlow()

    private val _latestAnalysis = MutableStateFlow<DogAnalysis?>(null)
    val latestAnalysis: StateFlow<DogAnalysis?> = _latestAnalysis.asStateFlow()

    private val _serverInfo = MutableStateFlow<String>("")
    val serverInfo: StateFlow<String> = _serverInfo.asStateFlow()

    private val _latencyMs = MutableStateFlow(0L)
    val latencyMs: StateFlow<Long> = _latencyMs.asStateFlow()

    fun connect(serverIp: String, port: Int = 8765) {
        serverUrl = "ws://$serverIp:$port/ws"
        reconnectAttempts = 0
        doConnect()
    }

    private fun doConnect() {
        _connectionState.value = ConnectionState.CONNECTING
        
        client?.dispatcher?.cancelAll()
        client = OkHttpClient.Builder()
            .readTimeout(0, TimeUnit.MILLISECONDS) // No timeout for WebSocket
            .pingInterval(15, TimeUnit.SECONDS)
            .build()

        val request = Request.Builder()
            .url(serverUrl)
            .build()

        Log.i(TAG, "Connecting to $serverUrl")

        webSocket = client?.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.i(TAG, "Connected to server")
                _connectionState.value = ConnectionState.CONNECTED
                reconnectAttempts = 0
                _serverInfo.value = serverUrl
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val json = gson.fromJson(text, Map::class.java) as Map<*, *>
                    val type = json["type"] as? String ?: ""

                    when (type) {
                        "analysis" -> {
                            val analysis = gson.fromJson(text, DogAnalysis::class.java)
                            _latestAnalysis.value = analysis
                            _latencyMs.value = analysis.processing_time_ms.toLong()
                        }
                        "ping" -> {
                            webSocket.send("""{"type":"pong"}""")
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to parse message: ${e.message}")
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket failure: ${t.message}")
                _connectionState.value = ConnectionState.ERROR
                attemptReconnect()
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.i(TAG, "WebSocket closed: $reason")
                _connectionState.value = ConnectionState.DISCONNECTED
            }
        })
    }

    private fun attemptReconnect() {
        if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
            Log.e(TAG, "Max reconnect attempts reached")
            _connectionState.value = ConnectionState.ERROR
            return
        }

        reconnectAttempts++
        _connectionState.value = ConnectionState.RECONNECTING

        scope.launch {
            delay(RECONNECT_DELAY_MS)
            Log.i(TAG, "Reconnect attempt $reconnectAttempts/$MAX_RECONNECT_ATTEMPTS")
            doConnect()
        }
    }

    /**
     * Send a frame (image + audio) to the server for analysis.
     */
    fun sendFrame(imageBase64: String, audioBase64: String? = null) {
        if (_connectionState.value != ConnectionState.CONNECTED) return

        val payload = buildString {
            append("""{"type":"frame","image":"""")
            append(imageBase64)
            append('"')
            if (audioBase64 != null) {
                append(""","audio":"""")
                append(audioBase64)
                append('"')
            }
            append(""","timestamp":""")
            append(System.currentTimeMillis() / 1000)
            append('}')
        }

        webSocket?.send(payload)
    }

    fun disconnect() {
        webSocket?.close(1000, "Client disconnect")
        client?.dispatcher?.cancelAll()
        _connectionState.value = ConnectionState.DISCONNECTED
        _latestAnalysis.value = null
    }

    fun destroy() {
        disconnect()
        scope.cancel()
    }
}
