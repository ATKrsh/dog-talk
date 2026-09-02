package com.example.dogtalk.ui.main

import android.Manifest
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.dogtalk.model.DogAnalysis
import com.example.dogtalk.model.EmotionResult
import com.example.dogtalk.network.WebSocketManager
import com.example.dogtalk.theme.*
import com.example.dogtalk.viewmodel.DogTalkViewModel
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.rememberMultiplePermissionsState
import java.util.concurrent.Executors

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun MainScreen(
    onItemClick: (Any) -> Unit = {},
    modifier: Modifier = Modifier,
    vm: DogTalkViewModel = viewModel()
) {
    val permissionsState = rememberMultiplePermissionsState(
        permissions = listOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        )
    )

    val connectionState by vm.connectionState.collectAsState()
    val analysis by vm.latestAnalysis.collectAsState()
    val isAnalyzing by vm.isAnalyzing.collectAsState()
    val serverIp by vm.serverIp.collectAsState()
    val latency by vm.latencyMs.collectAsState()
    val audioEnabled by vm.audioEnabled.collectAsState()

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(DeepNavy)
    ) {
        if (permissionsState.allPermissionsGranted) {
            when (connectionState) {
                WebSocketManager.ConnectionState.CONNECTED -> {
                    // Main camera + analysis view
                    CameraAnalysisScreen(
                        vm = vm,
                        analysis = analysis,
                        isAnalyzing = isAnalyzing,
                        latency = latency,
                        audioEnabled = audioEnabled,
                        connectionState = connectionState
                    )
                }
                else -> {
                    // Connection screen
                    ConnectionScreen(
                        serverIp = serverIp,
                        connectionState = connectionState,
                        onIpChange = { vm.updateServerIp(it) },
                        onConnect = { vm.connect() },
                        onDisconnect = { vm.disconnect() }
                    )
                }
            }
        } else {
            // Permission request screen
            PermissionScreen(
                onRequestPermissions = { permissionsState.launchMultiplePermissionRequest() }
            )
        }
    }
}

@Composable
private fun PermissionScreen(onRequestPermissions: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // Animated paw icon
        val infiniteTransition = rememberInfiniteTransition(label = "paw")
        val scale by infiniteTransition.animateFloat(
            initialValue = 0.9f,
            targetValue = 1.1f,
            animationSpec = infiniteRepeatable(
                animation = tween(1000, easing = EaseInOutCubic),
                repeatMode = RepeatMode.Reverse
            ),
            label = "scale"
        )

        Text("🐾", fontSize = (80 * scale).sp)

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            "Dog Talk",
            fontSize = 36.sp,
            fontWeight = FontWeight.Bold,
            color = TextPrimary
        )

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            "Understand what your dog is saying",
            fontSize = 16.sp,
            color = TextSecondary,
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(48.dp))

        Text(
            "We need camera and microphone access to analyze dog behavior",
            fontSize = 14.sp,
            color = TextTertiary,
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = onRequestPermissions,
            colors = ButtonDefaults.buttonColors(containerColor = ElectricCyan),
            shape = RoundedCornerShape(16.dp),
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp)
        ) {
            Icon(Icons.Default.CameraAlt, contentDescription = null, tint = DeepNavy)
            Spacer(modifier = Modifier.width(8.dp))
            Text("Grant Permissions", color = DeepNavy, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
private fun ConnectionScreen(
    serverIp: String,
    connectionState: WebSocketManager.ConnectionState,
    onIpChange: (String) -> Unit,
    onConnect: () -> Unit,
    onDisconnect: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // Animated dog icon
        val infiniteTransition = rememberInfiniteTransition(label = "dog")
        val rotation by infiniteTransition.animateFloat(
            initialValue = -5f,
            targetValue = 5f,
            animationSpec = infiniteRepeatable(
                animation = tween(2000, easing = EaseInOutCubic),
                repeatMode = RepeatMode.Reverse
            ),
            label = "rotation"
        )

        Text("🐕", fontSize = 72.sp)

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            "Dog Talk",
            fontSize = 40.sp,
            fontWeight = FontWeight.ExtraBold,
            color = TextPrimary
        )

        Text(
            "AI Dog Behavior Interpreter",
            fontSize = 14.sp,
            color = ElectricCyan,
            fontWeight = FontWeight.Medium
        )

        Spacer(modifier = Modifier.height(48.dp))

        // Server IP input
        GlassCard(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(20.dp)) {
                Text(
                    "Server Connection",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = TextPrimary
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    "Enter your PC's IP address",
                    fontSize = 12.sp,
                    color = TextTertiary
                )

                Spacer(modifier = Modifier.height(16.dp))

                OutlinedTextField(
                    value = serverIp,
                    onValueChange = onIpChange,
                    label = { Text("Server IP") },
                    placeholder = { Text("192.168.1.100") },
                    singleLine = true,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = ElectricCyan,
                        unfocusedBorderColor = GlassBorder,
                        focusedLabelColor = ElectricCyan,
                        cursorColor = ElectricCyan,
                        focusedTextColor = TextPrimary,
                        unfocusedTextColor = TextPrimary
                    ),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Connection status
        if (connectionState == WebSocketManager.ConnectionState.CONNECTING ||
            connectionState == WebSocketManager.ConnectionState.RECONNECTING
        ) {
            CircularProgressIndicator(
                color = ElectricCyan,
                modifier = Modifier.size(40.dp)
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                if (connectionState == WebSocketManager.ConnectionState.RECONNECTING)
                    "Reconnecting..." else "Connecting...",
                color = TextSecondary
            )
        } else if (connectionState == WebSocketManager.ConnectionState.ERROR) {
            Text("❌ Connection failed. Check server IP and ensure server is running.",
                color = CoralRed, fontSize = 13.sp, textAlign = TextAlign.Center)
        }

        Spacer(modifier = Modifier.height(24.dp))

        // Connect button
        Button(
            onClick = {
                if (connectionState == WebSocketManager.ConnectionState.DISCONNECTED ||
                    connectionState == WebSocketManager.ConnectionState.ERROR
                ) onConnect() else onDisconnect()
            },
            colors = ButtonDefaults.buttonColors(
                containerColor = if (connectionState == WebSocketManager.ConnectionState.DISCONNECTED ||
                    connectionState == WebSocketManager.ConnectionState.ERROR)
                    ElectricCyan else CoralRed
            ),
            shape = RoundedCornerShape(16.dp),
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp)
        ) {
            Icon(
                if (connectionState == WebSocketManager.ConnectionState.DISCONNECTED ||
                    connectionState == WebSocketManager.ConnectionState.ERROR)
                    Icons.Default.Wifi else Icons.Default.WifiOff,
                contentDescription = null,
                tint = DeepNavy
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                if (connectionState == WebSocketManager.ConnectionState.DISCONNECTED ||
                    connectionState == WebSocketManager.ConnectionState.ERROR)
                    "Connect to Server" else "Disconnect",
                color = DeepNavy,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

@Composable
private fun CameraAnalysisScreen(
    vm: DogTalkViewModel,
    analysis: DogAnalysis?,
    isAnalyzing: Boolean,
    latency: Long,
    audioEnabled: Boolean,
    connectionState: WebSocketManager.ConnectionState
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    Box(modifier = Modifier.fillMaxSize()) {
        // Camera Preview (full screen)
        AndroidView(
            factory = { ctx ->
                PreviewView(ctx).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER

                    val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                    cameraProviderFuture.addListener({
                        val cameraProvider = cameraProviderFuture.get()

                        val preview = Preview.Builder().build().also {
                            it.surfaceProvider = surfaceProvider
                        }

                        val imageAnalysis = ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .build()
                            .apply {
                                setAnalyzer(cameraExecutor) { imageProxy ->
                                    vm.onImageCaptured(imageProxy)
                                }
                            }

                        try {
                            cameraProvider.unbindAll()
                            cameraProvider.bindToLifecycle(
                                lifecycleOwner,
                                CameraSelector.DEFAULT_BACK_CAMERA,
                                preview,
                                imageAnalysis
                            )
                        } catch (e: Exception) {
                            e.printStackTrace()
                        }
                    }, ContextCompat.getMainExecutor(ctx))
                }
            },
            modifier = Modifier.fillMaxSize()
        )

        // Gradient overlay at top
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(120.dp)
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            DeepNavy.copy(alpha = 0.8f),
                            Color.Transparent
                        )
                    )
                )
        )

        // Top bar
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .statusBarsPadding()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            // Connection status badge
            StatusBadge(connectionState)

            // Latency indicator
            if (latency > 0) {
                GlassPill {
                    Text(
                        "${latency}ms",
                        fontSize = 11.sp,
                        color = if (latency < 200) NeonGreen else if (latency < 500) WarmAmber else CoralRed,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }
        }

        // Bottom analysis overlay
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.BottomCenter)
        ) {
            // Analysis results (animated slide-up)
            AnimatedVisibility(
                visible = analysis != null && analysis.dog_detected,
                enter = slideInVertically(initialOffsetY = { it }),
                exit = slideOutVertically(targetOffsetY = { it })
            ) {
                analysis?.let { AnalysisOverlay(it) }
            }

            // No dog message
            AnimatedVisibility(
                visible = analysis != null && !analysis.dog_detected && isAnalyzing
            ) {
                GlassCard(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                ) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("🔍", fontSize = 24.sp)
                        Spacer(modifier = Modifier.width(12.dp))
                        Text(
                            "Point your camera at a dog to start analyzing...",
                            color = TextSecondary,
                            fontSize = 14.sp
                        )
                    }
                }
            }

            // Control bar
            ControlBar(
                isAnalyzing = isAnalyzing,
                audioEnabled = audioEnabled,
                onToggleAnalysis = {
                    if (isAnalyzing) vm.stopAnalysis() else vm.startAnalysis()
                },
                onToggleAudio = { vm.toggleAudio() },
                onDisconnect = { vm.disconnect() }
            )
        }
    }
}

@Composable
private fun AnalysisOverlay(analysis: DogAnalysis) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                Brush.verticalGradient(
                    colors = listOf(Color.Transparent, DeepNavy.copy(alpha = 0.95f))
                )
            )
            .padding(horizontal = 16.dp, vertical = 12.dp)
    ) {
        // Emotions row
        if (analysis.emotions.isNotEmpty()) {
            LazyRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                items(analysis.emotions) { emotion ->
                    EmotionChip(emotion)
                }
            }
            Spacer(modifier = Modifier.height(12.dp))
        }

        // Interpretation card
        GlassCard(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                // "What they're saying" header
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("💬", fontSize = 18.sp)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        "What They're Saying",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        color = ElectricCyan
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    analysis.interpretation,
                    fontSize = 14.sp,
                    color = TextPrimary,
                    lineHeight = 20.sp,
                    maxLines = 3,
                    overflow = TextOverflow.Ellipsis
                )

                // Prediction
                if (analysis.prediction.action.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(12.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("🔮", fontSize = 14.sp)
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            "Next: ${analysis.prediction.action}",
                            fontSize = 12.sp,
                            color = TextSecondary,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.weight(1f)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        ConfidenceBadge(analysis.prediction.confidence)
                    }
                }

                // Warning
                if (analysis.prediction.warning_level != "none") {
                    Spacer(modifier = Modifier.height(8.dp))
                    WarningBanner(
                        level = analysis.prediction.warning_level,
                        safeToApproach = analysis.prediction.safe_to_approach
                    )
                }
            }
        }

        // Body language details row
        Spacer(modifier = Modifier.height(8.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // Tail
            MiniInfoCard(
                emoji = "🐕",
                label = "Tail",
                value = analysis.body_language.tail.label.ifEmpty { analysis.body_language.tail.position },
                modifier = Modifier.weight(1f)
            )
            // Ears
            MiniInfoCard(
                emoji = "👂",
                label = "Ears",
                value = analysis.body_language.ears.label.ifEmpty { analysis.body_language.ears.position },
                modifier = Modifier.weight(1f)
            )
            // Sound
            MiniInfoCard(
                emoji = "🔊",
                label = "Sound",
                value = analysis.vocalization.label.ifEmpty { analysis.vocalization.type.replace("_", " ") },
                modifier = Modifier.weight(1f)
            )
        }
    }
}

@Composable
private fun EmotionChip(emotion: EmotionResult) {
    val chipColor = try {
        Color(android.graphics.Color.parseColor(emotion.color))
    } catch (_: Exception) {
        ElectricCyan
    }

    Surface(
        shape = RoundedCornerShape(20.dp),
        color = chipColor.copy(alpha = 0.2f),
        modifier = Modifier.border(1.dp, chipColor.copy(alpha = 0.4f), RoundedCornerShape(20.dp))
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(emotion.emoji, fontSize = 16.sp)
            Spacer(modifier = Modifier.width(4.dp))
            Text(
                emotion.name,
                fontSize = 12.sp,
                fontWeight = FontWeight.SemiBold,
                color = chipColor
            )
            Spacer(modifier = Modifier.width(4.dp))
            Text(
                "${(emotion.confidence * 100).toInt()}%",
                fontSize = 11.sp,
                color = chipColor.copy(alpha = 0.7f)
            )
        }
    }
}

@Composable
private fun ConfidenceBadge(confidence: Double) {
    val pct = (confidence * 100).toInt()
    val color = when {
        pct >= 80 -> NeonGreen
        pct >= 60 -> WarmAmber
        else -> CoralRed
    }
    Surface(
        shape = RoundedCornerShape(8.dp),
        color = color.copy(alpha = 0.2f)
    ) {
        Text(
            "$pct%",
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold,
            color = color,
            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
        )
    }
}

@Composable
private fun WarningBanner(level: String, safeToApproach: Boolean) {
    val (bgColor, text) = when (level) {
        "low" -> Pair(WarmAmber, "⚠️ Mild caution — dog may be uncertain")
        "moderate" -> Pair(WarningModerate, "⚠️ Caution — approach carefully")
        "high" -> Pair(WarningHigh, "🚨 Warning — dog may bite if approached")
        "critical" -> Pair(WarningCritical, "🚨 DANGER — do NOT approach!")
        else -> return
    }

    Surface(
        shape = RoundedCornerShape(8.dp),
        color = bgColor.copy(alpha = 0.15f),
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, bgColor.copy(alpha = 0.3f), RoundedCornerShape(8.dp))
    ) {
        Text(
            text,
            fontSize = 12.sp,
            fontWeight = FontWeight.SemiBold,
            color = bgColor,
            modifier = Modifier.padding(8.dp)
        )
    }
}

@Composable
private fun MiniInfoCard(emoji: String, label: String, value: String, modifier: Modifier = Modifier) {
    Surface(
        shape = RoundedCornerShape(12.dp),
        color = GlassWhite,
        modifier = modifier
    ) {
        Column(
            modifier = Modifier.padding(8.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(emoji, fontSize = 16.sp)
            Spacer(modifier = Modifier.height(2.dp))
            Text(label, fontSize = 10.sp, color = TextTertiary)
            Text(
                value.replaceFirstChar { it.uppercase() },
                fontSize = 11.sp,
                fontWeight = FontWeight.SemiBold,
                color = TextPrimary,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
private fun ControlBar(
    isAnalyzing: Boolean,
    audioEnabled: Boolean,
    onToggleAnalysis: () -> Unit,
    onToggleAudio: () -> Unit,
    onDisconnect: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .navigationBarsPadding()
            .padding(horizontal = 24.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.SpaceEvenly,
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Audio toggle
        IconButton(
            onClick = onToggleAudio,
            modifier = Modifier
                .size(48.dp)
                .clip(CircleShape)
                .background(if (audioEnabled) GlassWhite else CoralRed.copy(alpha = 0.3f))
        ) {
            Icon(
                if (audioEnabled) Icons.Default.Mic else Icons.Default.MicOff,
                contentDescription = "Toggle Audio",
                tint = if (audioEnabled) TextPrimary else CoralRed
            )
        }

        // Main analyze button (large, pulsating when active)
        val infiniteTransition = rememberInfiniteTransition(label = "pulse")
        val pulseScale by infiniteTransition.animateFloat(
            initialValue = 1f,
            targetValue = if (isAnalyzing) 1.08f else 1f,
            animationSpec = infiniteRepeatable(
                animation = tween(800, easing = EaseInOutCubic),
                repeatMode = RepeatMode.Reverse
            ),
            label = "pulse"
        )

        FloatingActionButton(
            onClick = onToggleAnalysis,
            containerColor = if (isAnalyzing) CoralRed else ElectricCyan,
            contentColor = DeepNavy,
            modifier = Modifier.size((72 * pulseScale).dp),
            shape = CircleShape
        ) {
            Icon(
                if (isAnalyzing) Icons.Default.Stop else Icons.Default.PlayArrow,
                contentDescription = if (isAnalyzing) "Stop" else "Start",
                modifier = Modifier.size(36.dp)
            )
        }

        // Disconnect
        IconButton(
            onClick = onDisconnect,
            modifier = Modifier
                .size(48.dp)
                .clip(CircleShape)
                .background(GlassWhite)
        ) {
            Icon(
                Icons.Default.LinkOff,
                contentDescription = "Disconnect",
                tint = TextSecondary
            )
        }
    }
}

@Composable
private fun StatusBadge(state: WebSocketManager.ConnectionState) {
    val (color, text) = when (state) {
        WebSocketManager.ConnectionState.CONNECTED -> Pair(NeonGreen, "Connected")
        WebSocketManager.ConnectionState.CONNECTING -> Pair(WarmAmber, "Connecting...")
        WebSocketManager.ConnectionState.RECONNECTING -> Pair(WarmAmber, "Reconnecting...")
        WebSocketManager.ConnectionState.ERROR -> Pair(CoralRed, "Error")
        WebSocketManager.ConnectionState.DISCONNECTED -> Pair(TextTertiary, "Disconnected")
    }

    GlassPill {
        Box(
            modifier = Modifier
                .size(8.dp)
                .clip(CircleShape)
                .background(color)
        )
        Spacer(modifier = Modifier.width(6.dp))
        Text(text, fontSize = 11.sp, color = color, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
private fun GlassCard(
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit
) {
    Surface(
        modifier = modifier
            .border(1.dp, GlassBorder, RoundedCornerShape(16.dp)),
        shape = RoundedCornerShape(16.dp),
        color = CardDark.copy(alpha = 0.85f),
    ) {
        content()
    }
}

@Composable
private fun GlassPill(content: @Composable RowScope.() -> Unit) {
    Surface(
        shape = RoundedCornerShape(20.dp),
        color = GlassWhite,
        modifier = Modifier.border(1.dp, GlassBorder, RoundedCornerShape(20.dp))
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
            verticalAlignment = Alignment.CenterVertically,
            content = content
        )
    }
}
