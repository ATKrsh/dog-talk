package com.example.dogtalk.model

import kotlinx.serialization.Serializable

/**
 * Complete analysis result received from the server.
 */
@Serializable
data class DogAnalysis(
    val type: String = "analysis",
    val dog_detected: Boolean = false,
    val emotions: List<EmotionResult> = emptyList(),
    val body_language: BodyLanguage = BodyLanguage(),
    val vocalization: VocalizationInfo = VocalizationInfo(),
    val interpretation: String = "",
    val prediction: PredictionResult = PredictionResult(),
    val timestamp: Double = 0.0,
    val processing_time_ms: Double = 0.0
)

@Serializable
data class EmotionResult(
    val name: String = "",
    val confidence: Double = 0.0,
    val emoji: String = "❓",
    val color: String = "#9E9E9E"
)

@Serializable
data class BodyLanguage(
    val tail: TailInfo = TailInfo(),
    val ears: EarInfo = EarInfo(),
    val posture: PostureInfo = PostureInfo(),
    val hackles: HackleInfo = HackleInfo(),
    val body_tension: Double = 0.0,
    val head_height: String = "neutral",
    val weight_forward: Double = 0.5
)

@Serializable
data class TailInfo(
    val position: String = "unknown",
    val movement: String = "unknown",
    val meaning: String = "",
    val label: String = ""
)

@Serializable
data class EarInfo(
    val position: String = "unknown",
    val meaning: String = "",
    val label: String = ""
)

@Serializable
data class PostureInfo(
    val stance: String = "unknown",
    val meaning: String = "",
    val label: String = ""
)

@Serializable
data class HackleInfo(
    val raised: Boolean = false
)

@Serializable
data class VocalizationInfo(
    val type: String = "no_vocalization",
    val label: String = "",
    val meaning: String = "",
    val confidence: Double = 0.0,
    val is_dog_sound: Boolean = false
)

@Serializable
data class PredictionResult(
    val action: String = "Waiting for analysis...",
    val confidence: Double = 0.0,
    val warning_level: String = "none",
    val warning_color: String = "#4CAF50",
    val warning_description: String = "",
    val safe_to_approach: Boolean = true
)

/**
 * Server health/status response.
 */
@Serializable
data class ServerHealth(
    val status: String = "",
    val detector: Boolean = false,
    val audio: Boolean = false,
    val llm: Boolean = false,
    val knowledge_base: Boolean = false,
    val connected_clients: Int = 0,
    val total_frames: Int = 0,
    val avg_processing_ms: Double = 0.0
)
