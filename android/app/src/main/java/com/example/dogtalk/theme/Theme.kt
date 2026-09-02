package com.example.dogtalk.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val DogTalkColorScheme = darkColorScheme(
    primary = ElectricCyan,
    onPrimary = DeepNavy,
    primaryContainer = MidNavy,
    onPrimaryContainer = ElectricCyanLight,
    secondary = VividViolet,
    onSecondary = DeepNavy,
    secondaryContainer = CardDark,
    onSecondaryContainer = VividVioletLight,
    tertiary = NeonGreen,
    onTertiary = DeepNavy,
    background = DeepNavy,
    onBackground = TextPrimary,
    surface = DarkNavy,
    onSurface = TextPrimary,
    surfaceVariant = MidNavy,
    onSurfaceVariant = TextSecondary,
    outline = GlassBorder,
    error = CoralRed,
    onError = DeepNavy,
)

@Composable
fun DogTalkTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = DogTalkColorScheme,
        typography = Typography,
        content = content
    )
}
