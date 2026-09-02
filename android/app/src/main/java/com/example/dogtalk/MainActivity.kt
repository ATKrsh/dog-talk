package com.example.dogtalk

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.example.dogtalk.theme.DogTalkTheme
import com.example.dogtalk.ui.main.MainScreen
import com.example.dogtalk.viewmodel.DogTalkViewModel

class MainActivity : ComponentActivity() {
    private val viewModel: DogTalkViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            DogTalkTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen(vm = viewModel)
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        viewModel.disconnect()
    }
}
