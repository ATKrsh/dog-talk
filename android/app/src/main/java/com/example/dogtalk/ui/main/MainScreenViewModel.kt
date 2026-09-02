package com.example.dogtalk.ui.main

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Placeholder ViewModel from the template.
 * The actual ViewModel used is DogTalkViewModel in the viewmodel package.
 */
class MainScreenViewModel : ViewModel() {
    private val _uiState = MutableStateFlow("")
    val uiState: StateFlow<String> = _uiState.asStateFlow()
}
