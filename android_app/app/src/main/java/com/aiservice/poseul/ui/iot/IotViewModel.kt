package com.aiservice.poseul.ui.iot

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.*

data class IotDevice(
    val id: String,
    val name: String,
    val type: String,
    val isOnline: Boolean,
    val currentTemperature: Int? = null,
    val targetTemperature: Int? = null
)

class IotViewModel : ViewModel() {
    
    private val _iotDevices = MutableLiveData<List<IotDevice>>()
    val iotDevices: LiveData<List<IotDevice>> = _iotDevices

    private val _isLoading = MutableLiveData<Boolean>()
    val isLoading: LiveData<Boolean> = _isLoading

    private val _errorMessage = MutableLiveData<String>()
    val errorMessage: LiveData<String> = _errorMessage

    private val viewModelScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    init {
        loadIotDevices()
    }

    private fun loadIotDevices() {
        viewModelScope.launch {
            _isLoading.value = true
            try {
                // 실제 구현에서는 API 호출을 통해 IOT 기기 정보를 가져옵니다
                simulateIotDevices()
            } catch (e: Exception) {
                _errorMessage.value = "IOT 기기 정보를 불러오는 중 오류가 발생했습니다: ${e.message}"
            } finally {
                _isLoading.value = false
            }
        }
    }

    private suspend fun simulateIotDevices() {
        delay(1500) // 1.5초 대기 (실제 로딩 시뮬레이션)
        
        val devices = listOf(
            IotDevice(
                id = "ac_001",
                name = "거실 에어컨",
                type = "air_conditioner",
                isOnline = true,
                currentTemperature = 24,
                targetTemperature = 26
            ),
            IotDevice(
                id = "sensor_001",
                name = "온도 센서",
                type = "temperature_sensor",
                isOnline = true
            ),
            IotDevice(
                id = "humidifier_001",
                name = "가습기",
                type = "humidifier",
                isOnline = false
            )
        )
        
        _iotDevices.value = devices
    }

    fun refreshDevices() {
        loadIotDevices()
    }

    override fun onCleared() {
        super.onCleared()
        viewModelScope.cancel()
    }
}
