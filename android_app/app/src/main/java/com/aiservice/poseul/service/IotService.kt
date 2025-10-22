package com.aiservice.poseul.service

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.delay

class IotService {
    
    suspend fun getIotDevices(): List<IotDeviceInfo> = withContext(Dispatchers.IO) {
        // 실제 구현에서는 IOT 기기 API를 호출합니다
        delay(1000) // 네트워크 지연 시뮬레이션
        
        listOf(
            IotDeviceInfo(
                id = "ac_001",
                name = "거실 에어컨",
                type = "air_conditioner",
                isOnline = true,
                currentTemperature = 24,
                targetTemperature = 26,
                powerOn = true
            ),
            IotDeviceInfo(
                id = "sensor_001",
                name = "온도 센서",
                type = "temperature_sensor",
                isOnline = true,
                currentTemperature = 23,
                targetTemperature = null,
                powerOn = true
            ),
            IotDeviceInfo(
                id = "humidifier_001",
                name = "가습기",
                type = "humidifier",
                isOnline = false,
                currentTemperature = null,
                targetTemperature = null,
                powerOn = false
            )
        )
    }
    
    suspend fun updateDeviceTemperature(deviceId: String, targetTemperature: Int): Boolean = withContext(Dispatchers.IO) {
        // 실제 구현에서는 IOT 기기 제어 API를 호출합니다
        delay(500)
        true
    }
    
    suspend fun toggleDevicePower(deviceId: String): Boolean = withContext(Dispatchers.IO) {
        // 실제 구현에서는 IOT 기기 전원 제어 API를 호출합니다
        delay(500)
        true
    }
}

data class IotDeviceInfo(
    val id: String,
    val name: String,
    val type: String,
    val isOnline: Boolean,
    val currentTemperature: Int?,
    val targetTemperature: Int?,
    val powerOn: Boolean
)
