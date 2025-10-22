package com.aiservice.poseul.service

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.delay
import java.net.HttpURLConnection
import java.net.URL
import java.io.OutputStreamWriter
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import android.util.Log

class ModelService {
    
    companion object {
        private const val SERVER_URL = "http://10.0.2.2:5000" // 현재 PC의 Wi-Fi IP 주소
        private const val PREDICT_ENDPOINT = "/predict"
        private const val HEALTH_ENDPOINT = "/health"
        private const val MODEL_INFO_ENDPOINT = "/model_info"
    }
    
    private val gson = Gson()
    
    data class PredictionRequest(
        @SerializedName("hr_mean") val hrMean: Double,
        @SerializedName("hrv_sdnn") val hrvSdnn: Double,
        @SerializedName("bmi") val bmi: Double,
        @SerializedName("mean_sa02") val meanSa02: Double,
        @SerializedName("gender") val gender: String,
        @SerializedName("age") val age: Int
    )
    
    data class PredictionResponse(
        @SerializedName("success") val success: Boolean,
        @SerializedName("predicted_temperature") val predictedTemperature: Double,
        @SerializedName("temperature_category") val temperatureCategory: String,
        @SerializedName("input_data") val inputData: PredictionRequest?,
        @SerializedName("error") val error: String?
    )
    
    data class HealthResponse(
        @SerializedName("status") val status: String,
        @SerializedName("model_loaded") val modelLoaded: Boolean
    )
    
    suspend fun predictTemperature(
        heartRate: Int,
        hrvSdnn: Double,
        bmi: Double,
        meanSa02: Double,
        userGender: String,
        age: Int
    ): PredictionResult = withContext(Dispatchers.IO) {
        try {
            // 서버 상태 확인
            if (!checkServerHealth()) {
                return@withContext PredictionResult.Error("서버에 연결할 수 없습니다.")
            }
            
            // 예측 요청 데이터 준비
            val request = PredictionRequest(
                hrMean = heartRate.toDouble(),
                hrvSdnn = hrvSdnn,
                bmi = bmi,
                meanSa02 = meanSa02,
                gender = if (userGender.lowercase() == "female") "F" else "M",
                age = age
            )
            
            // HTTP 요청 수행
            val response = makePredictionRequest(request)
            
            if (response.success) {
                PredictionResult.Success(
                    temperature = response.predictedTemperature.toFloat(),
                    category = response.temperatureCategory
                )
            } else {
                PredictionResult.Error(response.error ?: "예측 실패")
            }
            
        } catch (e: Exception) {
            PredictionResult.Error("예측 중 오류 발생: ${e.message}")
        }
    }
    
    private suspend fun checkServerHealth(): Boolean = withContext(Dispatchers.IO) {
        try {
            val fullUrl = "$SERVER_URL$HEALTH_ENDPOINT"
            Log.i("ModelService", "🔍 [HEALTH CHECK] 서버 상태 확인 시작")
            Log.d("ModelService", "🌐 [HEALTH CHECK] 요청 URL: $fullUrl")
            
            val url = URL(fullUrl)
            val connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "GET"
            connection.connectTimeout = 5000
            connection.readTimeout = 5000
            
            Log.d("ModelService", "🔗 [HEALTH CHECK] HTTP 연결 설정 완료")
            Log.d("ModelService", "⏱️ [HEALTH CHECK] 연결 시도 중... (타임아웃: 5초)")
            
            val responseCode = connection.responseCode
            Log.i("ModelService", "📡 [HEALTH CHECK] HTTP 응답 코드: $responseCode")
            
            if (responseCode == HttpURLConnection.HTTP_OK) {
                val response = connection.inputStream.bufferedReader().use { it.readText() }
                Log.i("ModelService", "✅ [HEALTH CHECK] 서버 응답 수신: $response")
                
                val healthResponse = gson.fromJson(response, HealthResponse::class.java)
                Log.i("ModelService", "🏥 [HEALTH CHECK] 모델 로드 상태: ${healthResponse.modelLoaded}")
                Log.i("ModelService", "🎯 [HEALTH CHECK] 서버 상태: ${healthResponse.status}")
                
                if (healthResponse.modelLoaded) {
                    Log.i("ModelService", "🎉 [HEALTH CHECK] 서버 연결 성공! 모델 준비 완료")
                } else {
                    Log.w("ModelService", "⚠️ [HEALTH CHECK] 서버는 연결되었지만 모델이 로드되지 않음")
                }
                
                healthResponse.modelLoaded
            } else {
                Log.e("ModelService", "❌ [HEALTH CHECK] 서버 응답 실패: $responseCode")
                Log.e("ModelService", "❌ [HEALTH CHECK] 응답 메시지: ${connection.responseMessage}")
                false
            }
        } catch (e: Exception) {
            Log.e("ModelService", "💥 [HEALTH CHECK] 서버 연결 오류 발생", e)
            Log.e("ModelService", "💥 [HEALTH CHECK] 오류 메시지: ${e.message}")
            Log.e("ModelService", "💥 [HEALTH CHECK] 오류 타입: ${e.javaClass.simpleName}")
            false
        }
    }
    
    private suspend fun makePredictionRequest(request: PredictionRequest): PredictionResponse = withContext(Dispatchers.IO) {
        val predictUrl = "$SERVER_URL$PREDICT_ENDPOINT"
        Log.i("ModelService", "🚀 [PREDICTION] 예측 요청 시작")
        Log.d("ModelService", "🌐 [PREDICTION] 요청 URL: $predictUrl")
        Log.i("ModelService", "📊 [PREDICTION] 요청 데이터: HR=${request.hrMean}, HRV=${request.hrvSdnn}, BMI=${request.bmi}, SaO2=${request.meanSa02}, Gender=${request.gender}, Age=${request.age}")
        
        val url = URL(predictUrl)
        val connection = url.openConnection() as HttpURLConnection
        
        connection.requestMethod = "POST"
        connection.setRequestProperty("Content-Type", "application/json")
        connection.doOutput = true
        connection.connectTimeout = 10000
        connection.readTimeout = 10000
        
        Log.d("ModelService", "🔗 [PREDICTION] HTTP POST 연결 설정 완료")
        Log.d("ModelService", "⏱️ [PREDICTION] 연결 시도 중... (타임아웃: 10초)")
        
        // 요청 데이터 전송
        val requestJson = gson.toJson(request)
        Log.d("ModelService", "📤 [PREDICTION] JSON 데이터 전송: $requestJson")
        
        val outputStream = connection.outputStream
        val writer = OutputStreamWriter(outputStream)
        writer.write(requestJson)
        writer.flush()
        writer.close()
        
        Log.d("ModelService", "✅ [PREDICTION] 요청 데이터 전송 완료")
        
        // 응답 읽기
        val responseCode = connection.responseCode
        Log.i("ModelService", "📡 [PREDICTION] HTTP 응답 코드: $responseCode")
        
        val responseText = if (responseCode == HttpURLConnection.HTTP_OK) {
            val response = connection.inputStream.bufferedReader().use { it.readText() }
            Log.i("ModelService", "✅ [PREDICTION] 서버 응답 수신 성공")
            Log.d("ModelService", "📥 [PREDICTION] 응답 내용: $response")
            response
        } else {
            val errorResponse = connection.errorStream.bufferedReader().use { it.readText() }
            Log.e("ModelService", "❌ [PREDICTION] 서버 응답 실패")
            Log.e("ModelService", "❌ [PREDICTION] 에러 응답: $errorResponse")
            errorResponse
        }
        
        val predictionResponse = gson.fromJson(responseText, PredictionResponse::class.java)
        
        if (predictionResponse.success) {
            Log.i("ModelService", "🎉 [PREDICTION] 예측 성공!")
            Log.i("ModelService", "🌡️ [PREDICTION] 예측된 온도: ${predictionResponse.predictedTemperature}°C")
            Log.i("ModelService", "🏷️ [PREDICTION] 온도 카테고리: ${predictionResponse.temperatureCategory}")
        } else {
            Log.e("ModelService", "❌ [PREDICTION] 예측 실패: ${predictionResponse.error}")
        }
        
        predictionResponse
    }
    
    suspend fun getModelInfo(): String = withContext(Dispatchers.IO) {
        try {
            val url = URL("$SERVER_URL$MODEL_INFO_ENDPOINT")
            val connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "GET"
            connection.connectTimeout = 5000
            connection.readTimeout = 5000
            
            val responseCode = connection.responseCode
            if (responseCode == HttpURLConnection.HTTP_OK) {
                connection.inputStream.bufferedReader().use { it.readText() }
            } else {
                "모델 정보를 가져올 수 없습니다."
            }
        } catch (e: Exception) {
            "모델 정보 조회 실패: ${e.message}"
        }
    }
    
    suspend fun loadModel(): Boolean = withContext(Dispatchers.IO) {
        // 서버 기반 모델이므로 서버 상태만 확인
        checkServerHealth()
    }
    
    suspend fun retrainModel(newData: List<TrainingData>): Boolean = withContext(Dispatchers.IO) {
        try {
            // 실제 구현에서는 새로운 데이터로 모델을 재학습합니다
            // 1. 데이터 전처리
            // 2. 모델 학습
            // 3. 모델 검증
            // 4. 모델 저장
            
            // 시뮬레이션을 위한 대기
            delay(5000)
            true
        } catch (e: Exception) {
            false
        }
    }
}

sealed class PredictionResult {
    data class Success(
        val temperature: Float,
        val category: String
    ) : PredictionResult()
    
    data class Error(
        val message: String
    ) : PredictionResult()
}

data class TrainingData(
    val heartRate: Int,
    val roomTemperature: Float,
    val humidity: Float,
    val userAge: Int,
    val userGender: String,
    val actualTemperature: Float
)
