package com.aiservice.poseul.ui.home

import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.aiservice.poseul.databinding.FragmentHomeBinding
import com.aiservice.poseul.service.ModelService
import com.aiservice.poseul.service.PredictionResult
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import kotlinx.coroutines.launch
import kotlin.random.Random

class HomeFragment : Fragment() {

    private var _binding: FragmentHomeBinding? = null
    private val binding get() = _binding!!
    private val modelService = ModelService()

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        Log.d("HomeFragment", "onCreateView 시작")
        _binding = FragmentHomeBinding.inflate(inflater, container, false)
        Log.d("HomeFragment", "onCreateView 완료")
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        Log.d("HomeFragment", "onViewCreated 시작")
        setupUI()
        setupHeartRateChart()
        Log.d("HomeFragment", "onViewCreated 완료")
    }

    private fun setupUI() {
        // 기본 온도 표시
        binding.temperatureValue.text = "35.0°C"
        binding.temperatureStatus.text = "쾌적함"
        binding.temperatureStatus.setTextColor(0xFF4CAF50.toInt())
        
        // 모델 테스트 버튼
        binding.testModelButton.setOnClickListener {
            Log.d("ModelTest", "🧪 모델 테스트 시작...")
            performModelPrediction()
        }
    }

    private fun setupHeartRateChart() {
        // 심박수 차트 설정
        binding.heartRateChart.description.isEnabled = false
        binding.heartRateChart.setTouchEnabled(true)
        binding.heartRateChart.isDragEnabled = true
        binding.heartRateChart.setScaleEnabled(true)
        binding.heartRateChart.setPinchZoom(true)
        
        // 초기 심박수 데이터 설정
        updateHeartRateChart()
    }

    private fun updateHeartRateChart() {
        // 임의의 심박수 데이터 생성 (60-100 bpm)
        val heartRateData = generateHeartRateData()
        
        val entries = heartRateData.mapIndexed { index, value ->
            Entry(index.toFloat(), value.toFloat())
        }
        
        val dataSet = LineDataSet(entries, "심박수").apply {
            color = 0xFFF44336.toInt()
            setCircleColor(0xFFF44336.toInt())
            lineWidth = 2f
            circleRadius = 4f
            setDrawFilled(true)
            fillColor = 0x1AF44336.toInt()
            valueTextSize = 10f
            setDrawValues(false)
        }
        
        val lineData = LineData(dataSet)
        binding.heartRateChart.data = lineData
        binding.heartRateChart.invalidate()
        
        Log.d("HeartRateChart", "심박수 차트 업데이트 완료: ${heartRateData.size}개 데이터")
    }

    private fun generateHeartRateData(): List<Int> {
        // 20개의 임의 심박수 데이터 생성 (60-100 bpm 범위)
        return (1..20).map { 
            Random.nextInt(60, 101)
        }
    }

    private fun performModelPrediction() {
        Log.i("HomeFragment", "🎯 [MODEL TEST] 모델 테스트 버튼 클릭됨")
        
        // 로딩 상태 표시
        binding.testModelButton.text = "🔄 예측 중..."
        binding.testModelButton.isEnabled = false
        
        // 에러 메시지 숨기기
        binding.errorText.visibility = View.GONE
        
        Log.i("HomeFragment", "🔄 [MODEL TEST] UI 상태 변경: 로딩 중...")
        
        lifecycleScope.launch {
            try {
                Log.i("HomeFragment", "🚀 [MODEL TEST] 예측 프로세스 시작")
                
                // 임의의 사용자 데이터 생성 (실제 앱에서는 센서나 사용자 입력에서 가져옴)
                val heartRate = Random.nextInt(60, 101)
                val hrvSdnn = Random.nextDouble(20.0, 50.0)
                val bmi = Random.nextDouble(18.5, 30.0)
                val meanSa02 = Random.nextDouble(95.0, 100.0)
                val gender = if (Random.nextBoolean()) "male" else "female"
                val age = Random.nextInt(20, 60)
                
                Log.i("HomeFragment", "📊 [MODEL TEST] 생성된 사용자 데이터:")
                Log.i("HomeFragment", "   💓 심박수: $heartRate bpm")
                Log.i("HomeFragment", "   📈 HRV SDNN: ${String.format("%.2f", hrvSdnn)}")
                Log.i("HomeFragment", "   ⚖️ BMI: ${String.format("%.2f", bmi)}")
                Log.i("HomeFragment", "   🫁 산소포화도: ${String.format("%.2f", meanSa02)}%")
                Log.i("HomeFragment", "   👤 성별: $gender")
                Log.i("HomeFragment", "   🎂 나이: ${age}세")
                
                Log.i("HomeFragment", "🌐 [MODEL TEST] ModelService.predictTemperature() 호출")
                
                // 모델 예측 수행
                val result = modelService.predictTemperature(
                    heartRate = heartRate,
                    hrvSdnn = hrvSdnn,
                    bmi = bmi,
                    meanSa02 = meanSa02,
                    userGender = gender,
                    age = age
                )
                
                Log.i("HomeFragment", "📥 [MODEL TEST] ModelService 응답 수신")
                
                when (result) {
                    is PredictionResult.Success -> {
                        Log.i("HomeFragment", "🎉 [MODEL TEST] 예측 성공!")
                        Log.i("HomeFragment", "🌡️ [MODEL TEST] 예측된 온도: ${result.temperature}°C")
                        Log.i("HomeFragment", "🏷️ [MODEL TEST] 온도 카테고리: ${result.category}")
                        
                        // UI 업데이트
                        binding.temperatureValue.text = "${String.format("%.1f", result.temperature)}°C"
                        updateTemperatureStatus(result.temperature)
                        
                        Log.i("HomeFragment", "🎨 [MODEL TEST] UI 업데이트 완료")
                        
                        // 심박수 차트 업데이트
                        updateHeartRateChart()
                        
                        Log.i("HomeFragment", "📊 [MODEL TEST] 심박수 차트 업데이트 완료")
                        
                        binding.errorText.visibility = View.GONE
                        Log.i("HomeFragment", "✅ [MODEL TEST] 모든 업데이트 완료")
                    }
                    is PredictionResult.Error -> {
                        Log.e("HomeFragment", "❌ [MODEL TEST] 예측 실패")
                        Log.e("HomeFragment", "❌ [MODEL TEST] 에러 메시지: ${result.message}")
                        
                        binding.errorText.text = "예측 실패: ${result.message}"
                        binding.errorText.visibility = View.VISIBLE
                        
                        Log.e("HomeFragment", "⚠️ [MODEL TEST] 에러 메시지 UI에 표시됨")
                    }
                }
                
            } catch (e: Exception) {
                Log.e("HomeFragment", "💥 [MODEL TEST] 예측 중 예외 발생", e)
                Log.e("HomeFragment", "💥 [MODEL TEST] 예외 타입: ${e.javaClass.simpleName}")
                Log.e("HomeFragment", "💥 [MODEL TEST] 예외 메시지: ${e.message}")
                
                binding.errorText.text = "오류 발생: ${e.message}"
                binding.errorText.visibility = View.VISIBLE
                
                Log.e("HomeFragment", "⚠️ [MODEL TEST] 예외 에러 메시지 UI에 표시됨")
            } finally {
                // 버튼 상태 복원
                binding.testModelButton.text = "🧪 모델 테스트 실행"
                binding.testModelButton.isEnabled = true
                
                Log.i("HomeFragment", "🔄 [MODEL TEST] UI 상태 복원: 버튼 활성화")
                Log.i("HomeFragment", "🏁 [MODEL TEST] 모델 테스트 프로세스 완료")
            }
        }
    }

    private fun updateTemperatureStatus(temperature: Float) {
        when {
            temperature < 34.6 -> {
                binding.temperatureStatus.text = "추움"
                binding.temperatureStatus.setTextColor(0xFF2196F3.toInt())
            }
            temperature > 35.6 -> {
                binding.temperatureStatus.text = "더움"
                binding.temperatureStatus.setTextColor(0xFFFF5722.toInt())
            }
            else -> {
                binding.temperatureStatus.text = "쾌적함"
                binding.temperatureStatus.setTextColor(0xFF4CAF50.toInt())
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
