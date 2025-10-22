package com.aiservice.poseul.ui.iot

import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.aiservice.poseul.databinding.FragmentIotBinding

class IotFragment : Fragment() {

    private var _binding: FragmentIotBinding? = null
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        Log.d("IotFragment", "onCreateView 시작")
        _binding = FragmentIotBinding.inflate(inflater, container, false)
        Log.d("IotFragment", "onCreateView 완료")
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        Log.d("IotFragment", "onViewCreated 시작")
        setupUI()
        Log.d("IotFragment", "onViewCreated 완료")
    }

    private fun setupUI() {
        // 기본 IOT 기기 정보 표시
        binding.airConditionerStatus.text = "온라인"
        binding.airConditionerCurrentTemp.text = "24°C"
        binding.airConditionerTargetTemp.text = "26°C"
        binding.airConditionerStatus.setTextColor(0xFF4CAF50.toInt())
        
        binding.otherDevicesList.text = "온도 센서: 온라인\n가습기: 오프라인"
        
        // 카드 표시
        binding.airConditionerCard.visibility = View.VISIBLE
        binding.otherDevicesCard.visibility = View.VISIBLE
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
