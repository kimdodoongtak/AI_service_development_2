"""
AI 체온 예측 서버
앙상블 모델을 사용하여 체온을 예측하는 Flask API 서버
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import zipfile
import tempfile
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # CORS 허용

# 전역 변수
model = None
model_loaded = False

def load_model():
    """앙상블 모델 로드"""
    global model, model_loaded
    
    try:
        # 모델 파일 경로 (age 포함 모델)
        model_path = '../pycode/ai_thermal_model_with_age.pkl'
        
        if not os.path.exists(model_path):
            logger.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
            return False
        
        # 모델 로드
        model = joblib.load(model_path)
        
        if model is None:
            logger.error("모델을 로드할 수 없습니다.")
            return False
            
        model_loaded = True
        logger.info("앙상블 모델 로드 완료")
        return True
        
    except Exception as e:
        logger.error(f"모델 로드 실패: {str(e)}")
        return False

def predict_temperature(hr_mean, hrv_sdnn, bmi, mean_sa02, gender, age):
    """
    체온 예측 함수 (나이 포함)
    
    Parameters:
    - hr_mean: 평균 심박수
    - hrv_sdnn: 심박변이도 (SDNN)
    - bmi: 체질량지수
    - mean_sa02: 평균 산소포화도
    - gender: 성별 ('M' 또는 'F')
    - age: 나이
    
    Returns:
    - 예측된 체온 (°C)
    """
    if not model_loaded:
        raise ValueError("모델이 로드되지 않았습니다.")
    
    # 파생 피처 계산
    hrv_hr_ratio = hrv_sdnn / hr_mean
    bmi_hr_interaction = bmi * hr_mean
    age_bmi_interaction = age * bmi
    age_hrv_ratio = age / (hrv_sdnn + 1)  # 0으로 나누기 방지
    
    # 데이터 준비
    data = pd.DataFrame({
        'bmi': [bmi],
        'mean_sa02': [mean_sa02], 
        'HRV_SDNN': [hrv_sdnn],
        'hrv_hr_ratio': [hrv_hr_ratio],
        'bmi_hr_interaction': [bmi_hr_interaction],
        'age': [age],
        'age_bmi_interaction': [age_bmi_interaction],
        'age_hrv_ratio': [age_hrv_ratio],
        'gender': [gender]
    })
    
    # 예측
    temp_pred = model.predict(data)[0]
    return float(temp_pred)

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    })

@app.route('/predict', methods=['POST'])
def predict():
    """체온 예측 API"""
    try:
        if not model_loaded:
            return jsonify({
                'error': '모델이 로드되지 않았습니다.'
            }), 500
        
        # 요청 데이터 파싱
        data = request.get_json()
        logger.info(f"📱 앱에서 예측 요청 받음: {data}")
        
        # 필수 파라미터 확인
        required_params = ['hr_mean', 'hrv_sdnn', 'bmi', 'mean_sa02', 'gender', 'age']
        for param in required_params:
            if param not in data:
                return jsonify({
                    'error': f'필수 파라미터가 누락되었습니다: {param}'
                }), 400
        
        # 예측 수행
        predicted_temp = predict_temperature(
            hr_mean=float(data['hr_mean']),
            hrv_sdnn=float(data['hrv_sdnn']),
            bmi=float(data['bmi']),
            mean_sa02=float(data['mean_sa02']),
            gender=str(data['gender']),
            age=int(data['age'])
        )
        
        # 온도 분류
        def classify_temperature(temp, cold_threshold=33.0, hot_threshold=35.0):
            if temp < cold_threshold:
                return "냉기"
            elif temp > hot_threshold:
                return "더위"
            else:
                return "적정"
        
        temperature_category = classify_temperature(predicted_temp)
        
        result = {
            'success': True,
            'predicted_temperature': predicted_temp,
            'temperature_category': temperature_category,
            'input_data': data
        }
        logger.info(f"✅ 예측 완료: {predicted_temp:.2f}°C ({temperature_category})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"예측 실패: {str(e)}")
        return jsonify({
            'error': f'예측 실패: {str(e)}'
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """모델 정보 반환"""
    if not model_loaded:
        return jsonify({
            'error': '모델이 로드되지 않았습니다.'
        }), 500
    
    return jsonify({
        'model_type': '앙상블 모델 (RandomForest + ExtraTrees + GradientBoosting) - 나이 포함',
        'features': ['bmi', 'mean_sa02', 'HRV_SDNN', 'hrv_hr_ratio', 'bmi_hr_interaction', 'age', 'age_bmi_interaction', 'age_hrv_ratio', 'gender'],
        'target': 'TEMP_median (체온)',
        'model_loaded': model_loaded
    })

if __name__ == '__main__':
    # 서버 시작 시 모델 로드
    if load_model():
        logger.info("서버 시작 중...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("모델 로드 실패로 서버를 시작할 수 없습니다.")
