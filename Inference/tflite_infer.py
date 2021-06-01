"""
텐서플로우 라이트 추론 모듈

Writer : KHS0616
Last Update : 2021-05-25
"""
# 시스템 환경 설정
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# 텐서플로우 가져오기
import tensorflow as tf

# 이미지 처리 관련 모듈 가져고이
import cv2
import numpy as np

# 텐스플로우 라이트 인터프리터 열기
interpreter = tf.compat.v1.lite.Interpreter("test222.tflite")
interpreter.allocate_tensors()

# 등록된 입력, 출력 텐서 정보를 저장하기
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# 이미지 열기
lr = cv2.imread("00001.png").astype(np.float32)

# 이미지 전처리
lr = cv2.resize(lr, (640, 360))
lr = np.expand_dims(lr/255., axis=0)

print(input_details)

# 이미지를 인터프리터에 등록
interpreter.set_tensor(input_details['index'], lr)

# 추론
interpreter.invoke()

# 출력 결과 저장
output_data = interpreter.get_tensor(output_details['index'])[0]

# 후처리
out =output_data.squeeze(0)*255.

# 이미지 저장
cv2.imwrite("test333.png", out)