"""
ONNX 추론 모듈

Writer : KHS0616

"""
# GPU 설정
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ONNX 관련 설정
import onnxruntime

import cv2
import numpy as np
import torch

# 모델 불러오기
ort_session = onnxruntime.InferenceSession("../Onnx/bsrgan.onnx")

# memo = open("memo-onnx.txt", "w")
# 추론 시작
for _ in range(1):
    img = cv2.imread("00001.png").astype(np.float32)
    lr = cv2.resize(img, (960, 540), interpolation=cv2.INTER_CUBIC)
    lr = lr.transpose(2,0,1)
    lr = np.expand_dims(lr/255., axis=0)
    
    ort_inputs = {ort_session.get_inputs()[0].name: lr}
    ort_outs = ort_session.run(None, ort_inputs)
    out = ort_outs[0].squeeze(0)
    out = out.transpose(1,2,0)*255.
    # memo.write(str(time.time() - start) + "\n")
# memo.close()

# 결과 저장
cv2.imwrite("test23.png", out)