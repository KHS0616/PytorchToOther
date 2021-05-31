"""
텐서플로우 라이트 추론 모듈

Writer : KHS0616
Last Update : 2021-05-25
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import tensorflow as tf
import cv2
import numpy as np

interpreter = tf.compat.v1.lite.Interpreter("test222.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

lr = cv2.imread("00001.png").astype(np.float32)
lr = cv2.resize(lr, (640, 360))
lr = np.expand_dims(lr/255., axis=0)

print(input_details)

interpreter.set_tensor(input_details['index'], lr)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details['index'])[0]

out =output_data.squeeze(0)*255.
cv2.imwrite("test333.png", out)