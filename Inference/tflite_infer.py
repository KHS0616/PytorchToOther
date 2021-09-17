"""
텐서플로우 라이트 추론 모듈

Writer : KHS0616
Last Update : 2021-05-25
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import tensorflow as tf
import cv2
import numpy as np

interpreter = tf.compat.v1.lite.Interpreter("../TensorLite/LDSR_x2_16x128.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

lr = cv2.imread("00001.png").astype(np.float32)
lr = cv2.resize(lr, (50, 50))
lr = lr.transpose(2,0,1)
lr = np.expand_dims(lr/255., axis=0)

print(input_details)

interpreter.set_tensor(input_details['index'], lr)

print("start")
interpreter.invoke()
print("end")

output_data = interpreter.get_tensor(output_details['index'])[0]
output_data = output_data.transpose(1,2,0)*255.
print(output_data.shape)
#out =output_data.squeeze(0)*255.
cv2.imwrite("testldsr.png", output_data)