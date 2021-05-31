"""
Tensorflow 모델을 TensorLite 모델로 변환하는 모듈

Writer : KHS0616
Last Update : 2021-05-27
"""
# 경로 및 시스템 관련 모듈
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Tensorflow & TensorLite 관련 모듈
import tensorflow as tf
from Maker.tensorflow_maker import Main as TM

class Main():
    """ 메인 클래스 """
    def __init__(self, opt):
        self.opt = opt

    def making(self):
        """ Tensorlite tflite 를 만드는 일련의 과정 """
        # 파이토치 모델일 경우 onnx 먼저 생성
        if self.opt["origin"] == "pytorch" or self.opt["origin"] == "onnx":
            TM(self.opt).making()

        if self.opt["origin"] == "pytorch" or self.opt["origin"] == "onnx" or self.opt["origin"] == "tensorflow":
            self.maketflite()
        else:
            print("잘못된 파일")
            return

    def maketflite(self):
        """ Tensorflow PB 파일을 TFLite 파일로 변환하는 함수 """
        pb_path = os.path.join(os.path.join(self.opt["root"]["tensorflow"], self.opt["tensorflow"]["path"]))
        converter = tf.lite.TFLiteConverter.from_saved_model(pb_path)

        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                            tf.lite.OpsSet.SELECT_TF_OPS]

        # converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]

        tf_lite_model = converter.convert()
        open(os.path.join(self.opt["root"]["tensorlite"], self.opt["tensorlite"]["path"]), "wb").write(tf_lite_model)