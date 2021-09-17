"""
Tensorflow 모델을 TensorLite 모델로 변환하는 모듈

Writer : KHS0616
Last Update : 2021-09-17
"""
# 경로 및 시스템 관련 모듈
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Tensorflow & TensorLite 관련 모듈
import tensorflow as tf
from Maker.tensorflow_maker import Main as TM

# 이미지 처리 관련 라이브러리
import cv2

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

    def makeCalibrationDataset(self):
        """ Calibration 데이터 셋 생성 메소드 """
        self.cali_datasets = []
        for v in os.listdir(self.opt["tensorlite"]["cali_path"]):
            img = cv2.imread(os.path.join(self.opt["tensorlite"]["cali_path"], v)).astype(np.float32)
            img = cv2.resize(img, (1280,720), cv2.INTER_CUBIC)
            img = img.transpose(2, 0, 1) / 255.
            img = np.expand_dims(img[0], 0)
            # img = tf.io.read_file(os.path.join(self.opt["tensorlite"]["cali_path"], v))
            # img = tf.image.decode_jpeg(img)
            # img = tf.image.random_crop(img, [128, 128, 3], seed=None, name=None)
            # img = tf.cast(img, tf.float32) / 255.0
            self.cali_datasets.append(img)

    def representative_data_gen(self):
        """ TensorLite 버전 Calibration 메소드 """ 
        self.makeCalibrationDataset()
        for input_value in tf.data.Dataset.from_tensor_slices(self.cali_datasets).batch(1).take(100):
            # Model has only one input so each data point has one element.
            yield [input_value]

    def maketflite(self):
        """ Tensorflow PB 파일을 TFLite 파일로 변환하는 함수 """
        pb_path = os.path.join(os.path.join(self.opt["root"]["tensorflow"], self.opt["tensorflow"]["path"]))
        converter = tf.lite.TFLiteConverter.from_saved_model(pb_path)

        # 에러 방지
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                            tf.lite.OpsSet.SELECT_TF_OPS]

        if self.opt["tensorlite"]["int8_full"]:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # converter.target_spec.supported_types = [tf.float16]
            converter.representative_dataset = self.representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

        tf_lite_model = converter.convert()
        open(os.path.join(self.opt["root"]["tensorlite"], self.opt["tensorlite"]["path"]), "wb").write(tf_lite_model)