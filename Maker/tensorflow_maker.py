"""
ONNX 모델을 Tensorflow 모델로 변환하는 모듈

Writer : KHS0616
Last Update : 2021-05-27
"""
# 경로 및 시스템 관련 모듈
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ONNX 관련 모듈
from Maker.onnx_maker import Main as OM
import onnx
from onnx_tf.backend import prepare

class Main():
    """ 메인 클래스 """
    def __init__(self, opt):
        self.opt = opt

    def making(self):
        """ Tensorflow PB를 만드는 일련의 과정 """
        # 파이토치 모델일 경우 onnx 먼저 생성
        if self.opt["origin"] == "pytorch":
            OM(self.opt).making()

        if self.opt["origin"] == "pytorch" or self.opt["origin"] == "onnx":
            self.onnx2tfpb()
        else:
            print("잘못된 파일")
            return

    def onnx2tfpb(self):
        """ ONNX를 Tensorflow PB 파일로 변환하는 함수 """
        # ONNX 모델을 불러오기
        onnx_model = onnx.load(os.path.join(self.opt["root"]["onnx"], self.opt["onnx"]["path"]))
        tf_rep = prepare(onnx_model)

        # pb파일 경로 설정
        pb_path = os.path.join(os.path.join(self.opt["root"]["tensorflow"], self.opt["tensorflow"]["path"]))

        # pb 파일로 Converting
        tf_rep.export_graph(pb_path)