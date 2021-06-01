"""
Pytorch 모델을 ONNX 모델로 변환하는 모듈

Writer : KHS0616
Last Update : 2021-05-27
"""

# 파이토치 모델 관련 모듈
from Maker.models import BSRGAN

# 파일 및 시스템 관련 모듈
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 파이토치 관련 모듈
import torch

class Main():
    """ 메인 클래스 """
    def __init__(self, opt):
        self.opt = opt

    def making(self):
        """ ONNX를 만드는 일련의 과정 """
        if self.opt["custom_mode"]:
            self.customMaking(self.opt["custom_model"], self.opt["custom_input_tensor"])            
        else:
            self.setTorchModel()
            self.setInputSize()
            self.convertONNX()            

    def customMaking(self, model, input_tensor):
        """ 사용자 정의 모델을 사용하는 경우 """
        self.setTorchModel(model)
        self.setInputSize(input_tensor)
        self.convertONNX()

    def setTorchModel(self, model=None):
        """ 파이토치 모델 등록 """
        # 현재 모델은 BSRGAN만 적용 가능 추후 추가 예정
        # 모델을 받아오면 그대로 사용하고 없으면 새로 지정하고 불러오기
        if model==None:
            if self.opt["pytorch"]["name"] == "BSRGAN":
                self.model = BSRGAN(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=self.opt["BSRGAN"]["scale_factor"])
                self.model.load_state_dict(torch.load(os.path.join(self.opt["root"]["pytorch"], self.opt["pytorch"]["path"])), strict=True)
        else:
            self.model = model

    def setInputSize(self, input_tensor=None):
        """ 입력 이미지 크기 및 형태 설정 """
        # 입력 이미지 정보 저장
        self.input_batch = self.opt["onnx"]["input_batch"]
        self.width = self.opt["onnx"]["input_width"]
        self.height = self.opt["onnx"]["input_height"]
        self.channels = self.opt["onnx"]["input_channels"]

        # 입력 토치텐서 생성 전달받은 텐서가 있으면 그대로 사용
        # 여기서 모델 ONNX 입력 형식을 지정합니다.
        if input_tensor == None:
            if self.opt["pytorch"]["name"] == "BSRGAN":
                self.input_tensor = torch.randn(self.input_batch, self.channels, self.height, self.width, dtype=torch.float32)
        else:
            self.input_tensor = input_tensor

    def convertONNX(self):
        """ 파이토치 모델을 ONNX 파일로 변환 """
        torch.onnx.export(self.model,
                    self.input_tensor,
                    os.path.join(self.opt["root"]["onnx"], self.opt["onnx"]["path"]),
                    export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                    opset_version=int(self.opt["onnx"]["version"]),          # 모델을 변환할 때 사용할 ONNX 버전
                    do_constant_folding=True,  # 최적하시 상수폴딩을 사용할지의 여부
                    input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                    output_names = ['output'], # 모델의 출력값을 가리키는 이름
                    )