"""
Pytorch 모델을 ONNX 모델로 변환하는 모듈

Writer : KHS0616
Last Update : 2021-05-27
"""

# 파이토치 모델 관련 모듈
from Maker.models import LDSR, BSRGAN, HSDSR, HSDSR_DENSE

# 파일 및 시스템 관련 모듈
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 파이토치 관련 모듈
import torch

class Main():
    """ 메인 클래스 """
    def __init__(self, opt):
        self.opt = opt

    def making(self):
        """ ONNX를 만드는 일련의 과정 """
        self.setTorchModel()
        self.setInputSize()
        self.convertONNX()

    def setTorchModel(self, model=None):
        """ 파이토치 모델 등록 """
        # 현재 모델은 LDSR만 적용 가능 추후 추가 예정
        if model==None:
            if self.opt["pytorch"]["name"] == "LDSR":
                self.model = LDSR(self.opt["LDSR"]["scale_factor"])

                # 모델 파라미터 불러오기
                state_dict = self.model.state_dict()
                for n, p in torch.load(os.path.join(self.opt["root"]["pytorch"], self.opt["pytorch"]["path"]), map_location=lambda storage, loc: storage).items():
                    if n in state_dict.keys():
                        state_dict[n].copy_(p)
                    else:
                        raise KeyError(n)

            elif self.opt["pytorch"]["name"] == "HSDSR":
                self.model = HSDSR(self.opt["HSDSR"]["scale_factor"])

                # 모델 파라미터 불러오기
                state_dict = self.model.state_dict()
                for n, p in torch.load(os.path.join(self.opt["root"]["pytorch"], self.opt["pytorch"]["path"]), map_location=lambda storage, loc: storage)["model_state_dict"].items():
                    if n in state_dict.keys():
                        state_dict[n].copy_(p)
                    else:
                        raise KeyError(n)

            elif self.opt["pytorch"]["name"] == "HSDSR_DENSE":
                self.model = HSDSR_DENSE(self.opt["HSDSR_DENSE"]["scale_factor"])

                # 모델 파라미터 불러오기
                state_dict = self.model.state_dict()
                for n, p in torch.load(os.path.join(self.opt["root"]["pytorch"], self.opt["pytorch"]["path"]), map_location=lambda storage, loc: storage).items():
                    if n in state_dict.keys():
                        state_dict[n].copy_(p)
                    else:
                        raise KeyError(n)

            elif self.opt["pytorch"]["name"] == "BSRGAN":
                self.model = BSRGAN(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=self.opt["BSRGAN"]["scale_factor"])
                self.model.load_state_dict(torch.load(os.path.join(self.opt["root"]["pytorch"], self.opt["pytorch"]["path"])), strict=True)

    def setInputSize(self, input_tensor=None):
        """ 입력 이미지 크기 및 형태 설정 """
        # 입력 이미지 정보 저장
        self.input_batch = self.opt["onnx"]["input_batch"]
        self.width = self.opt["onnx"]["input_width"]
        self.height = self.opt["onnx"]["input_height"]
        self.channels = self.opt["onnx"]["input_channels"]

        # 입력 토치텐서 생성 전달받은 텐서가 있으면 그대로 사용
        if input_tensor == None:        
            if self.opt["pytorch"]["name"] == "LDSR" or self.opt["pytorch"]["name"] == "HSDSR" or self.opt["pytorch"]["name"] == "HSDSR_DENSE":
                self.input_tensor = torch.randn(self.input_batch, self.height, self.width, self.channels, dtype=torch.float32)
            elif self.opt["pytorch"]["name"] == "BSRGAN":
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