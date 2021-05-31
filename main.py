"""
파이토치 pt, pth 모델 파일을 다른 형태로 변환하는 모듈

현재 구현 된 파일 목록
1. pb 파일 (텐서플로우)
2. onnx 파일
3. trt 파일 (Tensor RT)
4. tflite 파일 (Tensorflow Lite)

Writer : KHS0616
Last Update : 2021-05-27
"""
# GPU 설정
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 옵션 파일 yml 사용을 위한 라이브러리
import yaml

# 옵션 값에 따른 실행 모듈을 가져오기 위한 라이브러리
import importlib

########## 메인 실행 구문 ##########
if __name__ == '__main__':
    with open("option.yml", "r") as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    # 설정된 모듈 불러오기
    module_name = "Maker." + opt["target"] + "_maker"
    module = importlib.import_module(module_name)

    # 모델의 메인함수 객체 생성하기
    FC = getattr(module, "Main")
    finc = FC(opt)
    
    # 변환 시작
    finc.making()