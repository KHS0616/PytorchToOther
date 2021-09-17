# PytorchToOther
파이토치 모델을 다른 형식으로 변환하는 코드 입니다.  
추가로 변환 후 추론을 진행 할 수 있는 코드도 제공합니다.

## 변환 가능 한 파일 형식
+ ONNX
+ TensorRT(trt)
+ TensorFlow(pb)
+ TensorLite(tflite)

## 사용 방법
1. 컨버팅 정보 설정  
기본적으로 모든 설정은 option.yml 파일에서 설정합니다.  
우선 컨버팅 관련 정보를 수정 합니다.  
origin 옵션 값은 현재 파일 형태, target 옵션 값은 변환할 파일 형태 입니다.  
origin, target 지원하는 형식은 아래 주석의 괄호 내용과 같습니다.  
사용자가 정의한 모델을 사용하는 경우 cusrom_mode 값을 True 값으로 설정하세요

2. 모델 정보 입력  
모델을 다른 형태로 Convert 하기 위해서는 모델 전체 코드를 입력해야 합니다.  
Maker 내부 models.py 파일에 모델에 대한 정보를 입력하고 main.py 코드에 모델 객체와 텐서를 저장시킵니다.  

```
########## 컨버팅 관련 정보 ##########
# origin - 현재 모델 형식 (pytorch, pb, onnx)
# target - 바꿀 모델 형식 (pb, onnx, tensorrt, tensorlite)
# custom_mode - 사용자 지정 모델 사용 여부
origin: onnx
target: tensorrt
custom_mode: False
```

2. 모델 경로 설정  
모델의 경로를 설정해야 합니다.  
기본적으로 제공하는 루트 폴더는 아래와 같으며  
해당 폴더 아래에 모델을 추가하시거나 코드의 결과로 저장될 것입니다.  
```
########## 루트 경로 설정 ##########
root:
  pytorch: ./Torch
  onnx: ./Onnx
  tensorflow: ./TensorFlow
  tensorrt: ./TensorRT
  tensorlite: ./TensorLite
```

3. 각 형태의 설정과 파이토치 모델 설정  
변환되는 형태의 설정을 하고 파이토치 모델의 설정을 해줍니다.  
현재 파이토치 모델의 설정은 내장된 모델만 지원하고 있습니다.  
변환되는 각각의 형태에 관련된 내용은 [변환 옵션](#옵션)을 참조해주세요  

4. 사용자 정의 모델 사용(추가 설정)
옵션 설정을 마쳤으면 실행 하기전에 확인해야 할 부분이 있습니다.  
코드에서 기본으로 제공하는 모델 이외에 모델을 사용하고 싶으신 분은 main 파일 아래 영역에 weights 값을 불러온 모델 모듈을 opt["custom_model"]에 대입하고 토치 텐서를 opt["custom_input_tensor"]에 대입해 줍니다.
```python
<main.py>
########## 추가 설정 부분 ##########
from models import TEST
test = TEST()
if opt["custom_mode"]:
    opt["custom_model"] = test
    opt["custom_input_tensor"] = torch.rand(1)
###################################
```

5. 실행  
설정이 완료되었으면 코드를 실행합니다.  
```
python3 main.py
```

## 옵션
+ 파이토치(Pytorch)  
파이토치 모델 이름과 파일 경로를 설정합니다.  
```
pytorch:
  name: BSRGAN
  path: BSRGAN.pth
```

+ ONNX  
ONNX 파일 경로와 입력 텐서의 형태, ONNX Convert 버전을 설정합니다.
```
onnx:
  path: bsrgan.onnx
  input_batch: 1
  input_channels: 3
  input_width: 960
  input_height: 540
  version: 11
```

+ TensorRT  
TensorRT 파일 경로를 설정하고 int8 양자화 적용 유무를 설정합니다.  
양자화 진행시 사용할 메모리 공간을 지정하고 Calibration 과정에서 사용되는 이미지 경로를 지정해 줍니다.
```
tensorrt:
  path: bsrgan2.trt
  int8_apply: True
  capacity_memory: 2
  save: True
  cali_path: ./DIV2K_test_HR
```

+ TensorFlow  
Tensorflow 파일 경로를 설정합니다.
```
tensorflow:
  path: bsrgan_tf
  save: True
```

+ TensorLite  
Tensorlite 파일 경로를 설정합니다.  
양자화 기법 사용여부 결정 및 Cali 데이터 셋 경로 설정  
```
tensorlite:
  path: bsrgan.tflite
  int8_full: True
  cali_path: ./DIV2K_test_HR
  save: True
```