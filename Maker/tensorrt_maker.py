"""
ONNX 모델을 TensorRT 모델로 변환하는 모듈

Writer : KHS0616
Last Update : 2021-05-27
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# TensorRT 관련 모듈 및 라이브러리 가져오기
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from Maker import common
from PIL import Image

from Maker.onnx_maker import Main as OM

import numpy as np
import glob

# TRT INT8 양자화 관련 변수
__all__ = [
    'PythonEntropyCalibrator',
    'ImageBatchStream'
]

class Main():
    """ 메인 클래스 """
    def __init__(self, opt):
        self.opt = opt
        self.TRT_LOGGER = trt.Logger()

    def making(self):
        """ TensorRT를 만드는 일련의 과정 """
        # 파이토치 모델일 경우 onnx 먼저 생성
        if self.opt["origin"] == "pytorch":
            OM(self.opt).making()

        if self.opt["origin"] == "pytorch" or self.opt["origin"] == "onnx":
            self.convertTRT()
        else:
            print("잘못된 파일")
            return

    def create_calibration_dataset(self):
        """ Calibration 데이터셋 생성 함수 """
        return glob.glob(os.path.join(self.opt["tensorrt"]["cali_path"], '*.png'))

    def convertTRT(self):
        """ ONNX 모델을 TRT 파일로 변환 """
        # Calib 파일, 배치 스트림 함수, 양자화 함수 선언
        if self.opt["tensorrt"]["int8_apply"]:
            calibration_files = self.create_calibration_dataset()
            batchstream = ImageBatchStream(1, calibration_files, self.opt["onnx"]["input_width"], self.opt["onnx"]["input_height"], self.opt["onnx"]["input_channels"])
            calib = PythonEntropyCalibrator(["data"], batchstream, 'temp.cache')

        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        # 빌더, 네트워크, 파서를 통한 trt엔진 생성 과정
        with trt.Builder(self.TRT_LOGGER) as builder, builder.create_network((EXPLICIT_BATCH)) as network, trt.OnnxParser(network, self.TRT_LOGGER) as parser:
            """ 빌드 생성, 네트워크 생성, onnx 파서 생성 """
            # ONNX 파일 읽기
            with open(os.path.join(self.opt["root"]["onnx"], self.opt["onnx"]["path"]), 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            print('Completed parsing of ONNX file')

            print('Network inputs:')

            for i in range(network.num_inputs):
                tensor = network.get_input(i)
                print(tensor.name, trt.nptype(tensor.dtype), tensor.shape)

            config = builder.create_builder_config()
            config.max_workspace_size = common.GiB(self.opt["tensorrt"]["capacity_memory"])  # 256MiB

            # 양자화 Calib 함수가 있으면 사용하도록 설정
            if calib:
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = calib
            else:
                builder.fp16_mode = True

            print('Building an engine from file {}; this may take a while...'.format(self.opt["onnx"]["path"]))
            # trt엔진 빌드
            engine = builder.build_engine(network, config)
            print("Completed creating Engine. Writing file to: {}".format(self.opt["tensorrt"]["path"]))

            # 빌드된 trt 엔진을 저장
            with open(os.path.join(self.opt["root"]["tensorrt"], self.opt["tensorrt"]["path"]), "wb") as f:
                f.write(engine.serialize())

class PythonEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """ Int8 양자화를 위해 Calibrate를 진행하는 클래스 """
    def __init__(self, input_layers, stream, cache_file):
        super(PythonEntropyCalibrator, self).__init__()

        # Tensor RT에 지정될 Input 레이어 이름 설정
        self.input_layers = input_layers

        # Calib 이미지 배치 스트림 저장
        self.stream = stream

        # 데이터 GPU 할당
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)

        # 캐시파일 경로 저장
        self.cache_file = cache_file

        # 현재 스트림 리셋
        stream.reset()

    def get_batch_size(self):
        """ 배치사이즈 반환 메소드 """
        return self.stream.batch_size

    def get_batch(self, names):
        try:
            batch = self.stream.next_batch()
            if not batch.size:   
                return None

            cuda.memcpy_htod(self.d_input, batch)
            return [int(self.d_input)]

        except StopIteration:
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        # cache = ctypes.c_char_p(int(ptr))
        with open(self.cache_file, 'wb') as f:
            f.write(cache)

class ImageBatchStream():
    """ 양자화 과정에서 사용되는 이미지 배치 스트림 """
    def __init__(self, batch_size, calibration_files, WIDTH, HEIGHT, CHANNEL):
        # 배치사이즈 결정
        self.batch_size = batch_size
        self.max_batches = (len(calibration_files) // batch_size) + \
                        (1 if (len(calibration_files) % batch_size) \
                            else 0)

        # 파일 목록을 변수에 저장
        self.files = calibration_files

        # 변수 초기화
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.CHANNEL = CHANNEL
        self.calibration_data = np.zeros((batch_size, CHANNEL, HEIGHT, WIDTH), dtype=np.float32)
        self.batch = 0

     
    @staticmethod
    def read_image(path, WIDTH, HEIGHT, CHANNEL):
        img = Image.open(path).convert('RGB').resize((WIDTH,HEIGHT), Image.BICUBIC)
        img = np.array(img, dtype=np.float32, order='C')
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img/255., axis=0)
        return img

    def reset(self):
        self.batch = 0
        
    def next_batch(self):
        if self.batch < self.max_batches:
            imgs = []
            files_for_batch = self.files[self.batch_size * self.batch : \
                                self.batch_size * (self.batch + 1)]
                    
            for f in files_for_batch:
                print("[ImageBatchStream] Processing ", f)
                img = ImageBatchStream.read_image(f, self.WIDTH, self.HEIGHT, self.CHANNEL)
                imgs.append(img)
            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1

            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])