import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import cv2
import numpy as np

class HostDeviceMem(object):
    """ CPU & CPU의 메모리를 설정하고 주소를 반환하는 클래스 """
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
 
    def __str__(self):
        """ CPU (host) & GPU (host)의 주소를 문자열로 반환하는 메소드 """
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
 
    def __repr__(self):
        """ CPU (host) & GPU (host)의 주소를 반환하는 메소드 """
        return self.__str__()

class TRTEngine(object):
    def __init__(self, trt_engine_path):
        """ trt engine 셋팅 """
        # trt 로그 초기화
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        # runtime에 로그 추가
        trt_runtime = trt.Runtime(TRT_LOGGER)
        # CPU & GPU의 원활한 통신을 하기 위해 stream 사용
        self.stream = cuda.Stream()
        # .trt 파일 로드 후 trt_runtime에 추가
        self.engine = self.load_engine(trt_runtime, trt_engine_path)
        # Inference 할 context 준비
        self.context = self.create_execution_context()
        # CPU & GPU 연동
        self.inputs, self.outputs, self.bindings = self.bindingProcess()

    def load_engine(self, trt_runtime, engine_path):
        """ 엔진 로드 메소드 """
        # Serialized 된 엔진 데이터 읽기모드
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        # Serialized 된 엔진 데이터를 다시 객체형태로 복원하여 엔진으로 생성
        return trt_runtime.deserialize_cuda_engine(engine_data)

    def create_execution_context(self):
        """ Inference 할 context를 만드는 메소드 """
        return self.engine.create_execution_context()

 
    def do_inference(self, frame):
        """ 추론 메소드 """
        # 호스트 설정
        hosts = [input.host for input in self.inputs]
        
        # 호스트에 lr 이미지 복사
        for host in hosts:
            # lr 이미지를 float16 데이터타입을 가진 1차원으로 변형
            numpy_array = np.asarray(frame).astype(trt.nptype(trt.float32)).ravel()
            np.copyto(host,numpy_array)

        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        return self.outputs[0].host

    def bindingProcess(self):
        """ CPU & GPU 연동 메소드 """
        inputs, outputs, bindings = [], [], []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * 1
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # CPU 메모리 사이즈 설정
            host_mem = cuda.pagelocked_empty(size, dtype)
            # GPU 메모리 사이즈 설정
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings

def inference(trt_path):
    """ 추론 함수 """
    # TensorRT 엔진 생성
    trt_engine = TRTEngine(trt_path)
    
    for _ in range(1):
        img = cv2.imread("00001.png").astype(np.float32)
        lr = cv2.resize(img, (960, 540), interpolation=cv2.INTER_CUBIC)
        lr = np.expand_dims(lr.transpose(2,0,1)/255., axis=0)
        print(lr.shape)
        lr = lr.astype(trt.nptype(trt.float32)).ravel()
        print(lr.shape)
        sr = np.reshape(trt_engine.do_inference(lr), (1, 3, 2160, 3840))
        print(sr.shape)
        sr = sr.squeeze(0).transpose(1, 2, 0)*255.
        print(sr.shape)

        # 결과 저장
        cv2.imwrite("test.png", sr.astype(np.uint8))

if __name__ == '__main__':
    inference("../TensorRT/bsrgan2.trt")