import tensorrt as trt 
import torch.onnx as onnx
import numpy as np 
import pycuda.autoinit
import time
import cv2 as cv
#import onnx

import pycuda.driver as cuda

model_path = "./vgg16FINALFINALFINAL.onnx"
# onnx.checker.check_model(model_path)
# onnx.helper.printable_graph(model.graph)

input_size = 224

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)       

def build_engine(model_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(flags = 1) as network, \
    trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1<<30
        builder.max_batch_size = 1
        builder.fp16_mode = 1

        
        with open(model_path, 'rb') as f:
            value = parser.parse(f.read())
            print("Parser: ", value)
            # if (value == False):
            #     err = trt.ParserError()
            #     p = err.code()
            #     print(p)  
            
        engine = builder.build_cuda_engine(network)
        print(engine)
        return engine

def alloc_buf(engine):
    h_in_size = trt.volume(engine.get_binding_shape(0))
    h_out_size = trt.volume(engine.get_binding_shape(1))
    h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))
    h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))
    in_cpu = cuda.pagelocked_empty(h_in_size, h_in_dtype)
    out_cpu = cuda.pagelocked_empty(h_out_size, h_out_dtype)
    in_gpu = cuda.mem_alloc(in_cpu.nbytes)
    out_gpu = cuda.mem_alloc(out_cpu.nbytes)
    stream = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream

def inference(engine, context, inputs, out_cpu, in_gpu, out_gpu, stream):
    cuda.memcpy_htod(in_gpu, inputs)
    context.execute(bindings=[int(in_gpu), int(out_gpu)])
    cuda.memcpy_dtoh(out_cpu, out_gpu)
    return out_cpu

if __name__ == "__main__":
    inputs = np.random.random((1, 3, input_size, input_size)).astype(np.float32)
    engine = build_engine(model_path)
    print(engine)
    context = engine.create_execution_context()
    with open("AlexNet.engine", "wb") as f:
        f.write(engine.serialize())
    for _ in range(10):
        t1 = time.time()
        in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
        print(inputs.shape)
        res = inference(engine, context, inputs.reshape(-1), out_cpu, in_gpu, out_gpu, stream)
        print(res)
        print(res.shape)
        print(np.argmax(res))
        print("cost time: ", time.time()-t1)

# if __name__ == "__main__":
#     path = "/home/kevin/CDAC/AlexNet/image.jpeg"
#     img = cv.imread(path)
#     img = cv.resize(img, (224, 224))
#     x = np.array(img, dtype=np.float32)
#     x /= 255.0
#     x -= 0.5
#     x *= 2.0

#     img = x.transpose([2, 0, 1])
#     img = np.ascontiguousarray(img)

#     engine = build_engine(model_path)
#     print(engine)

#     context = engine.create_execution_context()
#     output = np.empty(1000, dtype = np.float32)

#     #ALLOCATE MEMORY
#     d_input = cuda.mem_alloc(1*img.nbytes)
#     d_output = cuda.mem_alloc(1*output.nbytes)

#     bindings = [int(d_input), int(d_output)]
#     stream = cuda.Stream()

#     cuda.memcpy_htod_async(d_input, img, stream)
#     #context.enqueue(1, bindings, stream.handle, None)
#     context.execute(1, bindings)

#     cuda.memcpy_dtoh_async(output, d_output, stream)
#     stream.synchronize()

#     imagenet_file_path = './imagenet_classes.txt'
#     LABELS = open(imagenet_file_path,'r').readlines()
#     print("Prediction: ", LABELS[np.argmax(output)])
