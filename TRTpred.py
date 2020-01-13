import tensorrt as trt
import torch
import torchvision
import numpy as np 
import pycuda.driver as cuda
import pycuda.autoinit 
from PIL import Image
import cv2 as cv
import os

print(os.getcwd())

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open("AlexNet.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    print("Engine loaded")

##mport glob

path = "/home/kevin/CDAC/AlexNet/longhornedbeetle.jpg" 
#pics = []
#for path in glob.glob("./*.jpg"):
img = cv.imread(path)
img = cv.resize(img, (224, 224))
x = np.array(img, dtype=np.float32)
x /= 255.0
x -= 0.5
x *= 2.0

img = x.transpose([2, 0, 1])
img = np.ascontiguousarray(img)

#img = np.array(pics)

context = engine.create_execution_context()
output = np.empty(1000, dtype = np.float32)

#ALLOCATE MEMORY
d_input = cuda.mem_alloc(1*img.nbytes)
d_output = cuda.mem_alloc(1*output.nbytes)

bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype = np.float32)
h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
# Allocate device memory for inputs and outputs.
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
# Create a stream in which to copy inputs/outputs and run inference.
stream = cuda.Stream()

with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, img.reshape(-1), stream)
        # Run inference.
        context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output. 
        print(h_output.shape)
        print(np.argmax(h_output))
        
        #print(h_output.shape)





# cuda.memcpy_htod_async(d_input, img, stream)
# #context.enqueue(1, bindings, stream.handle, None)
# context.execute(1, bindings)

# cuda.memcpy_dtoh_async(output, d_output, stream)
# stream.synchronize()

# imagenet_file_path = './imagenet_classes.txt'
# LABELS = open(imagenet_file_path,'r').readlines()
# print("Prediction: ", LABELS[np.argmax(output)])
