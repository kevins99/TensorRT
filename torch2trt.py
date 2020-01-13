import torch
import torchvision.models as models
import onnx


model = models.vgg16(pretrained=True)

#SETTIGN IN EVAL MODE FOR DROPOUT AND BN LAYERS
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)


input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

print(model)


torch.onnx.export(model, dummy_input, "vgg16FINALFINALFINAL.onnx", verbose=True, input_names=input_names, output_names=output_names)


