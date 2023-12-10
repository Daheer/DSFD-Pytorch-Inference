import face_detection
import torch
from face_detection.dsfd.face_ssd import SSD
from face_detection.dsfd.config import resnet152_model_config
model = SSD(resnet152_model_config)
model.load_state_dict(torch.load('model.pth'))
model.eval()
model = model.to('cuda')
torch.onnx.export(model, (torch.rand(1, 3, 300, 300).to('cuda'), 0.5, 0.3), 'model.onnx', verbose = True, opset_version = 11)
