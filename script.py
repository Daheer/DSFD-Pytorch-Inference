import time
import torch
from torch2trt_dynamic import torch2trt_dynamic

def benchmark(model, trt_model, input_shape, dtype=torch.float32):
  model = model.cuda().eval()
  inp = torch.rand(input_shape).cuda()

  if dtype==torch.float16:
    model = model.half()
    inp = inp.half()

  total_time_torch = 0
  n_repeats = 128
  for i in range(n_repeats):
    start_torch = time.time()
    res_torch = model(inp)
    torch.cuda.synchronize()
    end_torch = time.time()
    run_time = end_torch - start_torch
    total_time_torch += run_time

  print(f"Time taken for Torch model {float(total_time_torch/n_repeats):.4f}")

  total_time_trt = 0
  for i in range(n_repeats):
    start_trt = time.time()
    res_trt = trt_model(inp)
    torch.cuda.synchronize()
    end_trt = time.time()
    run_time = end_trt - start_trt
    total_time_trt += run_time

  print(f"Time taken for TRT model {float(total_time_trt/n_repeats):.4f}")

  print(f"Estimated speedup: {float(total_time_torch/total_time_trt):.2f}x")

  if isinstance(res_torch, tuple):
    for i, item in enumerate(zip(res_torch, res_trt)):
      # thresh = item[0].max() // item[0].min()
      thresh = 5e-1
      assert (torch.allclose(item[0][0], item[1][0].half(), atol=thresh)), "Outputs from Torch and TensorRT models are too different"
  else:
    # thresh = res_torch.max() // res_torch.min()
    thresh = 5e-1
    assert (torch.allclose(res_torch, res_trt.half(), atol=thresh)), "Outputs from Torch and TensorRT models are too different"


def get_trt_model_dynamic(model, dtype = torch.float16, device = 'cuda'):
  model = model.to(device).eval()
  if dtype == torch.float16:
    model = model.half()
  opt_shape_param = [
      [
          [1, 3, 128, 128],   # min
          [1, 3, 256, 256],   # opt
          [1, 3, 512, 512]    # max
      ]
  ]
  model_trt = torch2trt_dynamic(model, [torch.rand(1, 3, 300, 300).cuda().half()], fp16_mode=True, opt_shape_param=opt_shape_param)

  return model_trt

def get_trt_model(model, input_shape = [1, 3, 300, 300], dtype = torch.float16, device = 'cuda'):
  model = model.to(device).eval()
  if dtype == torch.float16:
    model = model.half()
  traced_model = torch.jit.trace(model, torch.rand(input_shape).to(device).half())

  trt_model = torch_tensorrt.compile(traced_model, inputs = [torch_tensorrt.Input(
      input_shape,
      dtype = dtype)],
    enabled_precisions = torch.half,
  )
  return trt_model

from face_detection.dsfd.face_ssd import SSD_TensorRT

from face_detection.dsfd.config import resnet152_model_config

model = SSD_TensorRT(resnet152_model_config)
model.load_state_dict(torch.load('model.pth'))
trt_model = get_trt_model(model)

benchmark(model, trt_model, input_shape=[1, 3, 300, 300], dtype=torch.float16)
