import torch
import numpy as np
import typing
import time
from collections import OrderedDict
from .face_ssd import SSD, SSD_TensorRT
from .config import resnet152_model_config
from .. import torch_utils
from .utils import get_trt_model
from torch.hub import load_state_dict_from_url
from ..base import Detector
from ..build import DETECTOR_REGISTRY

model_url = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/61be4ec7-8c11-4a4a-a9f4-827144e4ab4f0c2764c1-80a0-4083-bbfa-68419f889b80e4692358-979b-458e-97da-c1a1660b3314"

def benchmark(model, trt_model, input_shape, dtype=torch.float16):
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
      thresh = 1e-3
      assert (torch.allclose(item[0][0], item[1][0], atol=thresh)), "Outputs from Torch and TensorRT models are too different"
  else:
    # thresh = res_torch.max() // res_torch.min()
    thresh = 1e-3
    assert (torch.allclose(res_torch, res_trt, atol=thresh)), "Outputs from Torch and TensorRT models are too different"


@DETECTOR_REGISTRY.register_module
class DSFDDetector(Detector):

    def __init__(
            self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state_dict = load_state_dict_from_url(
            model_url,
            map_location=self.device,
            progress=True)
        self.net = SSD(resnet152_model_config)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.net = self.net.to(self.device)
       

    @torch.no_grad()
    def _detect(self, x: torch.Tensor,) -> typing.List[np.ndarray]:
        """Batched detect
        Args:
            image (np.ndarray): shape [N, H, W, 3]
        Returns:
            boxes: list of length N with shape [num_boxes, 5] per element
        """
        # Expects BGR
        x = x[:, [2, 1, 0], :, :]
        with torch.cuda.amp.autocast(enabled=self.fp16_inference):
            boxes = self.net(
                x
            )
        return boxes

@DETECTOR_REGISTRY.register_module
class DSFDDetectorTensorRT(Detector):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        state_dict = torch.load('model.pth')
        self.ssd = SSD_TensorRT(resnet152_model_config)
        torch_model = self.ssd.feature_enhancer # Remove later
        torch_model.load_state_dict(state_dict) # Remove later
        self.ssd.feature_enhancer.load_state_dict(state_dict)
        
        loc_state_dict = self.ssd.loc.state_dict()
        loc_state_dict = OrderedDict(("loc." + key, value) for key, value in loc_state_dict.items())
        pretrained_loc_state_dict = {key[4:]: value for key, value in state_dict.items() if key in loc_state_dict.keys()}
        self.ssd.loc.load_state_dict(pretrained_loc_state_dict)
        
        conf_state_dict = self.ssd.conf.state_dict()
        conf_state_dict = OrderedDict(("conf." + key, value) for key, value in conf_state_dict.items())
        pretrained_conf_state_dict = {key[5:]: value for key,value in state_dict.items() if key in conf_state_dict.keys()}
        self.ssd.conf.load_state_dict(pretrained_conf_state_dict)

        self.ssd.feature_enhancer = get_trt_model(self.ssd.feature_enhancer, input_shape=[1, 3, 640, 640], fp16=False)
        self.ssd.feature_enhancer.eval()
        self.ssd.conf.eval()
        self.ssd.loc.eval()
        self.ssd.eval()
        self.ssd = self.ssd.to(self.device)
        benchmark(torch_model, self.ssd.feature_enhancer, input_shape=[1, 3, 640, 640], dtype=torch.float32)
    
    @torch.no_grad()
    def _detect(self, x: torch.Tensor,) -> typing.List[np.ndarray]:
        x = x[:, [2, 1, 0], :, :]
        
        with torch.cuda.amp.autocast(enabled=self.fp16_inference):
            boxes = self.ssd(
                x
            )

        #if self.fp16_inference:
        #  boxes = self.ssd(x.half())
          
        return boxes
