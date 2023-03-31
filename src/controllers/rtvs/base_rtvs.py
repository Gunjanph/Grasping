import os

import cv2
import numpy as np
import torch
from PIL import Image

from .calculate_flow import FlowNet2Utils

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# np.random.seed(0)
# warnings.filterwarnings("ignore")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
# torch.manual_seed(0)
# torch.autograd.set_detect_anomaly(True)


class BaseRtvs:
    def __init__(
        self,
        img_goal: np.ndarray,
        cam_k: np.ndarray,
        ct=1,
        horizon=20,
        LR=0.005,
        iterations=10,
    ):
        """
        img_goal: RGB array for final pose
        ct = image downsampling parameter (high ct => faster but less accurate)
        LR = learning rate of NN
        iterations = iterations to train NN (high value => slower but more accurate)
        horizon = MPC horizon
        """
        if isinstance(img_goal, str):
            img_goal = np.asarray(Image.open(img_goal))
        self.img_goal = img_goal
        self.horizon = horizon
        self.iterations = iterations
        self.cam_k = cam_k
        self.ct = ct
        self.flow_utils = FlowNet2Utils()
        self.optimiser = torch.optim.Adam(
            self.vs_lstm.parameters(), lr=LR, betas=(0.93, 0.999)
        )
        self.loss_fn = torch.nn.MSELoss(size_average=False)

    @staticmethod
    def detect_mask(rgb_img, pixrange=((0, 100, 100), (10, 255, 255))):
        hsv_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        # segment red colour from image
        mask = cv2.inRange(hsv_image, *pixrange)
        mask[mask != 0] = 1
        return mask.reshape((rgb_img.shape[0], rgb_img.shape[1], 1))

    def get_iou(self, img_src):
        mask_src = self.detect_mask(img_src, ((50, 100, 100), (70, 255, 255)))
        mask_goal = self.detect_mask(self.img_goal, ((50, 100, 100), (70, 255, 255)))
        intersection = np.logical_and(mask_src, mask_goal)
        union = np.logical_or(mask_src, mask_goal)
        iou_score = np.sum(intersection) / (np.sum(union) + 0.001)
        return iou_score
