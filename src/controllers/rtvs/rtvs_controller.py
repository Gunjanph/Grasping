import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.logger import logger

from ..base_controller import Controller
from . import Rtvs


class RTVSController(Controller):
    def __init__(
        self,
        rtvs: Rtvs,
        cam_to_gt_R: R,
        max_speed=0.5,
    ):
        self.rtvs = rtvs
        self.max_speed = max_speed
        self.cam_to_gt_R = cam_to_gt_R
        self.ready_to_grasp = False


    def _get_ee_val(self, rgb_img, depth_img, prev_rgb_img):
        ee_vel_cam, err, mse = self.rtvs.get_vel(
            rgb_img, depth=depth_img, pre_img_src=prev_rgb_img
        )
        ee_vel_cam = ee_vel_cam[:3]
        ee_vel_gt = self.cam_to_gt_R.apply(ee_vel_cam)
        speed = min(self.max_speed, np.linalg.norm(ee_vel_gt))
        vel = ee_vel_gt * (
            speed / np.linalg.norm(ee_vel_gt) if not np.isclose(speed, 0) else 1
        )
        if err > 0.9:
            self.ready_to_grasp = True

        logger.debug(
            "controller (gt frame):",
            pred_vel=vel,
            pred_speed=np.linalg.norm(vel),
            photo_err=err,
            mse=mse
        )
        return vel, mse

    def get_action(self, observations: dict):
        rgb_img = observations["rgb_img"]
        depth_img = observations.get("depth_img", None)
        prev_rgb_img = observations.get("prev_rgb_img", None)
        action = np.zeros(5)
        mse = 0
        if not self.ready_to_grasp:
            action[4] = -1
        else:
            action[4] = 1
        action[:3], mse = self._get_ee_val(rgb_img, depth_img, prev_rgb_img)
        return action, self.rtvs.get_iou(rgb_img), mse
