import numpy as np
from scipy.spatial.transform import Rotation as R
from .rtvs import Rtvs
from utils.logger import logger
from ..base_controller import Controller


class RTVSController(Controller):
    def __init__(
        self,
        grasp_time: float,
        post_grasp_dest,
        box_size,
        conveyor_level,
        ee_pos_scale,
        rtvs: Rtvs,
        cam_to_gt_R: R,
        max_speed=0.5,
    ):
        super().__init__(
            grasp_time,
            post_grasp_dest,
            box_size,
            conveyor_level,
            ee_pos_scale,
            max_speed,
        )
        self.rtvs = rtvs
        self.cam_to_gt_R = cam_to_gt_R

        self.ready_to_grasp = False
        self.real_grasp_time = None

    def _get_ee_val(self, rgb_img, depth_img):
        ee_vel_cam, err = self.rtvs.get_vel(rgb_img, depth=depth_img)
        ee_vel_cam = ee_vel_cam[:3]
        ee_vel_gt = self.cam_to_gt_R.apply(ee_vel_cam)
        speed = min(self.max_speed, np.linalg.norm(ee_vel_gt))
        vel = ee_vel_gt * (
            speed / np.linalg.norm(ee_vel_gt) if not np.isclose(speed, 0) else 1
        )
        if err < 0.05:
            self.ready_to_grasp = True

        logger.debug(pred_vel=vel, pred_speed=np.linalg.norm(vel), err=err)
        return vel

    def get_action(self, rgb_img, depth_img, cur_t, ee_pos):
        action = np.zeros(5)
        if cur_t <= self.grasp_time and not self.ready_to_grasp:
            action[4] = -1
            action[:3] = self._get_ee_val(rgb_img, depth_img)
            if cur_t <= 0.6 * self.grasp_time:
                tpos = self._action_vel_to_target_pos(action[:3], ee_pos)
                tpos[2] = max(tpos[2], self.conveyor_level + self.box_size[2] + 0.005)
                action[2] = self._target_pos_to_action_vel(tpos, ee_pos)[2]
        else:
            action[4] = 1
            if self.real_grasp_time is None:
                self.real_grasp_time = cur_t
            if cur_t <= self.real_grasp_time + 0.5:
                action[:3] = [0, 0, 0.5]
            else:
                action[:3] = self.post_grasp_dest - ee_pos
        return action