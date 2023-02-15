import numpy as np
from .ibvs_helper import IBVSHelper
from ..base_controller import Controller
from utils.logger import logger
from scipy.spatial.transform import Rotation as R


class IBVSController(Controller):
    def __init__(
        self,
        grasp_time: float,
        post_grasp_dest,
        box_size,
        conveyor_level,
        ee_pos_scale,
        ibvs_helper: IBVSHelper,
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
        ) # defining the parameters
        self.ibvs_helper = ibvs_helper
        self.cam_to_gt_R = cam_to_gt_R

        self.ready_to_grasp = False
        self.real_grasp_time = None

    def _get_ee_val(self, rgb_img, depth_img):  # getting ee velocity
        ee_vel_cam, err = self.ibvs_helper.get_velocity(rgb_img, depth_img) # get ee velocity wrt camera using current img and depth to track the feature, i.e corners
        ee_vel_gt = self.cam_to_gt_R.apply(ee_vel_cam) # vel of ee wrt ground
        speed = min(self.max_speed, np.linalg.norm(ee_vel_gt)) # speed shouldn't increase max speed
        vel = ee_vel_gt * (
            speed / np.linalg.norm(ee_vel_gt) if not np.isclose(speed, 0) else 1
        ) # velocity = ee wrt ground * speed/ee wrt ground
        if err < 0.05: # if error less than 0.5 then grasp
            self.ready_to_grasp = True

        logger.debug(pred_vel=vel, pred_speed=np.linalg.norm(vel), err=err)
        return vel

    def get_action(self, rgb_img, depth_img, cur_t, ee_pos):
        action = np.zeros(5) 
        if cur_t <= self.grasp_time and not self.ready_to_grasp: # if currently can't grasp then
            action[4] = -1
            action[:3] = self._get_ee_val(rgb_img, depth_img) # get ee velocity
            if cur_t <= 0.6 * self.grasp_time: #
                tpos = self._action_vel_to_target_pos(action[:3], ee_pos) # move to next position, given curr pos and vel
                tpos[2] = max(tpos[2], self.conveyor_level + self.box_size[2] + 0.005) # max depth
                action[2] = self._target_pos_to_action_vel(tpos, ee_pos)[2] # find next target pos
        else:
            action[4] = 1
            if self.real_grasp_time is None: # if time in not mentioned then current time
                self.real_grasp_time = cur_t
            if cur_t <= self.real_grasp_time + 0.5: # just to make sure that it doesn't miss in case.
                action[:3] = [0, 0, 0.5]
            else: # else grasp
                action[:3] = self.post_grasp_dest - ee_pos
        return action
