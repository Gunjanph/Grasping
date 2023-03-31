import numpy as np
from utils.logger import logger
from .base_controller import Controller

class GTController(Controller):
    def __init__(
        self,
        grasp_time,
        post_grasp_dest,
        box_size,
        conveyor_level,
        ee_pos_scale,
        max_speed=2,
    ):
        super().__init__(
            grasp_time,
            post_grasp_dest,
            box_size,
            conveyor_level,
            ee_pos_scale,
            max_speed,
        )

    def _predict_grasp_pos(self, obj_pos, obj_vel, cur_t):
        return obj_pos + obj_vel * (self.grasp_time - cur_t)

    def _get_vel_pre_grasp(self, ee_pos, obj_pos, obj_vel, cur_t):
        grasp_pos = self._predict_grasp_pos(obj_pos, obj_vel, cur_t)
        cur_target = grasp_pos.copy()
        # to allow box to penetrate further
        # arm config 1
        cur_target[2] += -self.box_size[2] * (1 / 3)
        # arm config 2
        # cur_target[1] += 0.06
        if cur_t <= 0.8 * self.grasp_time:
            cur_target[2] += 0.0
        dirn = cur_target - ee_pos
        speed = min(self.max_speed, 10 * np.linalg.norm(dirn))
        vel = speed * (dirn / np.linalg.norm(dirn))

        logger.debug(grasp_pos=grasp_pos, target_pos=cur_target)
        logger.debug(pred_vel=vel, pred_speed=round(np.linalg.norm(vel), 3))
        return vel

    def get_action(self, observations):
        ee_pos = observations["ee_pos"]
        obj_pos = observations["obj_pos"]
        obj_vel = observations["obj_vel"]
        cur_t = observations["cur_t"]
        action = np.zeros(5)
        if cur_t <= self.grasp_time:
            action[4] = -1
            action[:3] = self._get_vel_pre_grasp(ee_pos, obj_pos, obj_vel, cur_t)

        else:
            self.ready_to_grasp = True
            if self.real_grasp_time is None:
                self.real_grasp_time = cur_t
            action[4] = 1
            if cur_t <= self.grasp_time + 0.5:
                action[:3] = [0, 0, 0.5]
            else:
                action[:3] = self.post_grasp_dest - ee_pos
        return action, 0

