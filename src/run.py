import pybullet as p
import pybullet_data
import numpy as np
from PIL import Image
import os
import random
import argparse
from scipy.spatial.transform import Rotation as R

class Image_Extractor:
    def __init__(self, urfd_path, r, h, target_h ,id, num_frames, folder, plane):
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.resetSimulation()
        # Load our simulation floor plane at the origin (0, 0, 0).
        radius=10
        p.loadURDF(plane)
        p.configureDebugVisualizer(shadowMapWorldSize=5)
        p.configureDebugVisualizer(shadowMapResolution=8192)
        # Load an R2D2 droid at the position at 0.5 meters height in the z-axis.
        self.r2d2 = p.loadURDF(urfd_path, [0, 0, 0.8])
        # Set the gravity to Earth's gravity.
        p.setGravity(0, 0, -9.807)
        self.r = r
        self.h = h
        self.id = id
        self.num_frames = num_frames
        self.target_h = target_h
        self.folder = folder

    def find_camera_orientation(self, cameraEyePosition, cameraTargetPosition, cameraUpVector):
        dir_vec = np.array(cameraTargetPosition) - np.array(cameraEyePosition)
        dir_unit = dir_vec / np.linalg.norm(dir_vec)
        
        theta_pitch = np.arctan2(np.sqrt(dir_unit[0]**2 + dir_unit[1]**2), dir_unit[2])
        theta_yaw = np.arctan2(dir_unit[1], dir_unit[0])
        
        # Assuming no roll, as 'cameraUpVector' is specified.
        theta_roll = 0.0  
        
        return np.deg2rad([theta_pitch, theta_yaw, theta_roll])
    
    def capture_images(self, init_cfg):
        from controllers.rtvs import Rtvs, RTVSController
        
        width = 512
        height = 512
        pos = init_cfg[0]
        
        projectionmatrix=p.computeProjectionMatrixFOV(
                    fov=60,
                    aspect=width/height,
                    nearVal=0.01,
                    farVal=100,
                )
        print(projectionmatrix[0])
        f = projectionmatrix[0]
        
        self.cam_to_gt_R = R.from_euler("xyz", self.find_camera_orientation(pos, [0 , 0 , self.target_h], [0, 0, 1]))

        self.controller = RTVSController(
            Rtvs("./dest.png", np.array([[f, 0, width/2],
              [0, f, height/2],
              [0, 0, 1]])),
            self.cam_to_gt_R,
            max_speed=0.7,
        )
        action = -1
        init = 1
        itr=0
        observations = {}  # Define the dictionary
        while action == -1:
            p.configureDebugVisualizer(lightPosition=pos)
            img_arr = p.getCameraImage(
                width,
                height,
                viewMatrix=p.computeViewMatrix(
                    cameraEyePosition=pos,
                    cameraTargetPosition=[0 , 0 , self.target_h],
                    cameraUpVector=[0, 0, 1]
                ),
                projectionMatrix=projectionmatrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                shadow=False,
            )
            if itr!=0:
                prev_rgb = rgb    
            
            width, height, rgba, depth, mask = img_arr
            rgb = rgba[:, :, :3]
            rgbSave= Image.fromarray(rgba, 'RGBA')
            
            if itr==0:
                prev_rgb = rgb
            
            observations["rgb_img"] = rgb
            observations["depth_img"] = depth
            observations["prev_rgb_img"] = prev_rgb
            # print()
            fol = self.folder
            im_rgb = rgbSave.save(fol + "rgb_" + str(itr) + ".png")
            vel_ac, error, mse= self.controller.get_action(observations)
            action = vel_ac[4]
            print("action:    ",action, end='\n')
            vel = vel_ac[:3]
            pos = pos+vel
            itr = itr+1

    
def simulate(init_cfg, **kwargs):
    folder =   "../results/3822/"
    plane = "simple_plane.urdf"
    
    path = "./3822/mobility.urdf"
    id_list = [3822]
    id = str(id_list[0])
    print("Extracting: ", id)
    target_h = 0.75
    h = 1.2
    r = 2
    num_frames = 4
    path = path.replace("3822", id)
    image_extarctor = Image_Extractor(path, r, h, target_h, id, num_frames, folder, plane)
    image_extarctor.capture_images(init_cfg)
    # env = URRobotGym(*init_cfg, gui=gui, controller_type=controller, **kwargs)
    # return env.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--controller",
        type=str,
        default="gt",
        help="controller",
        choices=["gt", "rtvs", "ibvs", "ours", "deepmpc"],
    )
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--gui", action="store_true", help="show gui")
    parser.add_argument("--no-gui", dest="gui", action="store_false", help="no gui")
    parser.add_argument("--record", action="store_true", help="save imgs")
    parser.add_argument("--flowdepth", action="store_true", help="use flow_depth")
    parser.add_argument("--no-record", dest="record", action="store_false")
    parser.set_defaults(gui=False, record=True)
    parser.add_argument("--seed", type=int, default=None, help="seed")
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    init_cfg = ([2.93, 0.0, 1.61], [0, 0, 0])
    # init_cfg = [[0.45, -0.05, 0.851], [-0.01, 0.03, 0]]
    if args.random:
        init_cfg[1] = get_random_config()[1]

    return simulate(
        init_cfg
    )


if __name__ == "__main__":
    main()