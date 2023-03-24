import os
import shutil
from types import SimpleNamespace
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import pybullet as pb
from airobot import Robot
from airobot.arm.ur5e_pybullet import UR5ePybullet as UR5eArm
from airobot.utils.common import clamp
from airobot.utils.common import euler2quat, quat2euler, euler2rot
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
from utils.sim_utils import get_random_config
from utils.logger import logger
from scipy.spatial.transform import Rotation as R
from PIL import Image
import argparse
import pybullet_data

from pybullet_object_models import ycb_objects

robot = Robot(
                "ur5e_2f140",
                # Have to keep openGL render off for the texture to work
                pb_cfg={"gui": False, "realtime": False, "opengl_render": False},
            )
cam = robot.cam
pb.removeBody(robot.arm.floor_id)
# physicsClient = pb.connect(pb.DIRECT)
pb.resetDebugVisualizerCamera(cameraTargetPosition=[0, 0, 0],
            cameraDistance=2,
            cameraPitch=-40,
            cameraYaw=90,)
pb.setTimeStep(1 / 240.)

pb.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = pb.loadURDF(
    os.path.join(ycb_objects.getDataPath(), 'YcbChipsCan', "model.urdf"), 
    [0.45, -0.05, 0.851], 
    globalScaling = 0.5
    )

shutil.rmtree("pc", ignore_errors=True)
os.makedirs("pc", exist_ok=True)

flags = pb.URDF_USE_INERTIA_FROM_FILE

pb.setGravity(0, 0, 0)
pb.setRealTimeSimulation(1)

camPos = np.array([ 0.5 , -0.33,  0.92])
# camTarget = [0.45, -0.05, 0.851]
# camUp = [0, 1, 0]
camOri = np.deg2rad([-105, 0, 0])

cam.set_cam_ext(camPos, camOri)

rgb1, depth1, seg1 = cam.get_images(
            True, True, True, shadow=0, lightDirection=[0, 0, 2]
            ) 
pcd_3d1, pcd_rgb1 = cam.get_pcd(
            in_world = True, rgb_image=rgb1, depth_image=depth1
        )
plt.imsave("pc/depth1.png", depth1)
plt.imsave("pc/rgb1.png", rgb1)

camPos = np.array([ 0.5 , 0.23,  0.92])
# camTarget = [0.45, -0.05, 0.851]
# camUp = [0, 1, 0]
camOri = np.deg2rad([-105, 0, 180])

cam.set_cam_ext(camPos, camOri)

rgb2, depth2, seg2 = cam.get_images(
            True, True, True, shadow=0, lightDirection=[0, 0, 2]
            ) 
pcd_3d2, pcd_rgb2 = cam.get_pcd(
            in_world = True, rgb_image=rgb2, depth_image=depth2
        )
plt.imsave("pc/depth2.png", depth2)
plt.imsave("pc/rgb2.png", rgb2)
print(cam.get_cam_int())

pcd = np.vstack([pcd_3d1, pcd_3d2])
print(pcd.shape)
# pcd = [i for i in pcd if np.linalg.norm(i)<2]
# pcd = np.array(pcd)
np.save('pc/data', pcd)

# Creating 3D figure
xs = []
ys = []
zs = []

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax = plt.axes(projection='3d')
xs = pcd[:,0]
ys = pcd[:,1]
zs = pcd[:,2]
# for i in range(pcd.shape[0]):
#             [x,y,z,] = pcd[i]
#             # print(h,w)
#             xs.append(x)
#             ys.append(y)
#             zs.append(z)
# for i in range(pcd_3d2.shape[0]):
#             [x,y,z,] = pcd_3d2[i]
#             # print(h,w)
#             xs.append(x)
#             ys.append(y)
#             zs.append(z)
        
print(max(xs), min(xs), max(ys), min(ys), max(zs), min(zs))

ax.scatter(xs, ys, zs, c=zs, marker='.')
ax.set_xlim(0,2)
ax.set_ylim(0,2)
ax.set_zlim(-1,1)
plt.savefig("pc/pc.png")
plt.show()