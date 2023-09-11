import pybullet as p
import pybullet_data
import numpy as np
from PIL import Image
import os
import random

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

    def simulation(self):
        # Run the simulation for a fixed amount of steps.
        for i in range(20):
            position, orientation = p.getBasePositionAndOrientation(self.r2d2)
            x, y, z = position
            roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
            # print(f"{i:3}: x={x:0.10f}, y={y:0.10f}, z={z:0.10f}), roll={roll:0.10f}, pitch={pitch:0.10f}, yaw={yaw:0.10f}")
            p.stepSimulation()

    def capture_images(self):
        eye_pos = []

        angles = np.arange(0, 2*np.pi, 2*np.pi/self.num_frames)
        for n in angles:
            h = round(random.uniform(1.5,3), 2)
            r = round(random.uniform(2,4), 2)
            eye_pos.append([r*np.cos(n), r*np.sin(n), h])
        itr = 0
        for pos in eye_pos:
            print(pos)
            p.configureDebugVisualizer(lightPosition=[pos[0], pos[1], 3])
            position, orientation = p.getBasePositionAndOrientation(self.r2d2)
            print(position, orientation)
            width = 512
            height = 512
            img_arr = p.getCameraImage(
                width,
                height,
                viewMatrix=p.computeViewMatrix(
                    cameraEyePosition=pos,
                    cameraTargetPosition=[0 , 0 , self.target_h],
                    cameraUpVector=[0, 0, 1]
                ),
                projectionMatrix=p.computeProjectionMatrixFOV(
                    fov=60,
                    aspect=width/height,
                    nearVal=0.01,
                    farVal=100,
                ),
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                shadow=False,
            )

            width, height, rgba, depth, mask = img_arr
            im_rgb = Image.fromarray(rgba, 'RGBA')
            # fol = self.folder + self.id
            # im_rgb = im_rgb.save(fol + "_rgb_" + str(itr) + ".png")

            itr+=1

if __name__ == "__main__":
    folder =   "./frames/frames_pybullet_3new/default_plane/test/"
    plane = "plane.urdf"

    # folder =   "./frames/frames_pybullet_3new/custom_plane_new/test/"
    # plane = "simple_plane.urdf"

    path = "./3822/mobility.urdf"
    id_list = [3822]
    id = str(id_list[0])
    print("Extracting: ", id)
    target_h = 0.75
    h = 1.2
    r = 2
    num_frames = 4
    # path = path.replace("3635", id)
    image_extarctor = Image_Extractor(path, r, h, target_h, id, num_frames, folder, plane)
    image_extarctor.capture_images()