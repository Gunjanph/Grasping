import cv2
import warnings
import numpy as np
from .dcem_model import Model
from .calculate_flow import FlowNet2Utils
import os
import torch
from .utils.photo_error import mse_
from .utils.flow_utils import flow2img
from utils.logger import logger
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# np.random.seed(0)
# warnings.filterwarnings("ignore")
# torch.backends.cudnn.deterministic = True
torch.backFlowNet2Utilsends.cudnn.benchmark = True
# torch.manual_seed(0)
# torch.autograd.set_detect_anomaly(True)


class Rtvs:
    """
    Code for RTVS: Real-Time Visual Servoing, IROS 2021
    """

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
        if isinstance(img_goal, str): # finding goal img
            img_goal = np.asarray(Image.open(img_goal))
        self.img_goal = img_goal # defining initial conditions as given
        self.horizon = horizon
        self.iterations = iterations
        self.cam_k = cam_k
        self.ct = ct
        self.flow_utils = FlowNet2Utils() # It is calculating and storing the flow using FLowNet
        self.vs_lstm = Model().to(device="cuda:0") # defining the lstm model defined in dcem model.py
        self.optimiser = torch.optim.Adam(
            self.vs_lstm.parameters(), lr=LR, betas=(0.93, 0.999)
        ) # The optimser used is LSTM  is ADAM with learning rate of 0.005
        self.loss_fn = torch.nn.MSELoss(size_average=False) # MSE err/ L2 norm

    @staticmethod
    def detect_mask(rgb_img, pixrange=((0, 100, 100), (10, 255, 255))):
        hsv_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        # segment red colour from image
        mask = cv2.inRange(hsv_image, *pixrange)
        mask[mask != 0] = 1
        return mask[:, :, None] # detecting the mask for a given range, if not mask then remove pixel value and make it black

    @staticmethod
    def save_flow(flow, cnt): # save the image containing flow
        name = f"flow_{str(cnt).zfill(5)}.png"
        last_flow_path = "imgs/_flow_last.png"
        Image.fromarray(flow2img(flow)).save(f"imgs/{name}")
        if os.path.exists(last_flow_path):
            os.remove(last_flow_path)
        os.symlink(name, last_flow_path)

    def get_vel(self, img_src, pre_img_src=None, depth=None):
        """
        img_src = current RGB camera image
        prev_img_src = previous RGB camera image
                    (to be used for depth estimation using flowdepth)
        """
        img_goal = self.img_goal
        flow_utils = self.flow_utils
        vs_lstm = self.vs_lstm
        loss_fn = self.loss_fn
        optimiser = self.optimiser
        img_src = img_src
        if depth is not None:
            depth = depth
        if pre_img_src is not None:
            pre_img_src = pre_img_src
        ct = self.ct

        # if photo_error_val < 6000 and photo_error_val > 3600:
        #     self.horizon = 10 * (photo_error_val / 6000)
        # elif photo_error_val < 3000:
        #     self.horizon = 6

        self.cnt = 0 if not hasattr(self, "cnt") else self.cnt + 1 # define cnt if already doesn't exist
        obj_mask = self.detect_mask(img_src, ((50, 100, 100), (70, 255, 255))) # finding mask
        photo_error_val = mse_(img_src, img_goal) #mse of src and goal img
        obj_mask = obj_mask[::ct, ::ct] # depth mask
        f12 = flow_utils.flow_calculate(img_src, img_goal)[::ct, ::ct] # calculating flow at a current depth for a mask
        f12 = f12 * obj_mask
        self.save_flow(f12, self.cnt)

        if depth is None: # finding flow if depth is none
            flow_depth_proxy = (
                flow_utils.flow_calculate(img_src, pre_img_src).astype("float64")
            )[::ct, ::ct]
            flow_depth = np.linalg.norm(flow_depth_proxy, axis=2).astype("float64")
            final_depth = 0.6 / (1 + np.exp(-1 / flow_depth)) - 0.5
        else: # find final depth and scale it
            final_depth = (depth[::ct, ::ct] + 1) / 10

        vel, Lsx, Lsy = get_interaction_data(final_depth, ct, self.cam_k) # get interaction matrix
        Lsx = Lsx * obj_mask
        Lsy = Lsy * obj_mask

        Lsx = torch.tensor(Lsx, dtype=torch.float32).to(device="cuda:0") # multidimensional matrix
        Lsy = torch.tensor(Lsy, dtype=torch.float32).to(device="cuda:0")
        f12 = torch.tensor(f12, dtype=torch.float32).to(device="cuda:0")
        f12 = vs_lstm.pooling(f12.permute(2, 0, 1).unsqueeze(dim=0)) # redefining parameters of LSTM by defining a pooling layer

        for itr in range(self.iterations):
            vs_lstm.v_interm = [] # store velocity
            vs_lstm.f_interm = [] # store sigma
            vs_lstm.mean_interm = [] #store mean

            vs_lstm.zero_grad() # set all grads to zero to avoid error bcz of prev gradients calculated.
            f_hat = vs_lstm.forward(vel, Lsx, Lsy, self.horizon, f12) # run the lstm and converge it
            loss = loss_fn(f_hat, f12) # loss between LSTM and the flow calculated

            logger.debug(rtvs_mse=loss.item() ** 0.5, rtvs_itr=itr)
            loss.backward(retain_graph=True) #back propogation
            optimiser.step()

        # Do not accumulate flow and velocity at train time
        vs_lstm.v_interm = []
        vs_lstm.f_interm = []
        vs_lstm.mean_interm = []

        with torch.no_grad():
            f_hat = vs_lstm.forward(
                vel, Lsx, Lsy, -self.horizon, f12.to(torch.device("cuda:0"))
            ) # if no gradient then no need of back propogation

        vel = vs_lstm.v_interm[0].detach().cpu().numpy() # velocity from the lstm
        logger.info(RAW_RTVS_VELOCITY=vel)
        # vel = vel
        # vel[:] = 0
        # vel[2] = 1
        # vel[1] *= -1
        # vel[2] *= -1

        return vel, photo_error_val


def get_interaction_data(d1, ct, cam_k): # using fxn to define interaction matrix
    kx = cam_k[0, 0]
    ky = cam_k[1, 1]
    Cx = cam_k[0, 2]
    Cy = cam_k[1, 2]
    xyz = np.zeros([d1.shape[0], d1.shape[1], 3])
    Lsx = np.zeros([d1.shape[0], d1.shape[1], 6])
    Lsy = np.zeros([d1.shape[0], d1.shape[1], 6])

    med = np.median(d1)

    def xyz_func(i, j, k):
        i = i.astype(int)
        j = j.astype(int)
        return (
            ((0.5 * (k - 1) * (k - 2)) * (ct * j - Cx) / kx)
            + ((-k * (k - 2)) * (ct * i - Cy) / ky)
            + ((0.5 * k * (k - 1)) * ((d1[i, j] == 0) * med + d1[i, j]))
        )

    def lsx_func(i, j, k):
        i = i.astype(int)
        j = j.astype(int)
        k = k.astype(int)
        return (
            ((k == 0).astype(int) * -1 / xyz[i, j, 2])
            + ((k == 2).astype(int) * xyz[i, j, 0] / xyz[i, j, 2])
            + ((k == 3).astype(int) * xyz[i, j, 0] * xyz[i, j, 1])
            + ((k == 4).astype(int) * (-(1 + xyz[i, j, 0] ** 2)))
            + ((k == 5).astype(int) * xyz[i, j, 1])
        )

    def lsy_func(i, j, k):
        i = i.astype(int)
        j = j.astype(int)
        k = k.astype(int)
        return (
            ((k == 1).astype(int) * -1 / xyz[i, j, 2])
            + ((k == 2).astype(int) * xyz[i, j, 1] / xyz[i, j, 2])
            + ((k == 3).astype(int) * (1 + xyz[i, j, 1] ** 2))
            + ((k == 4).astype(int) * -xyz[i, j, 0] * xyz[i, j, 1])
            + ((k == 5).astype(int) * -xyz[i, j, 0])
        )

    xyz = np.fromfunction(xyz_func, (d1.shape[0], d1.shape[1], 3), dtype=float)
    Lsx = np.fromfunction(lsx_func, (d1.shape[0], d1.shape[1], 6), dtype=float)
    Lsy = np.fromfunction(lsy_func, (d1.shape[0], d1.shape[1], 6), dtype=float)

    return None, Lsx, Lsy
