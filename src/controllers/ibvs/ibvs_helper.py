import cv2
import numpy as np
from utils.logger import logger


def detect_corners(rgb_img):
    # cv2.imwrite("before.jpg", rgb_img)
    hsv_image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    # print(hsv_image)
    # cv2.imwrite("after.jpg", hsv_image)
    # segment red colour from image
    mask = cv2.inRange(hsv_image, (20, 0, 0), (86, 255, 255))
    # cv2.imwrite("mask.jpg", mask)
    whites = np.argwhere(mask == 255).tolist()
    # print(whites)
    if not whites:
        return None
    top_left = min(whites)
    bottom_right = max(whites)
    top_right = [top_left[0], bottom_right[1]]
    bottom_left = [bottom_right[0], top_left[1]]
    corners = np.array([top_left, top_right, bottom_right, bottom_left])

    # in x,y format
    corners[:, [0, 1]] = corners[:, [1, 0]]
    return corners


def approach_depth(depth_img, corners):
    corners = corners.astype(np.int32)
    top_left = corners[0]
    bottom_right = corners[2]
    depth = depth_img[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
    ap_depth = depth.mean()
    logger.info(f"Approach depth: {ap_depth}")
    return ap_depth

def detect_mask(rgb_img, pixrange=((0, 100, 100), (10, 255, 255))):
    hsv_image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    # segment red colour from image
    mask = cv2.inRange(hsv_image, *pixrange)
    mask[mask != 0] = 1
    return mask.reshape((rgb_img.shape[0], rgb_img.shape[1], 1))


class IBVSHelper:
    def __init__(
        self, target_image_file, cam_k, lm_params={}, show_corners_window=False
    ):
        lm_params.setdefault("mu", 0.000000000001)
        lm_params.setdefault("lambda", 0.00000000001)
        target_image = cv2.imread(target_image_file)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        self.target_image = target_image
        self.lm_params = lm_params
        self.lmda = self.lm_params["lambda"]
        self.lmda_multiplier = 1.1
        self.cam_k = np.array(cam_k, dtype=np.float32)
        self._init_targets(target_image)
        self.last_mult_mat = None
        self.show_corners_window = show_corners_window

    def _get_L(self, Z):
        if not isinstance(Z, np.ndarray):
            Z = np.full(self.target_image.shape[:2], Z, dtype=np.float32)
        L = np.zeros((len(self.target_s), 6), dtype=np.float32)
        for i in range(len(self.target_corners)):
            x, y = self.target_s[2 * i], self.target_s[2 * i + 1]
            _Z = Z[self.target_corners[i, 1], self.target_corners[i, 0]]
            L[2 * i, 0] = -1 / _Z
            L[2 * i, 1] = 0
            L[2 * i, 2] = x / _Z
            L[2 * i, 3] = x * y
            L[2 * i, 4] = -(1 + x**2)
            L[2 * i, 5] = y
            L[2 * i + 1, 0] = 0
            L[2 * i + 1, 1] = -1 / _Z
            L[2 * i + 1, 2] = y / _Z
            L[2 * i + 1, 3] = 1 + y**2
            L[2 * i + 1, 4] = -x * y
            L[2 * i + 1, 5] = -x
        return L

    def mult_mat(self, L):
        try:
            mm = (
                -self.lmda
                * np.linalg.pinv(L.T @ L + self.lm_params["mu"] * np.diag(L.T @ L))
                @ L.T
            )
        except np.linalg.LinAlgError:
            logger.warning("Singular IBVS matrix")
            mm = self.last_mult_mat.copy()
        self.last_mult_mat = mm.copy()
        return mm

    def _init_targets(self, target_image):
        self.target_corners = detect_corners(target_image)
        assert self.target_corners is not None, "No corners detected in target image"
        self.target_s = self._get_s_from_corners(self.target_corners)

    def _get_s_from_corners(self, corners):
        s = corners.copy().flatten().astype(np.float32)
        px = self.cam_k[0, 0]
        py = self.cam_k[1, 1]
        cx = self.cam_k[0, 2]
        cy = self.cam_k[1, 2]
        for i in range(len(corners)):
            s[2 * i] = (s[2 * i] - cx) / px
            s[2 * i + 1] = (s[2 * i + 1] - cy) / py
        return s

    def get_velocity(self, current_image, depth_image):
        current_corners = detect_corners(current_image)
        if current_corners is None:
            return np.zeros(3), 0.0
        display_img = current_image.copy()
        for x, y in self.target_corners:
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
        for x, y in current_corners:
            cv2.circle(display_img, (x, y), 5, (255, 0, 0), -1)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
        if self.show_corners_window:
            cv2.imshow("corners", display_img)
            cv2.waitKey(1)
        current_s = self._get_s_from_corners(current_corners)
        e = current_s - self.target_s
        # v = self.mult_mat(self._get_L(approach_depth(depth_image, current_corners))) @ e
        v = self.mult_mat(self._get_L(depth_image)) @ e
        self.lmda *= self.lmda_multiplier
        v = v[:3]
        
        # IOU error 
        mask_src = detect_mask(current_image, ((20, 0, 0), (86, 255, 255)))
        mask_goal = detect_mask(self.target_image, ((20, 0, 0), (86, 255, 255)))
        intersection = np.logical_and(mask_src, mask_goal)
        union = np.logical_or(mask_src, mask_goal)
        iou_score = np.sum(intersection) / (np.sum(union) + 0.001)
        
        return v, iou_score, np.linalg.norm(e)
