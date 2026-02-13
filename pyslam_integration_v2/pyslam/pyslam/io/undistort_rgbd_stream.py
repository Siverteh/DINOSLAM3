import cv2
import numpy as np

class UndistortRGBDStream:
    """
    Precompute undistort/rectify maps once and apply per-frame remap for RGB + depth.
    Works for pinhole + distortion (TUM).
    """

    def __init__(self, K: np.ndarray, D: np.ndarray, size_wh: tuple[int, int]):
        w, h = int(size_wh[0]), int(size_wh[1])
        self.size_wh = (w, h)

        K = np.asarray(K, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64).reshape(-1)

        # NOTE: Keep same intrinsics (no "optimal new K") to avoid changing coordinate system.
        newK = K.copy()

        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            K, D, R=np.eye(3), newCameraMatrix=newK, size=(w, h), m1type=cv2.CV_32FC1
        )

    def __call__(self, color_bgr: np.ndarray, depth: np.ndarray):
        """
        color_bgr: HxWx3 uint8
        depth: HxW (uint16 or float32) depth image

        Returns: (color_bgr_undist, depth_undist)
        """
        w, h = self.size_wh
        assert color_bgr.shape[1] == w and color_bgr.shape[0] == h, \
            f"Color shape {color_bgr.shape} mismatch expected {(h,w)}"
        assert depth.shape[1] == w and depth.shape[0] == h, \
            f"Depth shape {depth.shape} mismatch expected {(h,w)}"

        # Color: linear interpolation
        color_u = cv2.remap(color_bgr, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

        # Depth: nearest neighbor (do NOT interpolate depth with linear)
        if depth.dtype not in (np.uint16, np.float32, np.float64):
            depth = depth.astype(np.float32)
        depth_u = cv2.remap(depth, self.map1, self.map2, interpolation=cv2.INTER_NEAREST)

        # Keep contiguous arrays (pySLAM sometimes assumes contiguous)
        return np.ascontiguousarray(color_u), np.ascontiguousarray(depth_u)
