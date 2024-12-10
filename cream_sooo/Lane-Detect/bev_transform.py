# bev_transform.py

import numpy as np
import cv2
import math
import rospy
from utils_functions import calculate_ym_per_pix, calculate_xm_per_pix

def rotationMtx(yaw, pitch, roll):
    R_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(roll), -np.sin(roll), 0],
        [0, np.sin(roll), np.cos(roll), 0],
        [0, 0, 0, 1]
    ])
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch), 0],
        [0, 1, 0, 0],
        [-np.sin(pitch), 0, np.cos(pitch), 0],
        [0, 0, 0, 1]
    ])
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, 0],
        [np.sin(yaw), np.cos(yaw), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    R = R_z @ R_y @ R_x
    return R

def traslationMtx(x, y, z):
    M = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])
    return M

def project2img_mtx(params_cam):
    if params_cam.get("ENGINE", "DEFAULT") == 'UNITY':
        fc_x = params_cam["HEIGHT"] / (2 * np.tan(np.deg2rad(params_cam["FOV"] / 2)))
        fc_y = params_cam["HEIGHT"] / (2 * np.tan(np.deg2rad(params_cam["FOV"] / 2)))
    else:
        fc_x = params_cam["WIDTH"] / (2 * np.tan(np.deg2rad(params_cam["FOV"] / 2)))
        fc_y = params_cam["WIDTH"] / (2 * np.tan(np.deg2rad(params_cam["FOV"] / 2)))

    cx = params_cam["WIDTH"] / 2
    cy = params_cam["HEIGHT"] / 2

    R_f = np.array([
        [fc_x, 0, cx],
        [0, fc_y, cy]
    ])

    return R_f

class BEVTransform:
    """
    BEV (Bird's Eye View) 변환을 수행하는 클래스입니다.
    """

    def __init__(self, params_cam, xb=10.0, zb=10.0):
        self.xb = xb
        self.zb = zb

        self.theta = np.deg2rad(params_cam["PITCH"])
        self.width = params_cam["WIDTH"]
        self.height = params_cam["HEIGHT"]
        self.x = params_cam["X"]

        if params_cam.get("ENGINE", "DEFAULT") == "UNITY":
            self.alpha_r = np.deg2rad(params_cam["FOV"] / 2)
            self.fc_y = params_cam["HEIGHT"] / (2 * np.tan(np.deg2rad(params_cam["FOV"] / 2)))
            self.alpha_c = np.arctan2(params_cam["WIDTH"] / 2, self.fc_y)
            self.fc_x = self.fc_y
        else:
            self.alpha_c = np.deg2rad(params_cam["FOV"] / 2)
            self.fc_x = params_cam["WIDTH"] / (2 * np.tan(np.deg2rad(params_cam["FOV"] / 2)))
            self.alpha_r = np.arctan2(params_cam["HEIGHT"] / 2, self.fc_x)
            self.fc_y = self.fc_x

        self.h = params_cam["Z"] + 0.34  # 카메라 높이 + 차량 높이?

        self.n = float(self.width)
        self.m = float(self.height)

        # 차량 좌표계에서 세계 좌표계로의 변환 행렬 (4x4)
        self.RT_b2g = (
            traslationMtx(self.xb, 0, self.zb) @
            rotationMtx(np.deg2rad(-90), 0, 0) @
            rotationMtx(0, 0, np.deg2rad(180))
        )

        # 2D 프로젝션 행렬
        self.proj_mtx = project2img_mtx(params_cam)

        # BEV 변환 행렬 구축
        self._build_tf(params_cam)

    def calc_Xv_Yu(self, U, V):
        Xv = self.h * (np.tan(self.theta) * (1 - 2 * (V - 1) / (self.m - 1)) * np.tan(self.alpha_r) - 1) / \
             (-np.tan(self.theta) + (1 - 2 * (V - 1) / (self.m - 1)) * np.tan(self.alpha_r))

        Yu = (1 - 2 * (U - 1) / (self.n - 1)) * Xv * np.tan(self.alpha_c)

        return Xv, Yu

    def _build_tf(self, params_cam):
        """
        BEV 변환을 위한 원근 변환 행렬과 역변환 행렬을 계산합니다.
        """
        # 소스 포인트 설정 (수동 또는 자동)
        
        ### 10m 앞
        # src_pts = np.float32([
        #         [523, 492],  # 좌상 (Left Top)
        #         [752, 492],  # 우상 (Right Top)
        #         [224, 716],  # 좌하 (Left Bottom)
        #         [1035, 716]   # 우하 (Right Bottom)
        #     ])

        src_pts = np.float32([
                [447, 549],  # 좌상 (Left Top)
                [831, 549],  # 우상 (Right Top)
                [226, 713],  # 좌하 (Left Bottom)
                [1034, 714]   # 우하 (Right Bottom)
            ])



        # 목적지 포인트 설정 (BEV 이미지에서의 위치)
        dst_pts = np.float32([
            [self.width * 0.2, 0],                 # 좌상
            [self.width * 0.8, 0],                 # 우상
            [self.width * 0.2, self.height],       # 좌하
            [self.width * 0.8, self.height]        # 우하
        ])

        # 원근 변환 행렬 계산
        self.perspective_tf = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.perspective_inv_tf = cv2.getPerspectiveTransform(dst_pts, src_pts)

        rospy.logdebug(f"Source Points: {src_pts}")
        rospy.logdebug(f"Destination Points: {dst_pts}")
        rospy.logdebug(f"Perspective Transform Matrix:\n{self.perspective_tf}")
        rospy.logdebug(f"Inverse Perspective Transform Matrix:\n{self.perspective_inv_tf}")

    def warp_bev_img(self, img):
        """
        입력 이미지를 BEV 변환하여 반환합니다.

        Args:
            img (numpy.ndarray): 입력 이미지.

        Returns:
            numpy.ndarray: BEV 변환된 이미지.
        """
        img_warp = cv2.warpPerspective(img, self.perspective_tf, (self.width, self.height), flags=cv2.INTER_LINEAR)
        return img_warp

    def warp_inv_img(self, img):
        """
        BEV 변환된 이미지를 원래의 뷰로 역변환합니다.

        Args:
            img (numpy.ndarray): BEV 변환된 이미지.

        Returns:
            numpy.ndarray: 역변환된 이미지.
        """
        img_f = cv2.warpPerspective(img, self.perspective_inv_tf, (self.width, self.height), flags=cv2.INTER_LINEAR)
        return img_f

    def recon_lane_pts(self, img):
        """
        BEV 이미지의 차선 포인트를 차량 좌표계의 실제 거리 단위 포인트로 변환합니다.

        Args:
            img (numpy.ndarray): BEV 이미지.

        Returns:
            numpy.ndarray: 차량 좌표계의 포인트들 (4xN).
        """
        if cv2.countNonZero(img) != 0:
            UV_mark = cv2.findNonZero(img).reshape([-1, 2])
            U, V = UV_mark[:, 0].reshape([-1, 1]), UV_mark[:, 1].reshape([-1, 1])
            Xv, Yu = self.calc_Xv_Yu(U, V)

            xyz_g = np.concatenate([
                Xv.reshape([1, -1]) + self.x,
                Yu.reshape([1, -1]),
                np.zeros_like(Yu.reshape([1, -1])),
                np.ones_like(Yu.reshape([1, -1]))
            ], axis=0)

            xyz_g = xyz_g[:, xyz_g[0, :] >= 0]
            xyz_bird = np.linalg.inv(self.RT_b2g) @ xyz_g
        else:
            xyz_bird = np.zeros((4, 10))

        return xyz_bird

    def project_lane2img(self, x_pred, y_pred_l, y_pred_r):
        """
        차량 좌표계의 차선 포인트를 이미지 좌표로 투영합니다.

        Args:
            x_pred (numpy.ndarray): 예측된 X 좌표들.
            y_pred_l (numpy.ndarray): 왼쪽 차선의 Y 좌표들.
            y_pred_r (numpy.ndarray): 오른쪽 차선의 Y 좌표들.

        Returns:
            tuple: (xyl, xyr) 이미지 좌표계의 왼쪽 및 오른쪽 차선 포인트들.
        """
        xyz_l_g = np.concatenate([
            x_pred.reshape([1, -1]),
            y_pred_l.reshape([1, -1]),
            np.zeros_like(y_pred_l.reshape([1, -1])),
            np.ones_like(y_pred_l.reshape([1, -1]))
        ], axis=0)

        xyz_r_g = np.concatenate([
            x_pred.reshape([1, -1]),
            y_pred_r.reshape([1, -1]),
            np.zeros_like(y_pred_r.reshape([1, -1])),
            np.ones_like(y_pred_r.reshape([1, -1]))
        ], axis=0)

        xyz_l_b = np.linalg.inv(self.RT_b2g) @ xyz_l_g
        xyz_r_b = np.linalg.inv(self.RT_b2g) @ xyz_r_g

        xyl = self.project_pts2img(xyz_l_b)
        xyr = self.project_pts2img(xyz_r_b)

        xyl = self.crop_pts(xyl)
        xyr = self.crop_pts(xyr)

        return xyl, xyr

    def project_pts2img(self, xyz_bird):
        """
        3D 포인트를 2D 이미지 좌표로 투영합니다.

        Args:
            xyz_bird (numpy.ndarray): 4xN 형태의 3D 포인트들.

        Returns:
            numpy.ndarray: Nx2 형태의 2D 이미지 좌표들.
        """
        xc, yc, zc = xyz_bird[0, :].reshape([1, -1]), xyz_bird[1, :].reshape([1, -1]), xyz_bird[2, :].reshape([1, -1])

        xn, yn = xc / (zc + 1e-4), yc / (zc + 1e-4)

        xyi = self.proj_mtx @ np.concatenate([xn, yn, np.ones_like(xn)], axis=0)
        xyi = xyi[0:2, :].T

        return xyi

    def crop_pts(self, xyi):
        """
        이미지 범위 내의 포인트들만 남깁니다.

        Args:
            xyi (numpy.ndarray): Nx2 형태의 2D 이미지 좌표들.

        Returns:
            numpy.ndarray: 범위 내의 Nx2 2D 이미지 좌표들.
        """
        xyi = xyi[np.logical_and(xyi[:, 0] >= 0, xyi[:, 0] < self.width), :]
        xyi = xyi[np.logical_and(xyi[:, 1] >= 0, xyi[:, 1] < self.height), :]
        return xyi