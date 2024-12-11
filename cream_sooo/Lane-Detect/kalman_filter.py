# kalman_filter.py

import numpy as np
from filterpy.kalman import KalmanFilter
from lane_detection import calculate_center_lane
import rospy

class KalmanFilterLane:
    def __init__(self):
        # 상태 벡터를 [a, b, c, a', b', c']로 확장
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.F = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        self.kf.P *= 1000
        self.kf.R = np.eye(3) * 5
        self.kf.Q = np.eye(6) * 0.1
        self.kf.x = np.zeros(6)

    def predict(self):
        self.kf.predict()
        return self.kf.x[:3]  # 다항식 계수 반환

    def update(self, measurement):
        self.kf.update(measurement)
        return self.kf.x[:3]

class LaneFitting:
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.center_fitx = None
        self.center_fit_history = []
        self.max_history = 5
        self.kalman_filter_left = KalmanFilterLane()
        self.kalman_filter_right = KalmanFilterLane()
        self.prev_fit = None

    def update_fit(self, left_fit, right_fit, ploty):
        # 유효한 차선만 업데이트
        valid_left = left_fit is not None
        valid_right = right_fit is not None

        if valid_left or valid_right:
            # 왼쪽 차선 업데이트
            if valid_left:
                predicted_left = self.kalman_filter_left.predict()
                updated_left = self.kalman_filter_left.update(left_fit)
                self.left_fit = updated_left
                left_fitx = updated_left[0]*ploty**2 + updated_left[1]*ploty + updated_left[2]
            else:
                left_fitx = None

            # 오른쪽 차선 업데이트
            if valid_right:
                predicted_right = self.kalman_filter_right.predict()
                updated_right = self.kalman_filter_right.update(right_fit)
                self.right_fit = updated_right
                right_fitx = updated_right[0]*ploty**2 + updated_right[1]*ploty + updated_right[2]
            else:
                right_fitx = None

            # 중앙 차선 계산
            self.center_fitx = calculate_center_lane(
                left_fitx, right_fitx, ploty, default_center_x=640
            )

            # 히스토리에 추가 및 평균
            if self.center_fitx is not None:
                self.center_fit_history.append(self.center_fitx)
                if len(self.center_fit_history) > self.max_history:
                    self.center_fit_history.pop(0)
                self.center_fitx = np.mean(self.center_fit_history, axis=0)
        else:
            rospy.logwarn("유효한 차선 검출이 없습니다. 이전 차선 정보를 유지합니다.")
            # 이전 차선 정보를 유지
            pass

    def get_center_fitx(self, ploty):
        return self.center_fitx if self.center_fitx is not None else None
