#!/usr/bin/env python3

# camera.py
import cv2
import numpy as np
import rospy
import threading
import time
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from std_msgs.msg import Float64, Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from lane_detection import fit_polynomial, calculate_center_lane, draw_center_lane, MIN_CONSECUTIVE_WINDOWS
from image_processing import draw_scale
from yolopv2 import YOLOPv2
from utils_functions import calculate_ym_per_pix, calculate_xm_per_pix, calculate_curvature, validate_lane_lines
from utils.utils import lane_line_mask, thin_lane_line_mask
from bev_transform import BEVTransform

class Camera:

    def __init__(self, debug=False):
        self.vehicle_length = 4.635  # 차량 길이 (미터, 아이오닉5)
        self.vehicle_rear_offset = 1.0  # base_link가 뒷바퀴 축 중심이므로 뒤쪽 오프셋은 1m
        self.vehicle_front_offset = self.vehicle_length - self.vehicle_rear_offset  # 앞 범퍼까지의 거리
        self.road_width = 3.7  # 도로 폭 (미터)
        self.car_width = 1.9    # 차량 폭 (아이오닉5)
        self.roi_margin = 0.5   # 도로 폭보다 약간 넓은 범위

        # 카메라의 위치와 자세 파라미터 설정 (base_link를 기준으로 업데이트)
        self.camera_params = {
            'X': 1.911,          # 카메라의 X축 위치 (전방 오프셋, 1.911 미터)
            'Y': 0.0,            # 카메라의 Y축 위치 (좌우 오프셋, 없으므로 0)
            'Z': 1.2,            # 카메라의 Z축 위치 (base_link로부터의 높이, 1.2 미터)
            'PITCH': 0,          # 카메라의 피치 각도 (0도로 가정)
            'YAW': 0.0,          # 카메라의 요 각도 (0으로 가정)
            'ROLL': 0.0,         # 카메라의 롤 각도 (0으로 가정)
            'WIDTH': 1270,       # 이미지 너비 
            'HEIGHT': 720,       # 이미지 높이
            'FOV': 63.57,        # 카메라 수직 시야각 (63.57도)
            'road_width': self.road_width,       
            'vehicle_length': self.vehicle_length 
        }

        # 캘리브레이션 데이터 (1920x1080 해상도에서 얻은 값)
        self.original_camera_matrix = np.array([
                    [1655.56329, 0.0, 1012.89128],
                    [0.0, 1667.01372, 521.642768],
                    [0.0, 0.0, 1.0]
                ])

        self.dist_coeffs = np.array([-0.36123691,  0.16505182, -0.00177654,  0.00029295, 0.52639758])  # 사용자 제공 왜곡 계수로 업데이트

        # 현재 이미지 해상도에 맞게 카메라 매트릭스 스케일링
        scale_x = self.camera_params['WIDTH'] / 1920  # 가로 스케일링 계수
        scale_y = self.camera_params['HEIGHT'] / 1080  # 세로 스케일링 계수

        self.scaled_camera_matrix = self.original_camera_matrix.copy()
        self.scaled_camera_matrix[0, 0] *= scale_x  # fx
        self.scaled_camera_matrix[0, 2] *= scale_x  # cx
        self.scaled_camera_matrix[1, 1] *= scale_y  # fy
        self.scaled_camera_matrix[1, 2] *= scale_y  # cy

        # 스케일링된 카메라 매트릭스와 왜곡 계수를 camera_params에 저장
        self.camera_params['camera_matrix'] = self.scaled_camera_matrix
        self.camera_params['dist_coeffs'] = self.dist_coeffs

        # BEVTransform 객체 생성
        self.bev_transform = BEVTransform(self.camera_params)

        # ROS 관련 설정
        self.sub = rospy.Subscriber("/gmsl_camera/dev/video1/compressed", CompressedImage, self.callback, queue_size=1)  # queue_size=1으로 변경하여 최신 프레임만 유지
        self.img2fusion = rospy.Publisher('/output_img', Image, queue_size=1)  # queue_size=1으로 변경
        self.curvature_pub = rospy.Publisher('/lane_curvature', Float64, queue_size=5)
        self.lane_info_pub = rospy.Publisher('/lane_info', Float64MultiArray, queue_size=5)
        self.lane_markers_pub = rospy.Publisher('/lane_markers', MarkerArray, queue_size=5)
        self.target_points_pub = rospy.Publisher('/target_points', MarkerArray, queue_size=5)
        self.last_target_point_pub = rospy.Publisher('/last_target_point', Marker, queue_size=5)

        # 모델 초기화
        self.weights = rospy.get_param("~weights_path", "/home/inha/casey_ws/src/Lane-Detect/data/weights/yolopv2.pt")
        use_gpu = rospy.get_param("~use_gpu", False)  
        device = 'cuda' if use_gpu else 'cpu'
        self.model = YOLOPv2(self.weights, device=device)
        self.bridge = CvBridge()

        # Lane Fitting 객체 제거 (사용되지 않음)
        # self.lane_fitting = LaneFitting()

        # 이전 차선 피팅 결과 저장
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.default_center_fit = np.array([0, 0, self.camera_params['WIDTH'] / 2])
        self.default_lane_width = 3.7

        self.debug = debug

        self.frame_count = 0

        self.translation_x = 0.0
        self.translation_y = 0.0

        rospy.loginfo(f"Camera parameters: {self.camera_params}")

        self.last_processed_time = 0.0
        self.process_interval = 0.4  # 50ms으로 변경하여 처리 속도 향상

        # 이미지 처리 스레드와 최신 프레임을 위한 공유 변수 설정
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.processing_thread.start()

        # 디버그 창을 별도의 스레드에서 처리하지 않음
        # if self.debug:
        #     self.debug_thread = threading.Thread(target=self.debug_window_thread, daemon=True)
        #     self.debug_thread.start()

    # debug_window_thread 메서드 제거
    # def debug_window_thread(self):
    #     """
    #     디버그 창을 별도의 스레드에서 실행하여 메인 스레드의 블록을 방지합니다.
    #     """
    #     while not rospy.is_shutdown():
    #         if self.debug:
    #             cv2.waitKey(1)
    #         else:
    #             time.sleep(0.1)

    def publish_lane_markers(self, left_fitx, right_fitx, center_fitx, ploty, xm_per_pix, ym_per_pix):

        marker_array = MarkerArray()

        for i in range(len(ploty)):
            y_forward = (ploty[i] * ym_per_pix)  # 차량 전방 방향 (base_link의 x축)
            x_meter_left = None
            x_meter_right = None
            x_meter_center = None

            # 왼쪽 차선 마커 (노란색)
            if left_fitx is not None:
                x_pix = left_fitx[i]
                x_meter_left = (x_pix - self.camera_params['WIDTH'] / 2) * xm_per_pix  # 좌우 거리
                translated_y_forward = -(y_forward + self.translation_x) + 2.0  # x축 오프셋 +2m
                translated_x_meter = -(x_meter_left + self.translation_y)

                left_marker = Marker()
                left_marker.header.frame_id = "ego_car"
                left_marker.header.stamp = rospy.Time.now()
                left_marker.ns = "left_lane_markers"
                left_marker.id = i
                left_marker.type = Marker.SPHERE
                left_marker.action = Marker.ADD
                left_marker.pose.position.x = translated_y_forward  # 차량 전방 방향이 x축
                left_marker.pose.position.y = translated_x_meter  # 좌우 방향이 y축
                left_marker.pose.position.z = 0.0
                left_marker.pose.orientation.w = 1.0
                left_marker.scale.x = 0.5 
                left_marker.scale.y = 0.5
                left_marker.scale.z = 0.5
                left_marker.color.a = 1.0  # 불투명도
                left_marker.color.r = 1.0  # 노란색
                left_marker.color.g = 1.0
                left_marker.color.b = 0.0
                marker_array.markers.append(left_marker)

            # 오른쪽 차선 마커 (하얀색)
            if right_fitx is not None:
                x_pix = right_fitx[i]
                x_meter_right = (x_pix - self.camera_params['WIDTH'] / 2) * xm_per_pix  # 좌우 거리
                translated_y_forward = -(y_forward + self.translation_x) + 2.0  # x축 오프셋 +2m
                translated_x_meter = -(x_meter_right + self.translation_y)

                right_marker = Marker()
                right_marker.header.frame_id = "ego_car"
                right_marker.header.stamp = rospy.Time.now()
                right_marker.ns = "right_lane_markers"
                right_marker.id = i + len(ploty) 
                right_marker.type = Marker.SPHERE
                right_marker.action = Marker.ADD
                right_marker.pose.position.x = translated_y_forward  # 차량 전방 방향이 x축
                right_marker.pose.position.y = translated_x_meter  # 좌우 방향이 y축
                right_marker.pose.position.z = 0.0
                right_marker.pose.orientation.w = 1.0
                right_marker.scale.x = 0.5 
                right_marker.scale.y = 0.5
                right_marker.scale.z = 0.5
                right_marker.color.a = 1.0  # 불투명도
                right_marker.color.r = 1.0  # 하얀색
                right_marker.color.g = 1.0
                right_marker.color.b = 1.0
                marker_array.markers.append(right_marker)

            # 중앙 차선 마커 (빨간색)
            if center_fitx is not None:
                x_pix = center_fitx[i]
                x_meter_center = (x_pix - self.camera_params['WIDTH'] / 2) * xm_per_pix 
                translated_y_forward = -(y_forward + self.translation_x) + 2.0  # x축 오프셋 +2m
                translated_x_meter = -(x_meter_center + self.translation_y)

                center_marker = Marker()
                center_marker.header.frame_id = "ego_car"
                center_marker.header.stamp = rospy.Time.now()
                center_marker.ns = "center_lane_markers"
                center_marker.id = i + 2 * len(ploty) 
                center_marker.type = Marker.SPHERE
                center_marker.action = Marker.ADD
                center_marker.pose.position.x = translated_y_forward  # 차량 전방 방향이 x축
                center_marker.pose.position.y = translated_x_meter  # 좌우 방향이 y축
                center_marker.pose.position.z = 0.0
                center_marker.pose.orientation.w = 1.0
                center_marker.scale.x = 0.5  
                center_marker.scale.y = 0.5
                center_marker.scale.z = 0.5
                center_marker.color.a = 1.0  # 불투명도
                center_marker.color.r = 1.0  # 빨간색
                center_marker.color.g = 0.0
                center_marker.color.b = 0.0
                marker_array.markers.append(center_marker)

        self.lane_markers_pub.publish(marker_array)
        rospy.logdebug("차선 마커를 퍼블리시했습니다.")

    def publish_central_path_points(self, center_fitx, ploty, xm_per_pix, ym_per_pix, num_points=5):

        if center_fitx is None or ploty is None:
            rospy.logwarn("중앙 경로 데이터가 없습니다.")
            return

        sorted_indices = np.argsort(ploty)[::-1]  # 내림차순 정렬: 멀리에서 가까운 순
        sorted_center_fitx = center_fitx[sorted_indices]
        sorted_ploty = ploty[sorted_indices]

        # 중앙 경로가 충분히 길지 않을 경우
        if len(sorted_ploty) < num_points:
            rospy.logwarn("중앙 경로 데이터가 충분하지 않습니다.")
            return

        # 각 포인트의 목표 y 픽셀 값 (등간격)
        target_ys = np.linspace(sorted_ploty[0], sorted_ploty[-1], num_points)
        target_markers = []
        last_marker = None

        # 가장 먼 포인트의 좌표를 저장
        farthest_y_pix = sorted_ploty[0]
        farthest_x_pix = sorted_center_fitx[0]
        farthest_y_forward = farthest_y_pix * ym_per_pix
        farthest_x_meter = (farthest_x_pix - self.camera_params['WIDTH'] / 2) * xm_per_pix

        # 평행 이동 벡터 계산 (farthest point를 (0,0,0)으로 이동)
        self.translation_x = -farthest_y_forward
        self.translation_y = -farthest_x_meter

        rospy.logdebug(f"Farthest Point before translation: x={farthest_y_forward}, y={farthest_x_meter}")
        rospy.logdebug(f"Translation Vector: x={self.translation_x}, y={self.translation_y}")

        for i, y_pix in enumerate(target_ys):
            # 해당 y_pix에 가장 가까운 인덱스 찾기
            idx = np.argmin(np.abs(sorted_ploty - y_pix))
            x_pix = sorted_center_fitx[idx]
            y_forward = sorted_ploty[idx] * ym_per_pix  # 차량 전방 방향이 x축
            x_meter = (x_pix - self.camera_params['WIDTH'] / 2) * xm_per_pix  # 좌우 거리

            translated_y_forward = -(y_forward + self.translation_x) + 2.0  # x축 오프셋 +2m
            translated_x_meter = -(x_meter + self.translation_y)

            # Marker 메시지 생성
            marker = Marker()
            marker.header.frame_id = "ego_car"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "target_points"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = translated_y_forward  # 차량 전방 방향이 x축
            marker.pose.position.y = translated_x_meter  # 좌우 방향이 y축
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.5  
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.a = 1.0  # 불투명도

            if i == num_points - 1:
                # 가장 먼 점 (last_target_point)
                marker.ns = "last_target_point"
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                last_marker = marker
                rospy.logdebug(f"Last Target Point after translation and rotation: x={marker.pose.position.x}, y={marker.pose.position.y}")
            elif i == 0:
                # 가장 가까운 점 - (0,0,0)
                marker.pose.position.x = 0.0
                marker.pose.position.y = 0.0
                marker.pose.position.z = 0.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                rospy.logdebug(f"Closest Target Point (Origin): x={marker.pose.position.x}, y={marker.pose.position.y}")
            else:
                # 중간 포인트는 초록색
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                rospy.logdebug(f"Target Path Marker {i}: x={marker.pose.position.x}, y={marker.pose.position.y}")

            target_markers.append(marker)

        # Publish target_points excluding last_target_point
        target_points_markers = [m for m in target_markers if m.ns == "target_points" and m.id != (num_points -1)]
        target_marker_array = MarkerArray(markers=target_points_markers)
        self.target_points_pub.publish(target_marker_array)
        rospy.logdebug("Target points published.")

        # Publish last_target_point separately
        if last_marker is not None:
            self.last_target_point_pub.publish(last_marker)
            rospy.logdebug("Last target point published.")

    def process_frames(self):
        while not rospy.is_shutdown():
            with self.frame_lock:
                if self.latest_frame is None:
                    continue
                frame_resized = self.latest_frame
                self.latest_frame = None  # 프레임 처리 후 초기화

            self.process_image(frame_resized)

    def process_image(self, img0):
        start_time = time.time()
        try:
            rospy.logdebug("새로운 이미지 프레임을 처리 중입니다.")
            # 카메라 왜곡 보정
            undistorted_img = cv2.undistort(
                img0,
                self.camera_params['camera_matrix'],
                self.dist_coeffs
            )

            # Y축 픽셀당 실제 거리 계산
            ym_per_pix = calculate_ym_per_pix(
                image_height=self.camera_params['HEIGHT'],
                real_world_distance=5.0  # 실제 측정값으로 업데이트하세요
            )

            # X축 픽셀당 실제 거리 계산
            xm_per_pix = calculate_xm_per_pix(
                road_width=self.camera_params['road_width'],
                image_width=550
            )

            rospy.logdebug(f"ym_per_pix: {ym_per_pix:.6f} m/pixel, xm_per_pix: {xm_per_pix:.6f} m/pixel")

            # 1. 원본 영상 BEV 변환 및 스케일 추가
            bev_original = self.bev_transform.warp_bev_img(undistorted_img)

            # 스케일을 BEV 원본 이미지에 추가
            bev_original_with_scale = draw_scale(bev_original, xm_per_pix, ym_per_pix)

            # 2. BEV 원본 이미지에 스케일을 추가하여 출력 (디버깅 모드일 경우)
            if self.debug:
                cv2.imshow("BEV Original Image with Scale", bev_original_with_scale)
                cv2.waitKey(1)  # waitKey 추가

            # YOLOPv2 모델을 사용한 차선 감지 및 세그멘테이션 (왜곡 보정된 이미지 사용)
            pred, seg, ll = self.model.detect(undistorted_img)
            ll_seg_mask = lane_line_mask(ll)

            # 3. 차선 세그멘테이션 영상 출력 (디버깅 모드일 경우)
            if self.debug:
                cv2.imshow("LL Segmentation", ll_seg_mask)
                cv2.waitKey(1)  # waitKey 추가

            # 얇은 차선 마스크 적용
            ll_seg_mask = thin_lane_line_mask(ll_seg_mask)

            # BEV 변환 적용
            bev_ll_seg_mask = self.bev_transform.warp_bev_img(ll_seg_mask)

            # ploty 배열 생성
            ploty = np.linspace(0, bev_ll_seg_mask.shape[0] - 1, bev_ll_seg_mask.shape[0])

            # 4. 수정된 draw_scale 함수 호출 (픽셀당 실제 거리 값 전달)
            bev_ll_seg_mask_with_scale = draw_scale(bev_ll_seg_mask, xm_per_pix, ym_per_pix)

            # 5. BEV 차선 세그멘테이션 영상 출력 (디버깅 모드일 경우)
            if self.debug:
                cv2.imshow("BEV LL Segmentation with Scale", bev_ll_seg_mask_with_scale)
                cv2.waitKey(1)  # waitKey 추가

            # 슬라이딩 윈도우 및 차선 피팅
            updated_left_fit, updated_right_fit, left_fitx, right_fitx, out_img = fit_polynomial(
                bev_ll_seg_mask,
                ploty=ploty,  # ploty 전달
                visualization=self.debug,
                prev_left_fit=self.prev_left_fit,
                prev_right_fit=self.prev_right_fit,
                xm_per_pix=xm_per_pix,
                ym_per_pix=ym_per_pix,
                min_consecutive_windows=MIN_CONSECUTIVE_WINDOWS
            )

            # 차선 폭 검증 및 조정 로직 추가
            if updated_left_fit is not None and updated_right_fit is not None:
                # 두 차선 간의 평균 폭 계산
                lane_width = np.mean(np.abs(np.polyval(updated_right_fit, ploty) - np.polyval(updated_left_fit, ploty)) * xm_per_pix)
                rospy.loginfo(f"Detected lane width: {lane_width:.2f} meters")

                if lane_width > 4.0:
                    # 더 가까운 차선 결정 (y축 기준, 낮은 y가 더 가까운)
                    left_y = np.mean(np.polyval(updated_left_fit, ploty))
                    right_y = np.mean(np.polyval(updated_right_fit, ploty))

                    if left_y < right_y:
                        # 왼쪽 차선이 더 가까움
                        rospy.logwarn("Lane width exceeds 4m. Trusting left lane and adjusting right lane.")
                        # 오른쪽 차선을 3.7m 간격으로 조정
                        # x_meter_shift는 3.7m를 픽셀로 변환
                        x_meter_shift = 3.7
                        x_pix_shift = x_meter_shift / xm_per_pix
                        # 오른쪽 차선을 왼쪽 차선으로부터 3.7m 떨어진 위치로 이동
                        shifted_right_fitx = np.polyval(updated_left_fit, ploty) + x_pix_shift
                        # 새 다항식 피팅
                        updated_right_fit = np.polyfit(ploty, shifted_right_fitx, 2)
                    else:
                        # 오른쪽 차선이 더 가까움
                        rospy.logwarn("Lane width exceeds 4m. Trusting right lane and adjusting left lane.")
                        # 왼쪽 차선을 3.7m 간격으로 조정
                        x_meter_shift = 3.7
                        x_pix_shift = x_meter_shift / xm_per_pix
                        # 왼쪽 차선을 오른쪽 차선으로부터 3.7m 떨어진 위치로 이동
                        shifted_left_fitx = np.polyval(updated_right_fit, ploty) - x_pix_shift
                        # 새 다항식 피팅
                        updated_left_fit = np.polyfit(ploty, shifted_left_fitx, 2)

            # 유효성 검증 및 차선 폭 조정 후 다시 차선 피팅
            if validate_lane_lines(updated_left_fit, updated_right_fit, ploty, expected_width=self.default_lane_width, xm_per_pix=xm_per_pix):
                # 업데이트된 피팅 결과 저장
                self.prev_left_fit = updated_left_fit
                self.prev_right_fit = updated_right_fit
                rospy.logdebug("차선 피팅 결과가 유효합니다.")
            else:
                rospy.logwarn("유효하지 않은 차선 피팅 결과입니다. 이전 피팅 결과를 유지합니다.")

            # 6. 슬라이딩 윈도우 시각화 출력 (디버깅 모드일 경우)
            if self.debug:
                cv2.imshow("Sliding Window Visualization", out_img)
                cv2.waitKey(1)  # waitKey 추가

            # 왼쪽과 오른쪽 차선의 x좌표를 이용하여 중앙 차선 계산
            if self.prev_left_fit is not None and self.prev_right_fit is not None:
                center_fitx = calculate_center_lane(
                    np.polyval(self.prev_left_fit, ploty),
                    np.polyval(self.prev_right_fit, ploty),
                    ploty,
                    self.default_center_fit[2]
                )
            else:
                center_fitx = None

            if center_fitx is None:
                rospy.logwarn("center_fitx가 None입니다. 추가 처리를 건너뜁니다.")
                return

            # 중앙 차선의 다항식 계수 계산
            center_fit = np.polyfit(ploty, center_fitx, 2)

            # 도로 곡률 계산 및 퍼블리시
            curvature = calculate_curvature(center_fit, ploty[-1], xm_per_pix, ym_per_pix)
            if curvature is not None:
                if curvature > 1000.0:  # 임계값 설정 (예: 1000m 이상은 직선으로 간주)
                    curvature = 0.0
                    rospy.loginfo("Straight road detected. Setting curvature to 0.")
                rospy.logdebug(f"계산된 곡률: {curvature:.2f} m.")
                self.curvature_pub.publish(curvature)
                rospy.loginfo(f"곡률을 퍼블리시했습니다: {curvature:.2f} m")
            else:
                rospy.logwarn("곡률 계산에 실패했습니다.")

            # 차선 정보 퍼블리시 (left, right, center fit 계수)
            lane_info = Float64MultiArray()
            if self.prev_left_fit is not None and self.prev_right_fit is not None:
                lane_info.data = np.concatenate([self.prev_left_fit, self.prev_right_fit, center_fit]).tolist()
            elif self.prev_left_fit is not None and self.prev_right_fit is None:
                lane_info.data = np.concatenate([self.prev_left_fit, self.default_center_fit, center_fit]).tolist()
            elif self.prev_right_fit is not None and self.prev_left_fit is None:
                lane_info.data = np.concatenate([self.default_center_fit, self.prev_right_fit, center_fit]).tolist()
            else:
                lane_info.data = np.concatenate([self.default_center_fit, self.default_center_fit, center_fit]).tolist()
            
            # 배열의 형태를 확인하여 오류를 방지
            try:
                rospy.logdebug(f"lane_info shapes: left_fit {self.prev_left_fit.shape if self.prev_left_fit is not None else None}, "
                              f"right_fit {self.prev_right_fit.shape if self.prev_right_fit is not None else None}, "
                              f"center_fit {center_fit.shape}")
                self.lane_info_pub.publish(lane_info)
                rospy.logdebug("차선 정보를 퍼블리시했습니다.")
            except ValueError as ve:
                rospy.logerr(f"차선 정보 퍼블리시에 실패했습니다: {ve}")
            except Exception as e:
                rospy.logerr(f"차선 정보 퍼블리시에 오류 발생: {e}")

            # Lane fitting 시각화
            bev_visual = cv2.cvtColor(bev_ll_seg_mask, cv2.COLOR_GRAY2BGR)
            if self.prev_left_fit is not None:
                pts_left = np.array([np.transpose(np.vstack([np.polyval(self.prev_left_fit, ploty), ploty]))])
                # 좌표 평행 이동, 180도 회전, x축 오프셋 적용
                pts_left_translated = pts_left + np.array([self.translation_x / ym_per_pix, self.translation_y / xm_per_pix])
                pts_left_translated = -pts_left_translated  # 180도 회전
                cv2.polylines(bev_visual, np.int32([pts_left_translated]), isClosed=False, color=(255, 0, 0), thickness=5)
            if self.prev_right_fit is not None:
                pts_right = np.array([np.transpose(np.vstack([np.polyval(self.prev_right_fit, ploty), ploty]))])
                pts_right_translated = pts_right + np.array([self.translation_x / ym_per_pix, self.translation_y / xm_per_pix])
                pts_right_translated = -pts_right_translated  # 180도 회전
                cv2.polylines(bev_visual, np.int32([pts_right_translated]), isClosed=False, color=(0, 0, 255), thickness=5)

            # 7. Lane fitting 영상 출력 (디버깅 모드일 경우)
            if self.debug:
                cv2.imshow("Lane Fitting", bev_visual)
                cv2.waitKey(1)  # waitKey 추가

            # 중앙 차선 그리기 (평행 이동 및 180도 회전 적용)
            bev_visual = draw_center_lane(bev_visual, center_fitx + (self.translation_y / xm_per_pix), ploty + (self.translation_x / ym_per_pix))
            bev_visual = cv2.flip(bev_visual, -1)  # 180도 회전 (x와 y 모두 반전)

            # 8. Center line 영상 출력 (디버깅 모드일 경우)
            if self.debug:
                cv2.imshow("Center Line", bev_visual)
                cv2.waitKey(1)  # waitKey 추가

            # 중앙 경로 곡선을 5등분하여 점 찍기 및 last_target_point 퍼블리시
            self.publish_central_path_points(center_fitx, ploty, xm_per_pix, ym_per_pix, num_points=5)

            # 차선 마커 퍼블리시
            self.publish_lane_markers(left_fitx, right_fitx, center_fitx, ploty, xm_per_pix, ym_per_pix)
            rospy.logdebug("차선 마커를 퍼블리시했습니다.")

            # 원본 이미지에 차선 오버레이
            lanes_bev = np.zeros((bev_ll_seg_mask.shape[0], bev_ll_seg_mask.shape[1], 3), dtype=np.uint8)
            if left_fitx is not None:
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                pts_left_translated = pts_left + np.array([self.translation_x / ym_per_pix, self.translation_y / xm_per_pix])
                pts_left_translated = -pts_left_translated
                cv2.polylines(lanes_bev, np.int32([pts_left_translated]), isClosed=False, color=(255, 0, 0), thickness=5)
            if right_fitx is not None:
                pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
                pts_right_translated = pts_right + np.array([self.translation_x / ym_per_pix, self.translation_y / xm_per_pix])
                pts_right_translated = -pts_right_translated
                cv2.polylines(lanes_bev, np.int32([pts_right_translated]), isClosed=False, color=(0, 0, 255), thickness=5)
            if center_fitx is not None:
                pts_center = np.array([np.transpose(np.vstack([center_fitx, ploty]))])
                pts_center_translated = pts_center + np.array([self.translation_x / ym_per_pix, self.translation_y / xm_per_pix])
                pts_center_translated = -pts_center_translated
                cv2.polylines(lanes_bev, np.int32([pts_center_translated]), isClosed=False, color=(0, 255, 0), thickness=5)  # 초록색으로 중앙 차선 표시

            if np.count_nonzero(lanes_bev) > 0:
                lanes_original = self.bev_transform.warp_inv_img(lanes_bev)

                if len(lanes_original.shape) == 2 or lanes_original.shape[2] == 1:
                    lanes_original = cv2.cvtColor(lanes_original, cv2.COLOR_GRAY2BGR)

                # 180도 회전 및 x축 오프셋 적용
                lanes_original = cv2.flip(lanes_original, -1)
                overlay_img = cv2.addWeighted(src1=undistorted_img, alpha=1, src2=lanes_original, beta=0.5, gamma=0)
            else:
                overlay_img = undistorted_img.copy()

            # 9. 원본 영상에 투영된 결과 영상 출력 (디버깅 모드일 경우)
            if self.debug:
                cv2.imshow("Inverse BEV with Lane Overlay", overlay_img)
                cv2.waitKey(1)  # waitKey 추가

            # 원본 이미지에 투영된 결과 이미지를 퍼블리시
            try:
                output_msg = self.bridge.cv2_to_imgmsg(overlay_img, encoding="bgr8")
                self.img2fusion.publish(output_msg)
                rospy.logdebug("output_img 퍼블리시 완료.")
            except cv2.error as e:
                rospy.logerr(f"OpenCV 변환 오류: {e}")
            except Exception as e:
                rospy.logerr(f"오버레이 이미지를 퍼블리시하지 못했습니다: {e}")

            end_time = time.time()
            processing_time = end_time - start_time
            rospy.logdebug(f"이미지 프레임이 성공적으로 처리되었습니다. 처리 시간: {processing_time:.4f} 초")
        
        except cv2.error as e:
            rospy.logerr(f"OpenCV 오류 발생: {e}")
        except Exception as e:
            rospy.logerr(f"process_image에서 오류 발생: {e}")

    def callback(self, msg):
        """
        ROS에서 이미지를 수신하는 콜백 함수입니다.

        Args:
            msg (CompressedImage): 수신된 이미지 메시지.
        """
        try:
            current_time = time.time()
            if (current_time - self.last_processed_time) < self.process_interval:
                rospy.logdebug("프레임 처리 간격이 너무 짧아 프레임을 건너뜁니다.")
                return  # 처리 간격이 짧으면 프레임을 무시

            self.last_processed_time = current_time

            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # 모델의 추론 속도를 고려하여 이미지 크기 축소 (1270x720 유지)
            frame_resized = cv2.resize(frame, (self.camera_params['WIDTH'], self.camera_params['HEIGHT']), interpolation=cv2.INTER_LINEAR)
            
            # 큐 대신 최신 프레임을 공유 변수에 저장
            with self.frame_lock:
                self.latest_frame = frame_resized

            self.frame_count += 1
            if self.frame_count % 100 == 0:
                rospy.loginfo(f"[INFO] 수신 및 크기 조정된 이미지 형태: {frame_resized.shape}")
        except cv2.error as e:
            rospy.logerr(f"OpenCV 오류: 수신된 이미지를 처리하지 못했습니다: {e}")
        except Exception as e:
            rospy.logerr(f"[ERROR] 수신된 이미지를 처리하지 못했습니다: {e}")

    def shutdown(self):
        """
        노드 종료 시 호출되는 메서드입니다.
        """
        if self.debug:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        rospy.init_node('camera_node', anonymous=True)
        # 디버깅 모드 활성화 시 OpenCV 창 표시, 기본값은 False로 설정
        debug_mode = rospy.get_param("~debug", False)
        camera = Camera(debug=debug_mode)
        rospy.on_shutdown(camera.shutdown)  # 노드 종료 시 shutdown 메서드 호출
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
