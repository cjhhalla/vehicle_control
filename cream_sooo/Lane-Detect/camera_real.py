#!/usr/bin/env python3

# camera.py
import cv2
import numpy as np
import rospy
import threading
import queue
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from std_msgs.msg import Float64, Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from lane_detection import fit_polynomial, calculate_center_lane, draw_center_lane, MIN_CONSECUTIVE_WINDOWS
from image_processing import draw_scale
from kalman_filter import LaneFitting
from yolopv2 import YOLOPv2
from utils_functions import calculate_ym_per_pix, calculate_xm_per_pix, calculate_curvature
from utils.utils import lane_line_mask, thin_lane_line_mask
from bev_transform import BEVTransform

class Camera:
    """
    ROS 노드로서 카메라 이미지를 구독하고 차선을 감지하여 다양한 정보를 퍼블리시합니다.
    """
    def __init__(self, debug=False):
        # 차량 및 도로 파라미터 설정
        self.vehicle_length = 4.635
        self.vehicle_rear_offset = 0.0
        self.vehicle_front_offset = self.vehicle_length - self.vehicle_rear_offset
        self.road_width = 3.7
        self.car_width = 1.9
        self.roi_margin = 0.5

        # 해상도 원래대로 유지 (1280x720)
        self.camera_params = {
            'X': 1.911,
            'Y': 0.0,
            'Z': 1.2,
            'PITCH': np.deg2rad(-3.0),
            'YAW': 0.0,
            'ROLL': 0.0,
            'WIDTH': 1280,
            'HEIGHT': 720,
            'FOV': 90,
            'road_width': self.road_width,
            'vehicle_length': self.vehicle_length,
            'camera_matrix': np.array([
                [1655.56329,    0.0,      1012.89128],
                [   0.0,     1667.01372,   521.642768],
                [   0.0,        0.0,           1.0   ]
            ]),
            'dist_coeffs': np.array([-0.22874612, -0.45882556, 0.01423776, -0.00239172, 0.52639758])
        }

        # BEVTransform 객체 생성
        self.bev_transform = BEVTransform(self.camera_params)

        # ROS 관련 설정
        self.sub = rospy.Subscriber("/gmsl_camera/dev/video1/compressed", CompressedImage, self.callback, queue_size=10)
        self.img2fusion = rospy.Publisher('/output_img', Image, queue_size=10)
        self.curvature_pub = rospy.Publisher('/lane_curvature', Float64, queue_size=10)
        self.lane_info_pub = rospy.Publisher('/lane_info', Float64MultiArray, queue_size=10)
        self.lane_markers_pub = rospy.Publisher('/lane_markers', MarkerArray, queue_size=10)
        self.target_points_pub = rospy.Publisher('/target_points', MarkerArray, queue_size=10)
        self.last_target_point_pub = rospy.Publisher('/last_target_point', Marker, queue_size=10)

        # YOLOPv2 모델 초기화
        self.weights = '/home/inha/casey_ws/src/Lane-Detect/data/weights/yolopv2.pt'
        self.model = YOLOPv2(self.weights)
        self.bridge = CvBridge()

        # Lane Fitting 객체 초기화
        self.lane_fitting = LaneFitting()

        # 이전 차선 피팅 결과
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.default_center_fit = np.array([0, 0, self.camera_params['WIDTH'] / 2])

        self.default_lane_width = 3.7

        # 디버깅 모드
        self.debug = False  # 디버그 비활성화

        # 프레임 카운트
        self.frame_count = 0

        # 평행 이동 벡터
        self.translation_x = 0.0
        self.translation_y = 0.0

        rospy.loginfo(f"Camera parameters: {self.camera_params}")

        # 이미지 큐 및 처리 스레드
        self.frame_queue = queue.Queue(maxsize=10)
        self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.processing_thread.start()

        # YOLO 결과 재사용을 위한 변수
        self.last_ll = None

    def publish_lane_markers(self, left_fitx, right_fitx, center_fitx, ploty, xm_per_pix, ym_per_pix):
        marker_array = MarkerArray()

        for i in range(len(ploty)):
            y_forward = (ploty[i] * ym_per_pix)
            # 왼쪽 차선
            if left_fitx is not None:
                x_pix = left_fitx[i]
                x_meter_left = (x_pix - self.camera_params['WIDTH'] / 2) * xm_per_pix
                translated_y_forward = -(y_forward + self.translation_x) + 4.0
                translated_x_meter = -(x_meter_left + self.translation_y)

                left_marker = Marker()
                left_marker.header.frame_id = "base_link"
                left_marker.header.stamp = rospy.Time.now()
                left_marker.ns = "left_lane_markers"
                left_marker.id = i
                left_marker.type = Marker.SPHERE
                left_marker.action = Marker.ADD
                left_marker.pose.position.x = translated_y_forward
                left_marker.pose.position.y = translated_x_meter
                left_marker.pose.position.z = 0.0
                left_marker.pose.orientation.w = 1.0
                left_marker.scale.x = 0.1
                left_marker.scale.y = 0.1
                left_marker.scale.z = 0.1
                left_marker.color.a = 1.0
                left_marker.color.r = 0.0
                left_marker.color.g = 0.0
                left_marker.color.b = 1.0
                marker_array.markers.append(left_marker)

            # 오른쪽 차선
            if right_fitx is not None:
                x_pix = right_fitx[i]
                x_meter_right = (x_pix - self.camera_params['WIDTH'] / 2) * xm_per_pix
                translated_y_forward = -(y_forward + self.translation_x) + 4.0
                translated_x_meter = -(x_meter_right + self.translation_y)

                right_marker = Marker()
                right_marker.header.frame_id = "base_link"
                right_marker.header.stamp = rospy.Time.now()
                right_marker.ns = "right_lane_markers"
                right_marker.id = i + len(ploty)
                right_marker.type = Marker.SPHERE
                right_marker.action = Marker.ADD
                right_marker.pose.position.x = translated_y_forward
                right_marker.pose.position.y = translated_x_meter
                right_marker.pose.position.z = 0.0
                right_marker.pose.orientation.w = 1.0
                right_marker.scale.x = 0.1
                right_marker.scale.y = 0.1
                right_marker.scale.z = 0.1
                right_marker.color.a = 1.0
                right_marker.color.r = 1.0
                right_marker.color.g = 0.0
                right_marker.color.b = 0.0
                marker_array.markers.append(right_marker)

            # 중앙 차선
            if center_fitx is not None:
                x_pix = center_fitx[i]
                x_meter_center = (x_pix - self.camera_params['WIDTH'] / 2) * xm_per_pix
                translated_y_forward = -(y_forward + self.translation_x) + 4.0
                translated_x_meter = -(x_meter_center + self.translation_y)

                center_marker = Marker()
                center_marker.header.frame_id = "base_link"
                center_marker.header.stamp = rospy.Time.now()
                center_marker.ns = "center_lane_markers"
                center_marker.id = i + 2 * len(ploty)
                center_marker.type = Marker.SPHERE
                center_marker.action = Marker.ADD
                center_marker.pose.position.x = translated_y_forward
                center_marker.pose.position.y = translated_x_meter
                center_marker.pose.position.z = 0.0
                center_marker.pose.orientation.w = 1.0
                center_marker.scale.x = 0.1
                center_marker.scale.y = 0.1
                center_marker.scale.z = 0.1
                center_marker.color.a = 1.0
                center_marker.color.r = 0.0
                center_marker.color.g = 1.0
                center_marker.color.b = 0.0
                marker_array.markers.append(center_marker)

        self.lane_markers_pub.publish(marker_array)

    def publish_central_path_points(self, center_fitx, ploty, xm_per_pix, ym_per_pix, num_points=5):
        if center_fitx is None or ploty is None:
            return

        sorted_indices = np.argsort(ploty)[::-1]
        sorted_center_fitx = center_fitx[sorted_indices]
        sorted_ploty = ploty[sorted_indices]

        if len(sorted_ploty) < num_points:
            return

        target_ys = np.linspace(sorted_ploty[0], sorted_ploty[-1], num_points)

        target_markers = []
        last_marker = None

        farthest_y_pix = sorted_ploty[0]
        farthest_x_pix = sorted_center_fitx[0]
        farthest_y_forward = farthest_y_pix * ym_per_pix
        farthest_x_meter = (farthest_x_pix - self.camera_params['WIDTH'] / 2) * xm_per_pix

        self.translation_x = -farthest_y_forward
        self.translation_y = -farthest_x_meter

        for i, y_pix in enumerate(target_ys):
            idx = np.argmin(np.abs(sorted_ploty - y_pix))
            x_pix = sorted_center_fitx[idx]
            y_forward = sorted_ploty[idx] * ym_per_pix
            x_meter = (x_pix - self.camera_params['WIDTH'] / 2) * xm_per_pix

            translated_y_forward = -(y_forward + self.translation_x) + 4.0
            translated_x_meter = -(x_meter + self.translation_y)

            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "target_points"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = translated_y_forward
            marker.pose.position.y = translated_x_meter
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0

            if i == num_points - 1:
                marker.ns = "last_target_point"
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                last_marker = marker
            elif i == 0:
                # 가장 가까운 점 원점으로
                marker.pose.position.x = 0.0
                marker.pose.position.y = 0.0
                marker.pose.position.z = 0.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0

            target_markers.append(marker)

        # 마지막 점 제외한 타겟 포인트 퍼블리시
        target_points_markers = [m for m in target_markers if m.ns == "target_points" and m.id != (num_points -1)]
        target_marker_array = MarkerArray(markers=target_points_markers)
        self.target_points_pub.publish(target_marker_array)

        # last_target_point 퍼블리시
        if last_marker is not None:
            self.last_target_point_pub.publish(last_marker)

    def process_image(self, img0):
        try:
            undistorted_img = cv2.undistort(img0, self.camera_params['camera_matrix'], self.camera_params['dist_coeffs'])

            ym_per_pix = calculate_ym_per_pix(
                camera_height=self.camera_params['Z'] + 0.6,
                fov=self.camera_params['FOV'],
                image_height=self.camera_params['HEIGHT'],
                pitch_angle=np.rad2deg(self.camera_params['PITCH'])
            )
            xm_per_pix = calculate_xm_per_pix(
                road_width=self.camera_params['road_width'],
                image_width=self.camera_params['WIDTH']
            )

            bev_original = self.bev_transform.warp_bev_img(undistorted_img)

            # YOLO 추론 빈도 낮추기 (예: 3프레임에 1번)
            if self.frame_count % 3 == 0:
                pred, seg, ll = self.model.detect(undistorted_img)
                self.last_ll = ll
            else:
                ll = self.last_ll

            if self.last_ll is None:
                return

            ll_seg_mask = lane_line_mask(self.last_ll)
            if ll_seg_mask is None:
                return

            ll_seg_mask = thin_lane_line_mask(ll_seg_mask)
            bev_ll_seg_mask = self.bev_transform.warp_bev_img(ll_seg_mask)
            ploty = np.linspace(0, bev_ll_seg_mask.shape[0] - 1, bev_ll_seg_mask.shape[0])
            bev_ll_seg_mask_with_scale = draw_scale(bev_ll_seg_mask, xm_per_pix, ym_per_pix)

            updated_left_fit, updated_right_fit, left_fitx, right_fitx, out_img = fit_polynomial(
                bev_ll_seg_mask,
                ploty=ploty,
                visualization=False,
                prev_left_fit=self.prev_left_fit,
                prev_right_fit=self.prev_right_fit,
                xm_per_pix=xm_per_pix,
                ym_per_pix=ym_per_pix,
                min_consecutive_windows=MIN_CONSECUTIVE_WINDOWS
            )

            self.prev_left_fit = updated_left_fit
            self.prev_right_fit = updated_right_fit

            center_fitx = calculate_center_lane(left_fitx, right_fitx, ploty, self.default_center_fit[2])
            if center_fitx is None:
                return

            curvature = calculate_curvature(updated_left_fit, ploty[-1], xm_per_pix, ym_per_pix)
            if curvature is not None:
                if self.frame_count % 3 == 0:
                    self.curvature_pub.publish(curvature)

            lane_info = Float64MultiArray()
            if updated_left_fit is not None and updated_right_fit is not None:
                center_fit = np.polyfit(ploty, center_fitx, 2)
                lane_info.data = np.concatenate([updated_left_fit, updated_right_fit, center_fit]).tolist()
            elif updated_left_fit is not None and updated_right_fit is None:
                center_fit = np.polyfit(ploty, center_fitx, 2)
                lane_info.data = np.concatenate([updated_left_fit, self.default_center_fit, center_fit]).tolist()
            elif updated_right_fit is not None and updated_left_fit is None:
                center_fit = np.polyfit(ploty, center_fitx, 2)
                lane_info.data = np.concatenate([self.default_center_fit, updated_right_fit, center_fit]).tolist()
            else:
                center_fit = np.polyfit(ploty, center_fitx, 2)
                lane_info.data = np.concatenate([self.default_center_fit, self.default_center_fit, center_fit]).tolist()

            if self.frame_count % 3 == 0:
                self.lane_info_pub.publish(lane_info)

            bev_visual = cv2.cvtColor(bev_ll_seg_mask, cv2.COLOR_GRAY2BGR)
            bev_visual = draw_center_lane(bev_visual, center_fitx, ploty)
            bev_visual = cv2.flip(bev_visual, -1)

            # 중앙 경로 포인트 퍼블리시 (3프레임에 한번)
            if self.frame_count % 3 == 0:
                self.publish_central_path_points(center_fitx, ploty, xm_per_pix, ym_per_pix, num_points=5)

            # 차선 마커 퍼블리시 (3프레임에 한번)
            if self.frame_count % 3 == 0:
                self.publish_lane_markers(left_fitx, right_fitx, center_fitx, ploty, xm_per_pix, ym_per_pix)

            # 오버레이 이미지 생성
            lanes_bev = np.zeros((bev_ll_seg_mask.shape[0], bev_ll_seg_mask.shape[1], 3), dtype=np.uint8)
            if left_fitx is not None:
                pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
                cv2.polylines(lanes_bev, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=5)
            if right_fitx is not None:
                pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
                cv2.polylines(lanes_bev, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=5)
            if center_fitx is not None:
                pts_center = np.array([np.transpose(np.vstack([center_fitx, ploty]))])
                cv2.polylines(lanes_bev, np.int32([pts_center]), isClosed=False, color=(0, 255, 0), thickness=5)

            if np.count_nonzero(lanes_bev) > 0:
                lanes_original = self.bev_transform.warp_inv_img(lanes_bev)
                if len(lanes_original.shape) == 2 or lanes_original.shape[2] == 1:
                    lanes_original = cv2.cvtColor(lanes_original, cv2.COLOR_GRAY2BGR)
                # 180도 회전
                lanes_original = cv2.flip(lanes_original, -1)
                overlay_img = cv2.addWeighted(src1=undistorted_img, alpha=1, src2=lanes_original, beta=0.5, gamma=0)
            else:
                overlay_img = undistorted_img.copy()

            # 이미지 퍼블리시 (3프레임에 한번)
            if self.frame_count % 3 == 0:
                try:
                    output_msg = self.bridge.cv2_to_imgmsg(overlay_img, encoding="bgr8")
                    self.img2fusion.publish(output_msg)
                except Exception as e:
                    rospy.logerr(f"오버레이 이미지를 퍼블리시하지 못했습니다: {e}")

        except Exception as e:
            rospy.logerr(f"process_image에서 오류 발생: {e}")

    def callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame_resized = cv2.resize(frame, (self.camera_params['WIDTH'], self.camera_params['HEIGHT']), interpolation=cv2.INTER_LINEAR)
            
            # 큐가 꽉 찼으면 이전 프레임 버리고 새 프레임 추가
            if not self.frame_queue.full():
                self.frame_queue.put(frame_resized)
            else:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame_resized)
                except queue.Empty:
                    pass

            self.frame_count += 1
            if self.frame_count % 100 == 0:
                rospy.loginfo(f"[INFO] 수신 및 크기 조정된 이미지 형태: {frame_resized.shape}")
        except Exception as e:
            rospy.logerr(f"[ERROR] 수신된 이미지를 처리하지 못했습니다: {e}")

    def process_frames(self):
        while not rospy.is_shutdown():
            try:
                frame_resized = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            self.process_image(frame_resized)
            self.frame_queue.task_done()

    def shutdown(self):
        if self.debug:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        rospy.init_node('camera_node', anonymous=True)
        camera = Camera(debug=False)
        rospy.spin()
    except rospy.ROSInterruptException:
        if camera.debug:
            cv2.destroyAllWindows()
        pass