# utils_functions.py

import numpy as np
import rospy
import math

def calculate_ym_per_pix(image_height, real_world_distance):
    """
    이미지의 높이와 실제 거리로부터 Y축 픽셀당 실제 거리 계산

    Args:
        image_height (int): 이미지 높이 (픽셀)
        real_world_distance (float): 이미지 하단에 해당하는 실제 거리 (미터)

    Returns:
        ym_per_pix (float): Y축 픽셀당 실제 거리 (미터/픽셀)
    """
    ym_per_pix = real_world_distance / image_height
    rospy.loginfo(f"Calculated ym_per_pix: {ym_per_pix:.6f} m/pixel")
    return ym_per_pix


def calculate_xm_per_pix(road_width, image_width):
    """
    X축 픽셀당 실제 거리 (미터/픽셀) 계산

    Args:
        road_width (float): 도로 폭 (미터)
        image_width (int): 이미지의 너비 (픽셀)

    Returns:
        float: X축 픽셀당 실제 거리 (미터/픽셀)
    """
    if image_width == 0:
        rospy.logwarn("Image width is zero. Using default xm_per_pix = 3.7/573.")
        return 3.7 / 573  # 기본값
    xm_per_pix = road_width / image_width
    rospy.logdebug(f"Calculated xm_per_pix: {xm_per_pix:.6f} m/pixel")
    return xm_per_pix

def calculate_curvature(polynomial, y_eval, xm_per_pix, ym_per_pix):
    """
    곡률 계산

    Args:
        polynomial (numpy.ndarray): 다항식 계수 (예: [A, B, C] for Ax^2 + Bx + C)
        y_eval (float): 곡률을 계산할 y 위치 (미터 단위)
        xm_per_pix (float): X축 픽셀당 실제 거리 (미터/픽셀)
        ym_per_pix (float): Y축 픽셀당 실제 거리 (미터/픽셀)

    Returns:
        float: 계산된 곡률 (미터 단위) 또는 None
    """
    if polynomial is None or len(polynomial) < 2:
        rospy.logwarn("곡률 계산 불가: 다항식 계수가 부족하거나 None.")
        return None

    # 실제 거리 단위로 변환된 계수
    A = polynomial[0] * xm_per_pix / (ym_per_pix ** 2)
    B = polynomial[1] * xm_per_pix / ym_per_pix
    y_eval_m = y_eval * ym_per_pix

    # 곡률 계산 공식
    curvature = ((1 + (2 * A * y_eval_m + B) ** 2) ** 1.5) / np.abs(2 * A)
    rospy.logdebug(f"Calculated curvature: {curvature:.2f} m at y = {y_eval_m:.2f} m")
    return curvature

def validate_lane_lines(left_fit, right_fit, ploty, expected_width=3.7, xm_per_pix=3.7/573, tolerance=0.5):
    """
    좌우 차선의 유효성을 검증합니다.

    Args:
        left_fit (numpy.ndarray): 왼쪽 차선의 다항식 계수.
        right_fit (numpy.ndarray): 오른쪽 차선의 다항식 계수.
        ploty (numpy.ndarray): y 좌표 배열.
        expected_width (float, optional): 기대되는 차선 폭 (미터). 기본값은 3.7m.
        xm_per_pix (float, optional): X축 픽셀당 실제 거리 (미터/픽셀). 기본값은 3.7/573.
        tolerance (float, optional): 허용 오차 (미터). 기본값은 0.5m.

    Returns:
        bool: 차선 유효 여부.
    """
    if left_fit is None or right_fit is None:
        rospy.logwarn("One or both lane fits are None.")
        return False

    # 좌우 차선의 x 좌표 계산
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)

    # 차선 폭 계산
    lane_width_pixels = np.abs(right_fitx - left_fitx)
    lane_width_meters = lane_width_pixels * xm_per_pix

    # 차선 폭의 평균과 표준편차 계산
    mean_lane_width = np.mean(lane_width_meters)
    std_lane_width = np.std(lane_width_meters)

    rospy.loginfo(f"Calculated lane width: {mean_lane_width:.2f} meters (Expected: {expected_width} meters)")

    # 차선 폭이 기대 범위 내에 있는지 확인
    if abs(mean_lane_width - expected_width) > tolerance:
        rospy.logwarn(f"Lane width deviation: {abs(mean_lane_width - expected_width):.2f} meters exceeds tolerance.")
        return False

    # 차선 폭의 표준편차가 너무 큰 경우
    if std_lane_width > tolerance:
        rospy.logwarn(f"Lane width standard deviation: {std_lane_width:.2f} meters exceeds tolerance.")
        return False

    return True
