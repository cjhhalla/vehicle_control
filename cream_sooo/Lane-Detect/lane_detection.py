# lane_detection.py

import numpy as np
import cv2
import rospy

MIN_CONSECUTIVE_WINDOWS = 4  # 최소 연속 윈도우 수

def find_lane_pixels(binary_warped, visualization=False, margin=50, minpix=30, min_consecutive_windows=4):
    """
    슬라이딩 윈도우 방법을 사용하여 차선 픽셀을 탐지하고,
    일정 길이 이상으로 검출되지 않은 차선은 무시합니다.

    Args:
        binary_warped (numpy.ndarray): 이진화된 원근 변환 이미지.
        visualization (bool, optional): 시각화 여부. 기본값은 False.
        margin (int, optional): 윈도우 마진. 기본값은 50.
        minpix (int, optional): 최소 픽셀 수. 기본값은 30.
        min_consecutive_windows (int, optional): 최소 연속 윈도우 수. 기본값은 4.

    Returns:
        tuple: (leftx, lefty, rightx, righty, out_img, left_valid, right_valid)
    """
    rospy.logdebug("Starting to find lane pixels.")
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    out_img = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)

    midpoint = int(histogram.shape[0]//2)
    leftx_base = int(np.argmax(histogram[:midpoint]))
    rightx_base = int(np.argmax(histogram[midpoint:])) + midpoint

    nwindows = 9
    window_height = int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    window_img = np.zeros_like(out_img)

    # 연속된 윈도우 카운터 초기화
    left_consecutive = 0
    right_consecutive = 0

    left_valid = False
    right_valid = False

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if visualization:
            # 왼쪽 및 오른쪽 윈도우 그리기 (녹색)
            cv2.rectangle(window_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(window_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # 윈도우 내의 픽셀 인덱스 찾기
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # 윈도우 내의 픽셀 인덱스 저장
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 피킹된 픽셀 수에 따라 다음 윈도우의 중심점 업데이트
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
            left_consecutive += 1
        else:
            left_consecutive = 0

        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
            right_consecutive += 1
        else:
            right_consecutive = 0

        # 충분히 연속된 윈도우가 감지되면 차선 유효성 표시
        if left_consecutive >= min_consecutive_windows:
            left_valid = True
        if right_consecutive >= min_consecutive_windows:
            right_valid = True

    # 인덱스를 배열로 변환
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 차선 픽셀 추출
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if visualization:
        # 왼쪽 차선 픽셀 그리기 (빨간색)
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # 오른쪽 차선 픽셀 그리기 (파란색)
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # 윈도우 시각화 그리기
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return leftx, lefty, rightx, righty, out_img, left_valid, right_valid

def fit_polynomial(binary_warped, ploty, visualization=False, margin=50, minpix=30, prev_left_fit=None, prev_right_fit=None, 
                  xm_per_pix=3.7/700, ym_per_pix=30/720, min_consecutive_windows=4):
    """
    차선 픽셀을 기반으로 2차 다항식을 피팅합니다.

    Args:
        binary_warped (numpy.ndarray): 이진화된 BEV 변환 이미지.
        ploty (numpy.ndarray): y 좌표 배열.
        visualization (bool, optional): 시각화 여부. 기본값은 False.
        margin (int, optional): 슬라이딩 윈도우 마진. 기본값은 50.
        minpix (int, optional): 최소 픽셀 수. 기본값은 30.
        prev_left_fit (numpy.ndarray, optional): 이전 왼쪽 차선의 다항식 계수.
        prev_right_fit (numpy.ndarray, optional): 이전 오른쪽 차선의 다항식 계수.
        xm_per_pix (float, optional): X축 픽셀당 실제 거리 (미터/픽셀).
        ym_per_pix (float, optional): Y축 픽셀당 실제 거리 (미터/픽셀).
        min_consecutive_windows (int, optional): 최소 연속 윈도우 수. 기본값은 4.

    Returns:
        tuple: (updated_left_fit, updated_right_fit, left_fitx, right_fitx, out_img)
    """
    leftx, lefty, rightx, righty, out_img, left_valid, right_valid = find_lane_pixels(
        binary_warped,
        visualization=visualization,
        margin=margin,
        minpix=minpix,
        min_consecutive_windows=min_consecutive_windows
    )

    # 왼쪽 차선 피팅
    if left_valid and len(leftx) > 0 and len(lefty) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    else:
        left_fit = prev_left_fit
        left_fitx = None

    # 오른쪽 차선 피팅
    if right_valid and len(rightx) > 0 and len(righty) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    else:
        right_fit = prev_right_fit
        right_fitx = None

    return left_fit, right_fit, left_fitx, right_fitx, out_img

def calculate_center_lane(left_fitx, right_fitx, ploty, default_center_x=640):
    """
    왼쪽과 오른쪽 차선의 x좌표를 이용하여 중앙 차선의 x좌표를 계산합니다.

    Args:
        left_fitx (numpy.ndarray): 왼쪽 차선의 x 좌표 배열.
        right_fitx (numpy.ndarray): 오른쪽 차선의 x 좌표 배열.
        ploty (numpy.ndarray): y 좌표 배열.
        default_center_x (float, optional): 기본 중앙 차선 x 위치. 기본값은 640.

    Returns:
        numpy.ndarray or None: 중앙 차선의 x 좌표 배열 또는 None.
    """
    if left_fitx is not None and right_fitx is not None:
        center_fitx = (left_fitx + right_fitx) / 2
        return center_fitx
    elif left_fitx is not None:
        # 오른쪽 차선이 없을 경우 기본 값을 사용
        center_fitx = left_fitx + (default_center_x - left_fitx[-1])
        return center_fitx
    elif right_fitx is not None:
        # 왼쪽 차선이 없을 경우 기본 값을 사용
        center_fitx = right_fitx - (right_fitx[-1] - default_center_x)
        return center_fitx
    else:
        return None

def draw_center_lane(img, center_fitx, ploty, color=(0, 255, 255), thickness=5):
    """
    중앙 차선을 이미지에 그립니다.

    Args:
        img (numpy.ndarray): 입력 이미지 (BGR).
        center_fitx (numpy.ndarray): 중앙 차선의 x 좌표 배열.
        ploty (numpy.ndarray): y 좌표 배열.
        color (tuple, optional): 차선 색상 (B, G, R). 기본값은 노란색.
        thickness (int, optional): 차선 두께. 기본값은 5.

    Returns:
        numpy.ndarray: 중앙 차선이 그려진 이미지.
    """
    if center_fitx is not None:
        pts_center = np.array([np.transpose(np.vstack([center_fitx, ploty]))])
        cv2.polylines(img, np.int32([pts_center]), isClosed=False, color=color, thickness=thickness)
    return img
