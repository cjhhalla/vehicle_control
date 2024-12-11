# image_processing.py

import cv2
import numpy as np

def draw_scale(img, xm_per_pix, ym_per_pix):
    h, w = img.shape[:2]

    # 축 그리기: x축과 y축에 대한 간격을 설정
    num_x_ticks = 10  # X축에 그릴 눈금 수
    num_y_ticks = 10  # Y축에 그릴 눈금 수
    x_spacing = w // num_x_ticks  # X축 눈금 간격 (픽셀 단위)
    y_spacing = h // num_y_ticks  # Y축 눈금 간격 (픽셀 단위)

    # X축 레이블 그리기
    for i in range(0, w, x_spacing):
        x_real = i * xm_per_pix  # 실제 X축 거리 계산
        cv2.putText(img, f"{x_real:.1f}m", (i, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img, (i, h), (i, h-10), (255, 255, 255), 1)

    # Y축 레이블 그리기
    for i in range(0, h, y_spacing):
        y_real = (h - i) * ym_per_pix  # 실제 Y축 거리 계산 (위쪽이 0이므로 y값을 뒤집어서 계산)
        cv2.putText(img, f"{y_real:.1f}m", (10, i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img, (0, i), (10, i), (255, 255, 255), 1)

    return img
