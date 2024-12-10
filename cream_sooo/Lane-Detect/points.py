import cv2
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

# 전역 변수 초기화
points = []
paused_frame = None  # 멈춘 프레임 저장
bridge = CvBridge()

# 좌표 선택을 위한 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global points, paused_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) == 0:
            paused_frame = param.copy()  # 첫 클릭에서 현재 프레임을 멈춤
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)} selected at: ({x}, {y})")
        if len(points) == 4:
            cv2.destroyAllWindows()  # 선택 완료 후 창 닫기

# ROS 압축 이미지 콜백 함수
def image_callback(msg):
    global paused_frame
    if paused_frame is None:  # 처음 클릭이 발생하면 멈추기
        # 압축 이미지를 OpenCV 형식으로 변환
        frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        resized_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Frame", resized_frame)
        cv2.setMouseCallback("Frame", mouse_callback, resized_frame)
    else:
        cv2.imshow("Frame", paused_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("User exit")

# ROS 노드 초기화 및 압축 이미지 토픽 구독
def main():
    rospy.init_node('compressed_image_listener', anonymous=True)
    rospy.Subscriber("/gmsl_camera/dev/video1/compressed", CompressedImage, image_callback)
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
