import rosbag
from geometry_msgs.msg import Point
import rospy
import pandas as pd

# CSV 파일 경로
csv_file = './3point.csv'

# CSV 파일 읽기
data = pd.read_csv(csv_file)

# ROSbag 생성
rosbag_filename = 'points_20hz.bag'
bag = rosbag.Bag(rosbag_filename, 'w')

try:
    # 20Hz로 퍼블리시 (주기 = 0.05초), 총 100초 동안 = 2000번 퍼블리시
    publish_rate = 20  # Hz
    total_duration = 44  # seconds
    total_messages = publish_rate * total_duration

    for i in range(total_messages):
        # Create Point 메시지
        msg = Point()
        index = i % len(data)  # 순환적으로 데이터 사용
        msg.x = data.iloc[index]['x']
        msg.y = data.iloc[index]['y']
        msg.z = data.iloc[index]['z']

        # 메시지 타임스탬프 계산
        timestamp = rospy.Time.from_sec(i * (1.0 / publish_rate))

        # ROSbag에 메시지 쓰기
        bag.write('/point_data', msg, timestamp)
finally:
    bag.close()

print(f"ROSbag 파일 생성 완료: {rosbag_filename}")
