#include <ros/ros.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <std_msgs/Bool.h>
#include <cmath>

class JudgmentNode {
public:
    JudgmentNode(ros::NodeHandle& nh) {
        // 파라미터 로드
        nh.param("safe_distance", safe_distance_, 20.0f); // 안전 거리 (미터)
        nh.param("min_speed", min_speed_, 5.56f);         // 최소 속도 (m/s, 20 km/h)

        // 퍼블리셔 초기화
        hazard_pub_ = nh.advertise<std_msgs::Bool>("/mobinha/hazard_warning", 10);

        // 서브스크라이버 초기화
        track_box_sub_ = nh.subscribe("/mobinha/perception/lidar/track_box", 10, 
                                      &JudgmentNode::trackBoxCallback, this);
    }

private:
    ros::Publisher hazard_pub_;
    ros::Subscriber track_box_sub_;

    float safe_distance_; // 안전 거리 (미터)
    float min_speed_;     // 최소 속도 (m/s)

    // 판단 콜백 함수
    void trackBoxCallback(const jsk_recognition_msgs::BoundingBoxArray::ConstPtr& msg) {
        bool hazard_detected = false;
        int hazard_count = 0;

        for (const auto& box : msg->boxes) {
            // 장애물의 클래스 (예: 1 = 차량)
            int cls = box.label;
            if (cls != 1) continue; // 차량만 판단 대상

            // 객체와 자차 간의 거리 계산
            float x = box.pose.position.x; // 전방 거리
            float y = box.pose.position.y; // 좌우 거리
            float distance = std::sqrt(x * x + y * y); // 객체와 자차의 거리

            // 상대 속도 계산
            float relative_speed = box.value; // m/s 단위 가정

            // 정적 장애물 처리
            if (relative_speed < 0.1) { // 정적 장애물로 간주
                hazard_detected = true;
                hazard_count++;
                ROS_WARN("Static hazard detected! Position: [%.2f, %.2f], Distance: %.2f m", 
                         x, y, distance);
                continue;
            }

            // 동적 장애물 처리
            float vx = box.pose.orientation.x;
            float vy = box.pose.orientation.y;
            float angle = std::atan2(vy, vx); // 라디안 단위

            // 자차 진행 방향(ego_car 기준 x축)과 상대 속도 비교
            bool approaching = (angle > M_PI / 2 || angle < -M_PI / 2);

            // 위험 판단 조건: 거리와 속도 기준
            if (relative_speed > min_speed_ && approaching) {
                hazard_detected = true;
                hazard_count++;
                ROS_WARN("Dynamic hazard detected! Distance: %.2f m, Speed: %.2f m/s, Approach Angle: %.2f degrees", 
                         distance, relative_speed, angle * 180.0 / M_PI);
            }
        }

        // 판단 결과 퍼블리시
        std_msgs::Bool hazard_msg;
        hazard_msg.data = hazard_detected;
        hazard_pub_.publish(hazard_msg);

        if (hazard_detected) {
            ROS_INFO("Total hazards detected: %d", hazard_count);
        } else {
            ROS_INFO("No hazards detected.");
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "judgment_node");
    ros::NodeHandle nh("~"); // private NodeHandle for parameters

    JudgmentNode judgment_node(nh);

    ROS_INFO("Judgment node started.");

    ros::spin();

    return 0;
}
