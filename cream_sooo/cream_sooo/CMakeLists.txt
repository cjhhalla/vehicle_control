cmake_minimum_required(VERSION 3.0.2)
project(cream_sooo)

## C++ 표준 설정 (C++14)
add_compile_options(-std=c++14)

## Catkin 패키지 찾기
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  jsk_recognition_msgs
)

## Catkin 패키지 선언
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs jsk_recognition_msgs
)

## 포함 디렉토리 설정
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## 실행 파일 선언
add_executable(judgment_node src/judgment_node.cpp)

## 메시지 생성 후 빌드되도록 의존성 추가
# 커스텀 메시지를 사용하지 않으므로 add_dependencies는 필요 없음

## 라이브러리 링크
target_link_libraries(judgment_node
  ${catkin_LIBRARIES}
)
