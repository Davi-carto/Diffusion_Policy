from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from scipy.spatial.transform import Rotation as R
import numpy as np

def pose_transform(origin_pose, frame_transform):
    # 定义旋转和平移
    rotation = R.from_euler('xyz', frame_transform[3:], degrees=False)  # 定义要应用的旋转 (roll, pitch, yaw)
    translation = np.array(frame_transform[:3])  # 定义要应用的平移

    # 提取原始坐标中的平移和旋转部分
    position = origin_pose[:3]
    orientation = R.from_euler('xyz', origin_pose[3:], degrees=False)

    # 应用变换
    new_position = rotation.apply(position) + translation
    new_orientation = orientation * rotation


    # 转换结果从旋转对象回到欧拉角形式
    new_orientation_euler = new_orientation.as_euler('xyz', degrees=False)

    #组合新的姿态
    new_pose = np.concatenate((new_position, new_orientation_euler))

    print("原始六维坐标:", origin_pose)
    print("变换后的六维坐标:", new_pose)

    return new_pose



rtde_r = RTDEReceiveInterface("192.168.1.60",100)
rtde_c = RTDEControlInterface("192.168.1.60",100)

while True:
    actual_p = rtde_r.getActualTCPPose()
    # base_link_2_world = []
    print(actual_p)
