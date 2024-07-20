from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from scipy.spatial.transform import Rotation as R
import scipy.spatial.transform as st
import numpy as np
import time
import numpy
from keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from precise_sleep import precise_wait
##假设origin_pose为world坐标系下的六维坐标，frame_transform为从base坐标系到world坐标系的变换矩阵，则new_pose为base坐标系下的六维坐标。
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

    # print("原始六维坐标:", origin_pose)
    # print("变换后的六维坐标:", new_pose)

    return new_pose

frequency = 10
dt = 1/frequency
rtde_r = RTDEReceiveInterface("192.168.1.60",100)
rtde_c = RTDEControlInterface("192.168.1.60",100)
target_pose = rtde_r.getTargetTCPPose()
frame_transform = [0,0,0,-3.142, -0.785, -0.785]
with KeystrokeCounter() as key_counter:
    t_start = time.monotonic()
    iter_idx = 0
    while True:
        
        # actual_p = rtde_r.getActualTCPPose()
        # actual_q = rtde_r.getActualQ()
        # print("target_pose:", target_pose)
        # print("actual_p:   ", actual_p)
        # print("actual_q:", actual_q)    
        # print("tcp:", tcp)

        # calculate timing
        ##计算这次循环的结束时间，在循环末尾严格等待到这个时间点才会结束此次循环。
        ##目的是严格控制循环频率，便于后续相关数据延迟的计算和数据的时间同步
        t_cycle_end = t_start + (iter_idx + 1) * dt
        ##确定读取space mouse数据的时间点
        t_sample = t_cycle_end - 0.01
        ##确定执行机械臂动作的时间点
        t_command_target = t_cycle_end + dt

        precise_wait(t_sample)
        #     # get teleop command
        if key_counter[KeyCode(char='d')] >=50: x_vel = 1
        elif key_counter[KeyCode(char='d')] < 50 and key_counter[KeyCode(char='d')] >= 30: x_vel = 0.6
        elif key_counter[KeyCode(char='d')] < 30 and key_counter[KeyCode(char='d')] >= 20: x_vel = 0.4
        elif key_counter[KeyCode(char='d')] < 20 and key_counter[KeyCode(char='d')] >= 10: x_vel = 0.2
        elif key_counter[KeyCode(char='d')] < 10 and key_counter[KeyCode(char='d')] > 0 : x_vel = 0.1
        elif key_counter[KeyCode(char='a')] >=50: x_vel = -1
        elif key_counter[KeyCode(char='a')] < 50 and key_counter[KeyCode(char='a')] >= 30: x_vel = -0.6
        elif key_counter[KeyCode(char='a')] < 30 and key_counter[KeyCode(char='a')] >= 20: x_vel = -0.4
        elif key_counter[KeyCode(char='a')] < 20 and key_counter[KeyCode(char='a')] >= 10: x_vel = -0.2
        elif key_counter[KeyCode(char='a')] < 10 and key_counter[KeyCode(char='a')] > 0 : x_vel = -0.1
        else: x_vel = 0

        if key_counter[KeyCode(char='w')] >=50: y_vel = 1
        elif key_counter[KeyCode(char='w')] < 50 and key_counter[KeyCode(char='w')] >= 30: y_vel = 0.6
        elif key_counter[KeyCode(char='w')] < 30 and key_counter[KeyCode(char='w')] >= 20: y_vel = 0.4
        elif key_counter[KeyCode(char='w')] < 20 and key_counter[KeyCode(char='w')] >= 10: y_vel = 0.2
        elif key_counter[KeyCode(char='w')] < 10 and key_counter[KeyCode(char='w')] > 0 : y_vel = 0.1
        elif key_counter[KeyCode(char='s')] >=50: y_vel = -1
        elif key_counter[KeyCode(char='s')] < 50 and key_counter[KeyCode(char='s')] >= 30: y_vel = -0.6
        elif key_counter[KeyCode(char='s')] < 30 and key_counter[KeyCode(char='s')] >= 20: y_vel = -0.4
        elif key_counter[KeyCode(char='s')] < 20 and key_counter[KeyCode(char='s')] >= 10: y_vel = -0.2
        elif key_counter[KeyCode(char='s')] < 10 and key_counter[KeyCode(char='s')] > 0  : y_vel = -0.1
        else: y_vel = 0
        
        sm_state = numpy.array([x_vel, y_vel, 0, 0, 0, 0])
        print(sm_state)
        
        #sm_state是一个长度为6的数组，分别表示x,y,z,rx,ry,rz，数据大小为（-1,1），为一个比例尺度信息
        #还需要将sm_state转换为实际的机械臂动作指令，这里的转换方式是将比例尺度信息乘以最大速度，得到实际的动作指令。
        dpos = sm_state[:3] * (0.25/ frequency)
        drot_xyz = sm_state[3:] * (0 / frequency)
        origi_dpose = [dpos[0], dpos[1], 0, 0, 0, 0]
        new_dpose = pose_transform(origi_dpose, frame_transform)

        # 2D translation mode
        drot_xyz[:] = 0
        dpos[2] = 0

        # 将欧拉角转换为旋转矩阵
        drot = st.Rotation.from_euler('xyz', drot_xyz)
        # 更新目标位姿，格式满足实体机械臂的输入要求
        target_pose[:3] += new_dpose[:3]
        target_pose[3:] = (drot * st.Rotation.from_rotvec(
        target_pose[3:])).as_rotvec()
        print("target_pose:", target_pose)
        # 发送目标位姿
        # rtde_c.sendTargetTCPPose(target_pose)
        rtde_c.servoL(target_pose, 
                    0.5, 0.5, # dummy, not used by ur5
                    dt, 
                    0.1, 
                    300)

        precise_wait(t_cycle_end)
        iter_idx += 1
