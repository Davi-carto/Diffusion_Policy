"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np

import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R
from diffusion_policy.real_world.real_env_VR import RealEnvVR
# from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from diffusion_policy.real_world.VR_teleop.VR_shared_memory import VuerTeleop

def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret

def pose_transform(origin_pose, frame_transform):
    '''
    输入的origin_pose是七维坐标，格式为[x,y,z,qx,qy,qz,qw]
    输入的frame_transform是六维坐标，格式为[x,y,z,roll,pitch,yaw]
    '''
    # 定义旋转和平移
    rotation = R.from_euler('xyz', frame_transform[3:], degrees=False)  # 定义要应用的旋转 (roll, pitch, yaw)
    translation = np.array(frame_transform[:3])  # 定义要应用的平移

    # 提取原始坐标中的平移和旋转部分
    position = origin_pose[:3]
    orientation = R.from_quat(origin_pose[3:])  # 从四元数创建旋转对象

    # 应用变换
    new_position = rotation.apply(position) + translation
    new_orientation = rotation * orientation 

    # 转换结果从旋转对象到旋转向量形式
    new_orientation_rotvec = new_orientation.as_rotvec()

    #组合新的姿态
    new_pose = np.concatenate((new_position, new_orientation_rotvec))

    return new_pose

right_base2world=[-0.265, 0.265, 0.049,-2.356, 0.000, -2.356]
left_base2world=[0.265, 0.265, 0.049,2.356, -0.000, -0.785]
def pose_transform_inv(origin_pose, frame_transform=left_base2world):
    '''
    输入的origin_pose是七维坐标，格式为[x,y,z,qx,qy,qz,qw]
    输入��frame_transform是六维坐标，格式为[x,y,z,roll,pitch,yaw]
    先对frame_transform求逆矩阵，再做坐标变换
    '''
    # 构建frame_transform的4x4变换矩阵
    rotation = R.from_euler('xyz', frame_transform[3:], degrees=False)
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = rotation.as_matrix()
    transform_mat[:3, 3] = frame_transform[:3]
    
    # 求逆矩阵
    inv_transform_mat = fast_mat_inv(transform_mat)
    
    # 从逆矩阵中提取旋转和平移
    inv_rotation = R.from_matrix(inv_transform_mat[:3, :3])
    inv_translation = inv_transform_mat[:3, 3]

    # 提取原始坐标中的平移和旋转部分
    position = origin_pose[:3]
    orientation = R.from_rotvec(origin_pose[3:])

    # 应用变换
    new_position = inv_rotation.apply(position) + inv_translation
    new_orientation = inv_rotation * orientation

    # 构建4x4变换矩阵作为结果
    result_mat = np.eye(4)
    result_mat[:3, :3] = new_orientation.as_matrix()
    result_mat[:3, 3] = new_position

    # 组合新的姿态
    new_pose = result_mat

    return new_pose

# baes_world_frame_transform=[-0.23,0.23,-0.325,-3.142, -0.785, -0.785]
# 基于base——link运动 baes_world_frame_transform=[[0.265, -0.265, 0.049,-2.356, 0.000, 0.785],[0.265, 0.265, 0.049,2.356, -0.000, -0.785]
# 基于base运动baes_world_frame_transform=[[-0.265, 0.265, 0.049,-2.356, 0.000, -2.356],[0.265, 0.265, 0.049,2.356, -0.000, -0.785]]
def ur_controller(teleop_state,right_base2world=right_base2world,left_base2world=left_base2world):
    '''
    teleop_state = {
            'head_rmat': np.eye(3, dtype=np.float32),
            'left_pose': np.zeros(7, dtype=np.float32),
            'right_pose': np.zeros(7, dtype=np.float32),
            'left_gripper_position': np.zeros(1, dtype=np.float32),
            'right_gripper_position': np.zeros(1, dtype=np.float32),
            'receive_timestamp': time.time()
        }
    '''

    right_pose = teleop_state['right_pose']
    # print("right_pose:",right_pose)
    target_right_pose = pose_transform(right_pose,right_base2world)

    left_pose = teleop_state['left_pose']
    # print("left_pose:",left_pose)
    target_left_pose = pose_transform(left_pose,left_base2world)
    # print("target_left_pose:",target_left_pose)

    left_gripper_position = teleop_state['left_gripper_position']
    right_gripper_position = teleop_state['right_gripper_position']
    # 将gripper_position归一化到0-1之间
    gripper_length = 0.13  # 夹爪最大张开长度为0.13m
    left_gripper_position = np.clip(left_gripper_position / gripper_length, 0, 1)
    right_gripper_position = np.clip(right_gripper_position / gripper_length, 0, 1)

    # 根据夹爪开度判断夹爪状态
    left_gripper_closed = 1 if left_gripper_position < 0.4 else 0
    right_gripper_closed = 1 if right_gripper_position < 0.4 else 0

    return target_right_pose, target_left_pose, right_gripper_closed, left_gripper_closed

@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_num', '-rn', default=2, type=int, help="Number of robots to control")
@click.option('--robot_ips', '-ri', 
    default=('192.168.1.60', '192.168.1.50'), 
    multiple=True, 
    help="UR5's IP addresses e.g. 192.168.0.204. Provide one IP for each robot.")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=30, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(output, robot_num, robot_ips, vis_camera_idx, init_joints, frequency, command_latency):
    ##通过循环频率和设置每个时间步（time_step）的时长
    dt = 1/frequency
    ##Python 的 multiprocessing 模块提供了一个名为 SharedMemoryManager 的类,它用于在多个进程之间共享内存。
    ##SharedMemoryManager的对象必须在所有使用它的进程中创建和访问。共享内存管理器必须在所有使用它的进程中启动和关闭。
    ##该共享内存对象后续会被传入到管理相机数据和实体机械臂数据的类中，进一步实现SharedMemoryRingBuffer和SharedMemoryQueue类。
    with SharedMemoryManager() as shm_manager:
        ##KeystrokeCounter类用于记录键盘按键，并提供按键计数功能。
        ##Spacemouse类用于读取SpaceMouse数据。
        ####RealEnv类主要负责1.接收处理realsens的数据2.机械臂RTDE的数据（控制和读取）3.管理每个演示数据的开始，结束和丢弃。
        with KeystrokeCounter() as key_counter, \
            RealEnvVR(
                output_dir=output,
                robot_num=robot_num,
                robot_ips=robot_ips,
                # recording resolution
                obs_image_resolution=(1280,720),
                frequency=frequency,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager
            ) as env, \
            VuerTeleop(shm_manager=shm_manager, multi_realsense=env.realsense) as vuer_teleop :
            
            cv2.setNumThreads(1)
            # realsense exposure
            env.realsense.set_exposure(exposure=130, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)

            time.sleep(1.0)
            print('Ready!')
            ##读取机械臂最新一帧的状态（数据），作为初始状态。
            state = env.get_robot_state(id=1)
            '''state中包含的数据有：
            # {
            #     'ActualTCPPose',
            #     'ActualTCPSpeed',
            #     'ActualQ',
            #     'ActualQd',
            #
            #     'TargetTCPPose',
            #     'TargetTCPSpeed',
            #     'TargetQ',
            #     'TargetQd'
            # }'''
            target_pose = state['TargetTCPPose']
            # actual_pose = state['ActualTCPPose']
            # print("actual_pose:",actual_pose)
            # print("target_pose",target_pose)
            #time.monotonic() 是单调时钟，返回自系统启动后经过的秒数，不受系统时间被改变的影响，适用于需要测量时间间隔的场景。
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False

            stage = 0
            while not stop:
                # calculate timing
                ##计算这次循环的结束时间，在循环末尾严格等待到这个时间点才会结束此次循环。
                ##目的是严格控制循环频率，便于后续相关数据延迟的计算和数据的时间��步
                t_cycle_end = t_start + (iter_idx + 1) * dt
                ##确定读取space mouse数据的时间点
                t_sample = t_cycle_end - command_latency
                ##确定执行机械臂动作的时间点
                t_command_target = t_cycle_end + 10*dt
                # pump obs
                #get_obs()函数返回的是一个字典，包含了相机图像、机械臂状态、时间戳的信息。
                ''' 
                obs={
                    camera_0: (T_o,H,W,C)
                    camera_1:（T_o,H,W,C)
                    camera_2:（T_o,H,W,C)
                    ........................
                    robot_eef_pose: (T_o,6),
                    robot_eef_pose_vel: (T_o,6),
                    robot_joint: (T_o,6)
                    robot_joint_vel: (T_o,6)
                    timestamp=obs_align_timestamps:(T_o,1)
                        }
                '''
                #get_obs的详细内容请参考diffusion_policy/real_world/real_env.py的get_obs()函数
                obs = env.get_obs()

                # handle key presses
                ##key_counter.get_press_events()函数返回的是一个列表，包含了所有在前一时间步到当前时间步之间被按下的键的KeyCode。
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        # Start recording
                        # start_episode()函数用于开始一个新的演示数据。
                        #start_episode的详细内容请参考diffusion_policy/real_world/real_env.py的start_episode()函数
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='e'):
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                        # delete
                    # elif key_stroke == Key.space:
                    #     stage += 1
                ##stage为一个整数，记录着空格键被按的次数
                ##每按一次空格键，代表着进入了下一个阶段，因此stage加1
                stage = key_counter[Key.space]

                # visualize
                vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()
                episode_id = env.replay_buffer.n_episodes
                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )

                cv2.imshow('default', vis_img)
                #cv2.pollKey() 用于检查用户是否按下了键盘上的任意键，以便及时响应用户交互
                #但注意此函数是非阻塞的
                cv2.pollKey()

                ##严格控制时间，到t_sample时刻开始读取space mouse数据
                precise_wait(t_sample)

                # #调整机械臂末端位置到合适位置，转成world坐标系下
                # target_right_pose = pose_transform_inv(actual_pose)
                # print("target_right_pose:",target_right_pose)

                teleop_state = vuer_teleop.get_vr_state()
                target_right_pose, target_left_pose, right_gripper_closed, left_gripper_closed = ur_controller(teleop_state)
                # 将两个机械臂的控制指令组合成一个action
                action = np.concatenate([
                    target_right_pose.reshape(-1)[:6],  # 右臂位姿 (6维)
                    [right_gripper_closed],             # 右臂夹爪状态 (1维)
                    target_left_pose.reshape(-1)[:6],   # 左臂位姿 (6维)
                    [left_gripper_closed]               # 左臂夹爪状态 (1维)
                ])
                # execute teleop command
                env.exec_actions(
                    actions=[action],
                    #t_command_target = t_cycle_end + dt
                    #按我们的延迟设定，当前时间步的机械臂动作指令应该在下一时间步执行，也就是time_stamps = time.time() + dt
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=[stage])

                #严格控制时间，到t_cycle_end时刻结束此次循环，严格控制循环频率。
                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()
