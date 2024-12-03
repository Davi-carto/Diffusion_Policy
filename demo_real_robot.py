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
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R

# right_base2world=[0,0,0,-3.142, -0.785, -0.785]
right_base2world = [-0.265, 0.265, 0.049,-2.356, 0.000, -2.356]

def transform_velocity_and_update_pose(target_pose, base2world, world_velocity, dt):
    """
    将 World 坐标系下的六维速度转换到 Base 坐标系下，并将速度增量附加到目标位姿上。

    Parameters:
    - target_pose: np.ndarray, shape=(4, 4)
        Base 坐标系下的目标位姿齐次变换矩阵。
    - base2world: np.ndarray, shape=(4, 4)
        Base 到 World 坐标系的齐次变换矩阵（Base 为父坐标系）。
    - world_velocity: np.ndarray, shape=(6,)
        World 坐标系下的六维速度 [v_x, v_y, v_z, omega_x, omega_y, omega_z]。
    - dt: float
        时间步长。

    Returns:
    - new_pose: np.ndarray, shape=(4, 4)
        更新后的 Base 坐标系下的目标位姿齐次变换矩阵。
    """
    # 提取旋转矩阵 R_world_to_base
    R_world_to_base = base2world[:3, :3]  # 从 world 到 base 的旋转部分

    # 转换线速度和角速度到 Base 坐标系
    linear_velocity_world = world_velocity[:3]
    angular_velocity_world = world_velocity[3:]

    linear_velocity_base = R_world_to_base @ linear_velocity_world
    angular_velocity_base = R_world_to_base @ angular_velocity_world

    # 计算位姿增量
    delta_translation = linear_velocity_base * dt  # 平移增量

    # 使用 Rodrigues 公式计算旋转增量
    angular_velocity_magnitude = np.linalg.norm(angular_velocity_base)
    if angular_velocity_magnitude > 1e-6:  # 避免零角速度情况
        axis = angular_velocity_base / angular_velocity_magnitude
        delta_rotation = R.from_rotvec(axis * angular_velocity_magnitude * dt)
    else:
        delta_rotation = R.identity()

    # 更新目标位姿
    target_rotation = R.from_matrix(target_pose[:3, :3])
    new_rotation = delta_rotation * target_rotation  # 旋转增量叠加
    new_translation = target_pose[:3, 3] + delta_translation  # 平移增量叠加

    # 构造新的齐次变换矩阵
    new_pose = np.eye(4)
    new_pose[:3, :3] = new_rotation.as_matrix()
    new_pose[:3, 3] = new_translation

    return new_pose

@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency):
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
            Spacemouse(shm_manager=shm_manager) as sm, \
            RealEnv(
                output_dir=output,
                robot_ip=robot_ip,
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
                shm_manager=shm_manager,
                gripper_binary_mode = True
            ) as env:
            cv2.setNumThreads(1)
            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)

            time.sleep(3.0)
            print('Ready!')
            ##读取机械臂最新一帧的状态（数据），作为初始状态。
            state = env.get_robot_state()
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
            initial_pose = np.array([-0.43, 0.215, 0.610, 1.92, 0.95, -2.7])#(x,y,z,rx,ry,rz)
            gripper_closed = 0
            #time.monotonic() 是单调时钟，返回自系统启动后经过的秒数，不受系统时间被改变的影响，适用于需要测量时间间隔的场景。
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            while not stop:
                # calculate timing
                ##计算这次循环的结束时间，在循环末尾严格等待到这个时间点才会结束此次循环。
                ##目的是严格控制循环频率，便于后续相关数据延迟的计算和数据的时间同步
                t_cycle_end = t_start + (iter_idx + 1) * dt
                ##确定读取space mouse数据的时间点
                t_sample = t_cycle_end - command_latency
                ##确定执行机械臂动作的时间点
                t_command_target = t_cycle_end + dt
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
                    elif key_stroke == KeyCode(char='s'):
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
                # get teleop command
                sm_state = sm.get_motion_state_transformed()
                # print("sm_state:",sm_state)
                #sm_state是一个长度为6的数组，分别表示x,y,z,rx,ry,rz，数据大小为（-1,1），为一个比例尺度信息
                #还需要将sm_state转换为实际的机械臂动作指令，这里的转换方式是将比例尺度信息乘以最大速度，得到实际的动作指令。
                dpos = sm_state[:3] * (env.max_pos_speed / frequency) * 10
                drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency) * 5
                
                # 每按一次按钮，切换夹爪状态
                if sm.is_button_pressed(0):
                    # 等待按钮释放,避免一直按住导致多次切换
                    while sm.is_button_pressed(0):
                        time.sleep(0.01)
                    gripper_closed = 0 if gripper_closed == 1 else 1
                
                if sm.is_button_pressed(1):
                    target_pose = initial_pose.copy()
                else:
                    # Combine linear and angular velocities
                    world_velocity = np.concatenate([dpos, drot_xyz])

                    # Create base2world transformation matrix
                    base2world = np.eye(4)
                    base2world[:3, :3] = R.from_euler('xyz', right_base2world[3:]).as_matrix()
                    base2world[:3, 3] = right_base2world[:3]

                    # Convert current target_pose to 4x4 matrix if it isn't already
                    current_pose_matrix = np.eye(4)
                    current_pose_matrix[:3, :3] = R.from_rotvec(target_pose[3:]).as_matrix()
                    current_pose_matrix[:3, 3] = target_pose[:3]

                    # Transform velocity and update pose
                    new_pose_matrix = transform_velocity_and_update_pose(
                        target_pose=current_pose_matrix,
                        base2world=base2world,
                        world_velocity=world_velocity,
                        dt=1/frequency
                     )

                    # Convert back to target_pose format [x, y, z, rx, ry, rz]
                    target_pose[:3] = new_pose_matrix[:3, 3]
                    target_pose[3:] = R.from_matrix(new_pose_matrix[:3, :3]).as_rotvec()


                # execute teleop command
                # 将target_pose和gripper_closed合并为一个动作
                action = np.append(target_pose, gripper_closed)
                env.exec_actions(
                    actions=[action,],
                    #t_command_target = t_cycle_end + dt
                    #按我们的延迟设定，当前时间步的机械臂动作指令应该在下一时间步执行，也就是time_stamps = time.time() + dt
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=[stage])
                ##严格控制时间，到t_cycle_end时刻结束此次循环，严格控制循环频率。
                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()
