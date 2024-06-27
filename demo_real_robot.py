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
                shm_manager=shm_manager
            ) as env:
            cv2.setNumThreads(1)
            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)

            time.sleep(1.0)
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
                # print(sm_state)
                #sm_state是一个长度为6的数组，分别表示x,y,z,rx,ry,rz，数据大小为（-1,1），为一个比例尺度信息
                #还需要将sm_state转换为实际的机械臂动作指令，这里的转换方式是将比例尺度信息乘以最大速度，得到实际的动作指令。
                dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)
                #Press SpaceMouse right button to unlock z axis.
                #Press SpaceMouse left button to enable rotation axes.
                if not sm.is_button_pressed(0):
                    # translation mode
                    drot_xyz[:] = 0
                else:
                    dpos[:] = 0
                if not sm.is_button_pressed(1):
                    # 2D translation mode
                    dpos[2] = 0
                # 将欧拉角转换为旋转矩阵
                drot = st.Rotation.from_euler('xyz', drot_xyz)
                # 更新目标位姿，格式满足实体机械臂的输入要求
                target_pose[:3] += dpos
                target_pose[3:] = (drot * st.Rotation.from_rotvec(
                    target_pose[3:])).as_rotvec()

                # execute teleop command
                env.exec_actions(
                    actions=[target_pose],
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
