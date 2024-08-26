"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============

Make sure you can hit the robot hardware emergency-stop button quickly! 
c
Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env import RealEnv

from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform

from diffusion_policy.real_world.gamepad_shared_memory import Gamepad

import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R

OmegaConf.register_new_resolver("eval", eval, replace=True)

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


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
#当使用@click.option装饰器时，将is_flag=True设置为True时，它表示该选项可以作为布尔标志使用，当选项存在时，其值将为True；当选项不存在时，其值将为False。
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=10, type=int, help="Action horizon for inference.")
###max_duration默认设置为60秒，每个epoh都在一分钟内结束
@click.option('--max_duration', '-md', default=90, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--temporal_agg', '-act', is_flag=True, default=False, help="Whether to use ACT ")
def main(input, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency,temporal_agg):
    # load match_dataset
    ##指定使用dataset中那个camera的图像来进行匹配
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        ##diffusion_policy/data/pusht_real/real_pusht_20230105/videos/中有135个文件夹，每个文件夹名为一个数字，代表一个episode
        ##glob方法在match_video_dir中查找所有的文件和文件夹并返回一个迭代器。
        ##'*/'模式用于匹配match_video_dir中的所有文件夹
        ##vid_dir = diffusion_policy/data/pusht_real/real_pusht_20230105/videos/episod_idx
        for vid_dir in match_video_dir.glob("*/"):
            ##vid_dir.stem返回目录路径的最后一部分，一般为文件夹名，这里代表episod index 的数字
            episode_idx = int(vid_dir.stem)
            ##match_video_path = diffusion_policy/data/pusht_real/real_pusht_20230105/videos/episod_idx/0.mp4
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            ##使用 skvideo 库的 vread 函数从视频文件中读取一帧图像
            ##skvideo.io.vread: 这是 skvideo 库中的函数，用于从视频文件中读取帧。
            ##它接受视频文件的完整路径作为参数，并可以根据需要设置读取的帧数
            ##num_frames=1: 这是 vread 函数的参数，表示要读取的帧数。在这里，它设置为 1，表示只读取视频文件中的第一帧
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
    
    # load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    print(f"Instantiating {cfg._target_}")
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params

        #这里去噪过程的迭代次数，会不会太少了？
        # 从16变为30个迭代次数，性能显著增加
        policy.num_inference_steps = 50 # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    elif 'robomimic' in cfg.name:
        # BCRNN model
        policy: BaseImagePolicy
        policy = workspace.model

        device = torch.device('cuda')
        policy.eval().to(device)

        # BCRNN always has action horizon of 1
        steps_per_inference = 1
        action_offset = cfg.n_latency_steps
        delta_action = cfg.task.dataset.get('delta_action', False)

    elif 'ibc' in cfg.name:
        policy: BaseImagePolicy
        policy = workspace.model
        policy.pred_n_iter = 5
        policy.pred_n_samples = 4096

        device = torch.device('cuda')
        policy.eval().to(device)
        steps_per_inference = 1
        action_offset = 1
        delta_action = cfg.task.dataset.get('delta_action', False)
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # setup experiment
    dt = 1/frequency
    ##obs_res = out_res =(wo, ho)
    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    ##Spacemouse是管理空间鼠标的类
    ##RealEnv类主要负责1.接收处理realsens的数据2.机械臂RTDE的数据（控制和读取）
    with SharedMemoryManager() as shm_manager:
        with Gamepad(shm_manager=shm_manager) as gamepad, RealEnv(
            output_dir=output, 
            robot_ip=robot_ip, 
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=True,
            #Bool变量，Whether to initialize robot joint configuration in the beginning.
            init_joints=init_joints,
            enable_multi_cam_vis=True,
            record_raw_video=True,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager) as env:
            cv2.setNumThreads(1)

            ##为multi_realsense相机设置参数
            # Should be the same as demo
            # realsense exposure
            env.realsense.set_exposure(exposure=130, gain=3)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)

            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            obs = env.get_obs()
            with torch.no_grad():
                ## reset state for stateful policies
                ## 大部分policy都没用到，直接pass
                policy.reset()
                ##get_real_obs_dict函数对图像或位置数据进行裁剪，图像THWC -> TCHW，使数据与Policy的输入格式匹配
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                ##当调用unsqueeze(0)时，它会在第一个维度（索引为0的维度）上增加一个新的维度。例如，如果原始张量的形状为(T_o, C, H, W)，
                #那么调用unsqueeze(0)之后，形状将变为(1, T_o, C, H, W)，即在第一个维度上增加了一个维度。
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 3
                del result

            baes_world_frame_transform = [0,0,0,-3.142, -0.785, -0.785]

            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                state = env.get_robot_state()
                target_pose = state['TargetTCPPose']
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    ##t_cycle_end是当前while循环的结束时间，dt是循环的间隔时间
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    ##episode_id是repaly_buffer中最后一个的episode的index
                    episode_id = env.replay_buffer.n_episodes
                    ##vis_img = (H,W,C)
                    vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        ##把match_img的图像大小调整为vis_img的大小，然后进行融合，为什么要做融合？
                        vis_img = np.minimum(vis_img, match_img)

                    text = f'Episode: {episode_id}'
                    ''' 
                    这段代码使用了OpenCV库中的putText函数，该函数用于在图像上绘制文本。下面是参数的具体含义
                    vis_img：要绘制文本的图像。
                    text：要绘制的文本内容。
                    (10, 20)：文本的起始位置坐标。
                    fontFace：字体类型，这里使用了OpenCV内置的简单字体。
                    fontScale：字体大小因子，用于缩放字体大小。
                    thickness：文本轮廓的厚度，如果为负值（如-1），则表示填充文本。
                    color：文本的颜色，由BGR颜色通道值组成，这里是白色。
                    '''
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    #'default'是窗口的名字
                    # [...,::-1] 是Python中的NumPy数组切片语法。... 表示所有轴，::-1 表示倒序索引。这里的目的是将图像数组中的颜色通道（通常是BGR通道）的顺序颠倒
                    # ，使其从RGB格式变为BGR格式，以在窗口中正确显示图像的颜色。
                    cv2.imshow('default', vis_img[...,::-1])
                    #cv2.pollKey(): 这是OpenCV库中的函数，用于检查是否有按键被按下
                    # ord() 函数是一个内置函数，用于获取指定字符的ASCII码值
                    ##cv2.pollKey()函数是非阻塞的，即如果没有按键被按下，它不会等待,程序会继续执行。
                    key_stroke = cv2.pollKey()
                    if key_stroke == ord('q'):
                        # Exit program
                        env.end_episode()
                        exit(0)
                    elif key_stroke == ord('c'):
                        # Exit human control loop
                        # hand control over to the policy
                        break

                    precise_wait(t_sample)
                    ##后续操作是将空间鼠标的控制命令转换为机械臂的控制命令，并执行机械臂的控制命令。
                    # get teleop command
                    x_vel, y_vel, not_use, lowx_vel, lowy_vel= gamepad.get_axis_state()
                    sm_state = np.array([-x_vel*0.3-lowx_vel, y_vel*0.3+lowy_vel, 0, 0, 0, 0])
                    # print(sm_state)
                    dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                    drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)
  
                    origi_dpose = [dpos[0], dpos[1], 0, 0, 0, 0]
                    new_dpose = pose_transform(origi_dpose, baes_world_frame_transform)

                    # 将欧拉角转换为旋转矩阵
                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    # 更新目标位姿，格式满足实体机械臂的输入要求
                    target_pose[:3] += new_dpose[:3]
                    ##这里好像应该用rpy（欧拉角）而不是旋转矢量
                    target_pose[3:] = (drot * st.Rotation.from_rotvec(
                        target_pose[3:])).as_rotvec()
                
                    # clip target pose
                    # target_pose[:2] = np.clip(target_pose[:2], [0.25, -0.45], [0.77, 0.40])

                    # execute teleop command
                    env.exec_actions(
                        actions=[target_pose], 
                        timestamps=[t_command_target-time.monotonic()+time.time()])
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    ##time.time()能提供当前的绝对时间，但受到系统时间调整的影响；
                    # 而time.monotonic()提供的时间虽然不受系统时间调整的影响，但它无法提供绝对时间，更适合用来测量时间间隔
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    term_area_start_timestamp = float('inf')
                    perv_target_pose = None

                    # try act
                    if temporal_agg:
                        max_iter_idx = max_duration / dt
                        max_iter_idx = int(np.ceil(max_iter_idx))
                        num_action_steps = policy.n_action_steps
                        all_time_actions = np.zeros([max_iter_idx//steps_per_inference, max_iter_idx+num_action_steps, 3])

                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        print('get_obs')
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy()
                            print('Inference latency:', time.time() - s)
                            print('Action:', action)
                            print('Action longth:', len(action))
                        
                        # try act
                        if temporal_agg:
                            s = time.time()
                            all_time_actions[[iter_idx//steps_per_inference], iter_idx:iter_idx+num_action_steps] = action
                            act_action = np.zeros((num_action_steps, 3))

                            for i in range(num_action_steps):
                                actions_for_curr_step = all_time_actions[:, iter_idx + i]
                                actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                                actions_for_curr_step = actions_for_curr_step[actions_populated]
                                k = 0.01
                                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                                exp_weights = exp_weights / exp_weights.sum()
                                exp_weights = np.expand_dims(exp_weights, axis=1)
                                act_action[i] = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                            print('ACT latency:', time.time() - s)
                            print("Act_action: ", act_action)
                            print('Act_action longth:', len(action))

                        # convert policy action to env actions
                        if temporal_agg :
                            # if delta_action:
                            #     assert len(action) == 1
                            #     if perv_target_pose is None:
                            #         perv_target_pose = obs['robot_eef_pose'][-1]
                            #     this_target_pose = perv_target_pose.copy()
                            #     this_target_pose[[0,1]] += action[-1]
                            #     perv_target_pose = this_target_pose
                            #     this_target_poses = np.expand_dims(this_target_pose, axis=0)
                            # else:
                                this_target_poses = np.zeros((len(act_action), len(target_pose)), dtype=np.float64)
                                this_target_poses[:] = target_pose
                                this_target_poses[:,[0,1,2]] = act_action
                        else:
                            if delta_action:
                                assert len(action) == 1
                                if perv_target_pose is None:
                                    perv_target_pose = obs['robot_eef_pose'][-1]
                                this_target_pose = perv_target_pose.copy()
                                this_target_pose[[0,1]] += action[-1]
                                perv_target_pose = this_target_pose
                                this_target_poses = np.expand_dims(this_target_pose, axis=0)
                            else:
                                this_target_poses = np.zeros((len(action), len(target_pose)), dtype=np.float64)
                                this_target_poses[:] = target_pose
                                this_target_poses[:,[0,1,2]] = action
                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        
                        # 如何确定的？跟哪里有关？ 好像是command_latency
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # clip actions
                        # this_target_poses[:,:2] = np.clip(
                        #     this_target_poses[:,:2], [0.25, -0.45], [0.77, 0.40])

                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])


                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s'):
                            # Stop episode
                            # Hand control back to human
                            env.end_episode()
                            print('Stopped.')
                            break

                        # auto termination
                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')

                        term_pose = np.array([ -0.509,  0.286,  0.715,  2.22014183e+00, -2.22184883e+00, -4.07186655e-04])
                        curr_pose = obs['robot_eef_pose'][-1]
                        print("curr_pose: ",curr_pose)
                        dist = np.linalg.norm((curr_pose - term_pose)[:3], axis=-1)
                        if dist < 0.03:
                            # in termination area
                            curr_timestamp = obs['timestamp'][-1]
                            if term_area_start_timestamp > curr_timestamp:
                                term_area_start_timestamp = curr_timestamp
                            else:
                                term_area_time = curr_timestamp - term_area_start_timestamp
                                if term_area_time > 0.5:
                                    terminate = True
                                    print('Terminated by the policy!')
                        else:
                            # out of the area
                            term_area_start_timestamp = float('inf')

                        if terminate:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()