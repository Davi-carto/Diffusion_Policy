from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.rtde_interpolation_controller import RTDEInterpolationController
from diffusion_policy.real_world.multi_realsense import MultiRealsense, SingleRealsense
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator, 
    TimestampActionAccumulator,
    align_timestamps
)
from diffusion_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)

DEFAULT_OBS_KEY_MAP = {
    # robot
    #左侧的key为RTDE的key，右侧的key为实际的自定义的observation的key
    'ActualTCPPose': 'robot_eef_pose',
    'ActualTCPSpeed': 'robot_eef_pose_vel',
    'ActualQ': 'robot_joint',
    'ActualQd': 'robot_joint_vel',
    # timestamps
    'step_idx': 'step_idx',
    'timestamp': 'timestamp'
}

class RealEnvVR:
    def __init__(self, 
            output_dir,
            robot_ips,
            robot_num = 2,
            # env params
            frequency=10,
            n_obs_steps=2,
            # obs
            obs_image_resolution=(640,480),
            max_obs_buffer_size=30,
            camera_serial_numbers=None,
            obs_key_map=DEFAULT_OBS_KEY_MAP,
            obs_float32=False,

            # action
            max_pos_speed=0.05,
            max_rot_speed=0.3,
            # robot
            # tcp_offset=0.13,
            tcp_offset=0.1,
            init_joints=False,
            # video capture params
            video_capture_fps=30,
            video_capture_resolution=(1280,720),
            # saving params
            record_raw_video=True,
            thread_per_video=2,
            video_crf=21,
            # vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(1280,720),
            # shared memory
            shm_manager=None
            ):
        assert frequency <= video_capture_fps
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()

        # 用于将相机图像调整为模型需要的输入尺寸
        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution, 
            # obs output rgb
            bgr_to_rgb=True)
        color_transform = color_tf
        if obs_float32:
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255

        def transform(data):
            data['color'] = color_transform(data['color'])
            return data
        
        # 用于可视化多个相机视角，先获取最佳的行列数和应该调整的分辨率
        # 再生成一个新的transform用于调整视频图像分辨率
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(camera_serial_numbers),
            in_wh_ratio=obs_image_resolution[0]/obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution
        )
        vis_color_transform = get_image_transform(
            input_res=video_capture_resolution,
            output_res=(rw,rh),
            bgr_to_rgb=False
        )
        def vis_transform(data):
            data['color'] = vis_color_transform(data['color'])
            return data

        recording_transfrom = None
        recording_fps = video_capture_fps
        recording_pix_fmt = 'bgr24'
        if not record_raw_video:
            recording_transfrom = transform
            recording_fps = frequency
            recording_pix_fmt = 'rgb24'

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps, 
            codec='h264',
            input_pix_fmt=recording_pix_fmt, 
            crf=video_crf,
            thread_type='FRAME',
            thread_count=thread_per_video)

        realsense = MultiRealsense(
            serial_numbers=camera_serial_numbers,
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            record_fps=recording_fps,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            get_max_k=max_obs_buffer_size,

            transform=transform,
            vis_transform=None,
            recording_transform=recording_transfrom,

            video_recorder=video_recorder,
            verbose=False
            )
        
        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense,
                row=row,
                col=col,
                rgb_to_bgr=False
            )

        cube_diag = np.linalg.norm([1,1,1])
        j_init = np.array([0,-90,-90,-90,90,0]) / 180 * np.pi
        if not init_joints:
            j_init = None
        
        robots = []
        for i in range(robot_num):
            robot = RTDEInterpolationController(
                shm_manager=shm_manager,
                robot_ip=robot_ips[i],
                frequency=125, # UR5 CB3 RTDE 
                lookahead_time=0.1,
                gain=300,
                max_pos_speed=max_pos_speed*cube_diag,
                max_rot_speed=max_rot_speed*cube_diag,
                launch_timeout=3,
                tcp_offset_pose=[0,0,tcp_offset,0,0,0],
                payload_mass=None,
                payload_cog=None,
                joints_init=j_init,
                joints_init_speed=1.05,
                soft_real_time=False,
                verbose=False,
                receive_keys=None,
                get_max_k=max_obs_buffer_size
                )
            robots.append(robot)
        self.realsense = realsense
        self.robots = robots
        self.multi_cam_vis = multi_cam_vis
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.obs_key_map = obs_key_map
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_realsense_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

        self.start_time = None
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready and all([robot.is_ready for robot in self.robots])

    ##start和start_wait的区别？
    def start(self, wait=True):
        self.realsense.start(wait=False)
        for robot in self.robots:
            robot.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        for robot in self.robots:
            robot.stop(wait=False)
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.realsense.start_wait()
        for robot in self.robots:
            robot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()
    
    def stop_wait(self):
        for robot in self.robots:
            robot.stop_wait()
        self.realsense.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        "observation dict"
        assert self.is_ready
        # get data
        # 30 Hz, camera_receive_timestamp
        #math.ceil函数对一个值进行向上取整
        #k表示要获取最新的K帧数据
        #若相机帧率为30，交互频率为10（每秒10个time_step），则每个相机在一个time_step会新产生3张图片，
        #若设置每次预测时，参考2个time_step（n_obs_steps=2），则要从RingBuffer中获取k=2*3=6张图片，
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency))
        ##从MultiRealsense类的get方法中获得每个相机各自最新k帧数据，
        ##MultiRealsense类会为每个相机实例化一个SingleRealsense类，MultiRealsense类的get方法会调用SingleRealsense类的get方法get（）方法会调用SingleRealsense类的get方法，
        self.last_realsense_data = self.realsense.get(
            k=k, 
            out=self.last_realsense_data)
        #
        # 125 hz, robot_receive_timestamp
        # last_robot_data = self.robots[0].get_all_state()
        # 修改获取机器人状态的部分
        last_robot_data = []
        for robot in self.robots:
            last_robot_data.append(robot.get_all_state())

        # both have more than n_obs_steps data

        # align camera obs timestamps
        dt = 1 / self.frequency
        #last_realsense_data.values()返回的是一个字典，字典的键是‘rgb’或‘timestamp'
        #从self.last_realsense_data中获取所有摄像头的最新时间戳，然后找出这些时间戳中的最大值，即为当前时间戳
        last_timestamp = np.max([x['timestamp'][-1] for x in self.last_realsense_data.values()])
        #计算前T_o个time_step的时间戳
        #[::-1] 这部分是 Python 的列表切片语法，表示对列表进行倒序操作，将数组元素颠倒。
        # 因此，np.arange(self.n_obs_steps)[::-1] 将得到 [n_obs_steps-1, n_obs_steps-2,..., 1, 0]
        #以T_o = 3为例，
        #obs_align_timestamps = [last_timestamp-0.2,last_timestamp-0.1, last_timestamp]
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)

        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            ##找到最接近各时间戳对应的索引位置
            for t in obs_align_timestamps:
                #找出小于当前时间戳 t 的索引位置
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                #如果找到了小于当前时间戳 t 的索引位置，则取最后一个索引位置，否则取第一个索引位置
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            #valued的键是‘color'，为什么是color而不是rgb？
            camera_obs[f'camera_{camera_idx}'] = value['color'][this_idxs]
            #camera_obs={camera_0: (T_o,H,W,C),camera_1:（T_o,H,W,C),camera_2:（T_o,H,W,C)}


        robot_obs_raw = dict()
        robot_obs = dict()
        # align robot obs
        ##找到最接近各时间戳对应的索引位置
        for i, robot_data in enumerate(last_robot_data):
            robot_timestamps = robot_data['robot_receive_timestamp']
            this_timestamps = robot_timestamps
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
                
            ##将RTDE的key和自定义的observation的key进行映射
            # 为每个机器人添加前缀以区分
            for k, v in robot_data.items():
                if k in self.obs_key_map:
                    # 添加机器人编号前缀，例如 'robot_0_eef_pose'
                    new_key = f'robot_{i}_{self.obs_key_map[k]}'
                    robot_obs_raw[new_key] = v

        #挑选出与时间戳对应的robot_obs
        for k, v in robot_obs_raw.items():
            robot_obs[k] = v[this_idxs]

        #robot_obs={robot_eef_pose: (T_o,6),robot_eef_pose_vel: (T_o,6),robot_joint: (T_o,6),robot_joint_vel: (T_o,6)}
        # accumulate obsT_o
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                robot_obs_raw,
                robot_timestamps
            )

        # return obs
        #obs_data即为当前时刻的observation字典，包括摄像头的图像、机器人的状态等信息
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray, 
            stages: Optional[np.ndarray]=None):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)


        # convert action to pose
        #只执行时间戳大于当前时间的动作
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]

        # schedule waypoints  执行这里即向command_queue中添加了新命令，用来控制机械臂实际运动
        for i in range(len(new_actions)):
            self.robots[0].schedule_waypoint(
                pose=new_actions[i][:6],
                gripper_closed=new_actions[i][6],
                target_time=new_timestamps[i]
            )
            self.robots[1].schedule_waypoint(
                pose=new_actions[i][7:13], 
                gripper_closed=new_actions[i][13],
                target_time=new_timestamps[i]
            )
        
        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                new_stages,
                new_timestamps
            )

    def get_robot_state(self,id = 0):
        return self.robots[id].get_state()

       
    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        # start recording on realsense
        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!')
    
    def end_episode(self):
        "Stop recording"
        assert self.is_ready
        
        # stop video recorder
        self.realsense.stop_recording()

        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            stages = self.stage_accumulator.actions
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            if n_steps > 0:
                episode = dict()
                episode['timestamp'] = obs_timestamps[:n_steps]
                episode['action'] = actions[:n_steps]
                episode['stage'] = stages[:n_steps]
                for key, value in obs_data.items():
                    episode[key] = value[:n_steps]
                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')
            
            self.obs_accumulator = None
            self.action_accumulator = None
            self.stage_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            # shutil.rmtree函数的作用是删除目录树，即连同目录的所有文件和子目录都会被一并删除
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')

