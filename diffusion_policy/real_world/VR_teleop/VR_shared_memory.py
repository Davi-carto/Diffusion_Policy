import multiprocessing as mp
import numpy as np
import time
from pathlib import Path
import yaml
import cv2

from .TeleVision import OpenTeleVision
from .Preprocessor import VuerPreprocessor
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore
# from constants_vuer import tip_indices
# from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from scipy.spatial.transform import Rotation as R

class VuerTeleop(mp.Process):
    def __init__(self, 
                 shm_manager, 
                 multi_realsense,
                #  config_file_path,
                 get_max_k=1, 
                 frequency=60,
                 dtype=np.float32):
        super().__init__()
        
        self.frequency = frequency
        self.dtype = dtype
        self.multi_realsense = multi_realsense
        # 设置分辨率和裁剪大小
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        # 设置单个图像的形状和尺寸
        self.single_img_shape = (self.resolution[0], self.resolution[1], 3)
        self.img_height, self.img_width = self.resolution[:2]
        self.camera_num = multi_realsense.n_cameras
        # 使用shm_manager创建共享内存用于图像传输
        self.img_array = shm_manager.SharedMemory(size=np.prod((self.camera_num, ) + self.single_img_shape) * np.uint8().itemsize)
        self.img_array_np = np.ndarray((self.camera_num,) + self.single_img_shape, dtype=np.uint8, buffer=self.img_array.buf)
        
        # 创建图像队列和切换流媒体的事件
        self.image_queue = mp.Queue()
        self.toggle_streaming = mp.Event()
        
        # 初始化OpenTeleVision和VuerPreprocessor
        self.tv = OpenTeleVision(self.single_img_shape, self.img_array, self.image_queue, self.toggle_streaming)
        self.processor = VuerPreprocessor()

        # 注释掉重定向配置相关代码
        # RetargetingConfig.set_default_urdf_dir('../assets')
        # with Path(config_file_path).open('r') as f:
        #     cfg = yaml.safe_load(f)
        # left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        # right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        # self.left_retargeting = left_retargeting_config.build()
        # self.right_retargeting = right_retargeting_config.build()

        example = {
            'head_rmat': np.eye(3, dtype=np.float32),
            'left_pose': np.zeros(7, dtype=np.float32),
            'right_pose': np.zeros(7, dtype=np.float32),
            'left_gripper_position': np.zeros(1, dtype=np.float32),
            'right_gripper_position': np.zeros(1, dtype=np.float32),
            'receive_timestamp': time.time()
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(shm_manager, example, get_max_k, get_time_budget=0.2,
            put_desired_frequency=frequency)

        self.stop_event = mp.Event()
        self.ready_event = mp.Event()

    def get_vr_state(self):
        return self.ring_buffer.get()


    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        self.ready_event.set()

        while not self.stop_event.is_set():
            # 从image_ring_buffer读取最新的图像
            latest_image_data = self.multi_realsense.get_vis()
            if latest_image_data is not None and 'color' in latest_image_data:
                for i in range(self.camera_num):
                    np.copyto(self.img_array_np[i], latest_image_data['color'][i][:,:,::-1])
                # print("R,G,B:", self.img_array_np[0][300,100,0], self.img_array_np[0][300,100,1], self.img_array_np[0][300,100,2])
                # 使用OpenCV显示图像
                # cv2.imshow('Camera 0', self.img_array_np[0].copy())
                # cv2.imshow('Camera 1', self.img_array_np[1].copy())
                # cv2.waitKey(1)  # 给OpenCV一个更新窗口的机会

            # 处理输入数据
            head_mat, left_wrist_mat, right_wrist_mat, left_fingers_mat, right_fingers_mat = self.processor.process(self.tv)

            # 提取头部旋转矩阵
            head_rmat = head_mat[:3, :3]

            # 计算左右手的位姿
            #scipy返回的四元数是[x,y,z,w]
            left_pose = np.concatenate([left_wrist_mat[:3, 3],
                                        R.from_matrix(left_wrist_mat[:3, :3]).as_quat()])

            right_pose = np.concatenate([right_wrist_mat[:3, 3],
                                         R.from_matrix(right_wrist_mat[:3, :3]).as_quat()])
            
            # 计算拇指指尖和食指指尖的距离
            # 获取左手拇指和食指指尖的索引
            left_thumb_tip = left_fingers_mat[4]  # 拇指指尖索引为4
            left_index_tip = left_fingers_mat[8]  # 食指指尖索引为8
            
            # 获取右手拇指和食指指尖的索引  
            right_thumb_tip = right_fingers_mat[4]
            right_index_tip = right_fingers_mat[8]
            
            # 计算左右手拇指和食指指尖的欧氏距离
            left_pinch_dist = np.linalg.norm(left_thumb_tip - left_index_tip)
            right_pinch_dist = np.linalg.norm(right_thumb_tip - right_index_tip)
            


            self.ring_buffer.put({
                'head_rmat': head_rmat.astype(np.float32),
                'left_pose': left_pose.astype(np.float32),
                'right_pose': right_pose.astype(np.float32),
                'left_gripper_position': left_pinch_dist.astype(np.float32),
                'right_gripper_position': right_pinch_dist.astype(np.float32),
                'receive_timestamp': time.time()
            })
            # print("VR is running")
            time.sleep(1 / self.frequency)

if __name__ == "__main__":
    shm_manager = mp.Manager()
    vr_teleop = VuerTeleop(shm_manager)
    vr_teleop.start()
    try:
        while True:
            vr_state = vr_teleop.get_vr_state()
            print("Current VR state:", vr_state)
            # 模拟从外部获取左右图像并存入ring buffer
            dummy_left_image = np.random.randint(0, 255, vr_teleop.single_img_shape, dtype=np.uint8)
            dummy_right_image = np.random.randint(0, 255, vr_teleop.single_img_shape, dtype=np.uint8)
            vr_teleop.put_images(dummy_left_image, dummy_right_image)
            time.sleep(0.1)
    except KeyboardInterrupt:
        vr_teleop.stop()
