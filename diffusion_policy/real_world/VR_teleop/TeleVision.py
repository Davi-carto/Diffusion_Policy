import time
from vuer import Vuer
from vuer.events import ClientEvent
from vuer.schemas import ImageBackground, group, Hands, WebRTCStereoVideoPlane, DefaultScene
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore, Value
import numpy as np
import asyncio

import cv2
# from webrtc.zed_server import *


# OpenTeleVision类定义
# 该类用于处理视频流和手部追踪
class OpenTeleVision:
    # 初始化方法
    def __init__(self, single_img_shape, shared_img_array, queue, toggle_streaming, stream_mode="image", cert_file="./cert.pem", key_file="./key.pem", ngrok=False):
        # self.app=Vuer()
        # 设置图像形状
        self.single_img_shape = single_img_shape
        self.img_height, self.img_width = single_img_shape[:2]

        # 根据ngrok参数创建Vuer应用
        if ngrok:
            self.app = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
        else:
            self.app = Vuer(host='0.0.0.0', port=8037, cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)

        # 添加事件处理器
        # 一个装饰器模式的应用。这里的连续括号实际上是两个独立的函数调用：
        # self.app.add_handler("HAND_MOVE") 返回一个装饰器函数
        # 这个返回的装饰器函数立即被调用，参数是self.on_hand_move
        self.app.add_handler("HAND_MOVE")(self.on_hand_move)
        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)
        
        # 根据流模式设置不同的主函数
        if stream_mode == "image":
            self.img_array = np.ndarray((2,) + single_img_shape, dtype=np.uint8, buffer=shared_img_array.buf)
            self.app.spawn(start=False)(self.main_image)
        # elif stream_mode == "webrtc":
        #     self.app.spawn(start=False)(self.main_webrtc)
        else:
            raise ValueError("stream_mode must be either 'webrtc' or 'image'")

        # 初始化共享内存数组
        self.left_hand_shared = Array('d', 16, lock=True)
        self.right_hand_shared = Array('d', 16, lock=True)
        self.left_landmarks_shared = Array('d', 75, lock=True)
        self.right_landmarks_shared = Array('d', 75, lock=True)
        
        self.head_matrix_shared = Array('d', 16, lock=True)
        self.aspect_shared = Value('d', 1.0, lock=True)
        
        # # WebRTC服务器设置
        # if stream_mode == "webrtc":
        #     # webrtc server
        #     if Args.verbose:
        #         logging.basicConfig(level=logging.DEBUG)
        #     else:
        #         logging.basicConfig(level=logging.INFO)
        #     Args.img_shape = img_shape
        #     # Args.shm_name = shm_name
        #     Args.fps = 60

        #     ssl_context = ssl.SSLContext()
        #     ssl_context.load_cert_chain(cert_file, key_file)

        #     app = web.Application()
        #     cors = aiohttp_cors.setup(app, defaults={
        #         "*": aiohttp_cors.ResourceOptions(
        #             allow_credentials=True,
        #             expose_headers="*",
        #             allow_headers="*",
        #             allow_methods="*",
        #         )
        #     })
        #     rtc = RTC(img_shape, queue, toggle_streaming, 60)
        #     app.on_shutdown.append(on_shutdown)
        #     cors.add(app.router.add_get("/", index))
        #     cors.add(app.router.add_get("/client.js", javascript))
        #     cors.add(app.router.add_post("/offer", rtc.offer))

        #     self.webrtc_process = Process(target=web.run_app, args=(app,), kwargs={"host": "0.0.0.0", "port": 8080, "ssl_context": ssl_context})
        #     self.webrtc_process.daemon = True
        #     self.webrtc_process.start()
        #     # web.run_app(app, host="0.0.0.0", port=8080, ssl_context=ssl_context)

        # 启动主进程
        # 创建一个新的进程来运行主程序
        self.process = Process(target=self.run)
        # 将进程设置为守护进程，这样当主程序退出时，这个进程也会自动退出
        self.process.daemon = True
        # 启动进程
        self.process.start()

    # 运行方法
    def run(self):
        # start the vuer server
        self.app.run()

    # 相机移动事件处理器
    async def on_cam_move(self, event, session, fps=60):
        # only intercept the ego camera.
        # if event.key != "ego":
        #     return
        try:
            # with self.head_matrix_shared.get_lock():  # Use the lock to ensure thread-safe updates
            #     self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            # with self.aspect_shared.get_lock():
            #     self.aspect_shared.value = event.value['camera']['aspect']
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            self.aspect_shared.value = event.value['camera']['aspect']
        except:
            pass
        # self.head_matrix = np.array(event.value["camera"]["matrix"]).reshape(4, 4, order="F")
        # print(np.array(event.value["camera"]["matrix"]).reshape(4, 4, order="F"))
        # print("camera moved", event.value["matrix"].shape, event.value["matrix"])

    # 手部移动事件处理器
    async def on_hand_move(self, event, session, fps=60):
        try:
            # with self.left_hand_shared.get_lock():  # Use the lock to ensure thread-safe updates
            #     self.left_hand_shared[:] = event.value["leftHand"]
            # with self.right_hand_shared.get_lock():
            #     self.right_hand_shared[:] = event.value["rightHand"]
            # with self.left_landmarks_shared.get_lock():
            #     self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()
            # with self.right_landmarks_shared.get_lock():
            #     self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten()

            self.left_hand_shared[:] = event.value["leftHand"]
            self.right_hand_shared[:] = event.value["rightHand"]
            #存储手部关键点的坐标信息
            #每只手可能有25个关键点,每个点有3个坐标(x,y,z),总共75个浮点数
            self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()
            self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten()
        except: 
            pass
    
    # # WebRTC主循环
    # async def main_webrtc(self, session, fps=60):
    #     ## Add the scene to the vuer app
    #     ## 这种写法session.upsert @ Hands(...)等同于session.upsert(Hands(...))，
    #     ## 是Vuer框架提供的一种简洁的语法糖，使得代码更易读和维护。
    #     session.set @ DefaultScene(frameloop="always")
    #     session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
    #     session.upsert @ WebRTCStereoVideoPlane(
    #             src="https://192.168.8.102:8080/offer",
    #             # iceServer={},
    #             key="zed",
    #             aspect=1.33334,
    #             height = 8,
    #             position=[0, -2, -0.2],
    #         )
    #     # # keep the session alive.
    #     while True:
    #         await asyncio.sleep(1)
    
    # 图像流主循环
    async def main_image(self, session, fps=60):
        session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
        end_time = time.time()
        while True:
            start = time.time()
            # print(end_time - start)
            # aspect = self.aspect_shared.value
            display_image = self.img_array

            #使用OpenCV显示图像
            # cv2.imshow('Camera 0', display_image[0])
            # cv2.imshow('Camera 1', display_image[1])
            # cv2.waitKey(1)  # 给OpenCV一个更新窗口的机会

            #ImageBackground 是Vuer框架中的一个组件，用于在3D场景中显示图像,
            # 详见https://docs.vuer.ai/en/latest/examples/07b_vr_hud.html
            session.upsert(
            [ImageBackground(
                # Can scale the images down.
                #::2 表示在第一个维度（通常是图像的高度）上每隔一行取一行。
                # 这实际上是在垂直方向上将图像的分辨率降低了一半
                # 在水平方向上截取图像的左半部分,display_image 可能包含了左右眼的图像数据。这行代码提取了左眼的图像
                display_image[0],
                # display_image[:self.img_height:2, ::2],
                # 'jpg' encoding is significantly faster than 'png'.
                format="jpeg",
                quality=80,
                key="left-image",
                interpolate=True,
                # fixed=True,
                #aspect=是宽高比,
                aspect=1.6667,
                # distanceToCamera=0.5,
                height = 4,
                position=[0, -1, 3],
                # rotation=[0, 0, 0],
                #layers=0是背景层,layers=1是左眼,layers=2是右眼
                layers=0, 
                alphaSrc="./vinette.jpg"
            ),
            # ImageBackground(
            #     # Can scale the images down.
            #     display_image[1],
            #     # display_image[self.img_height::2, ::2],
            #     # 'jpg' encoding is significantly faster than 'png'.
            #     format="jpeg",
            #     quality=80,
            #     key="right-image",
            #     interpolate=True,
            #     # fixed=True,
            #     aspect=1.6667,
            #     # distanceToCamera=0.5,
            #     height = 5,
            #     position=[0, -1, 3],
            #     ### Can also rotate the plane in-place
            #     # rotation=[0, 0, 0],
            #     layers=2, 
            #     alphaSrc="./vinette.jpg"
            # )
            ],
            # we place this into the background children list, so that it is
            # not affected by the global rotation
            to="bgChildren",
            )
            # rest_time = 1/fps - time.time() + start
            end_time = time.time()
            await asyncio.sleep(0.03)

    # 左手属性
    @property
    def left_hand(self):
        # with self.left_hand_shared.get_lock():
        #     return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")
        #C order (行优先,默认): order="C"
        #Fortran order (列优先): order="F"
        return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")
        
    # 右手属性
    @property
    def right_hand(self):
        # with self.right_hand_shared.get_lock():
        #     return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")
        return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")
        
    # 左手关键点属性
    @property
    def left_landmarks(self):
        # with self.left_landmarks_shared.get_lock():
        #     return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
        return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
    
    # 右手关键点属性
    @property
    def right_landmarks(self):
        # with self.right_landmarks_shared.get_lock():
            # return np.array(self.right_landmarks_shared[:]).reshape(25, 3)
        return np.array(self.right_landmarks_shared[:]).reshape(25, 3)

    # 头部矩阵属性
    @property
    def head_matrix(self):
        # with self.head_matrix_shared.get_lock():
        #     return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    # 宽高比属性
    @property
    def aspect(self):
        # with self.aspect_shared.get_lock():
            # return float(self.aspect_shared.value)
        return float(self.aspect_shared.value)

# 主函数
if __name__ == "__main__":
    resolution = (720, 1280)
    crop_size_w = 340  # (resolution[1] - resolution[0]) // 2
    crop_size_h = 270
    resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)  # 450 * 600
    img_shape = (2, resolution_cropped[0], resolution_cropped[1], 3)  # 900 * 600
    img_height, img_width = resolution_cropped[:2]  # 450 * 600

    shm_manager = Manager()
    shared_img_array = shm_manager.SharedMemory(size=np.prod(img_shape) * np.uint8().itemsize)

    tv = OpenTeleVision(resolution_cropped, shared_img_array, Queue(), Event(), cert_file="../cert.pem", key_file="../key.pem")
    while True:
        # print(tv.left_landmarks)
        # print(tv.left_hand)
        # tv.modify_shared_image(random=True)
        time.sleep(1)
