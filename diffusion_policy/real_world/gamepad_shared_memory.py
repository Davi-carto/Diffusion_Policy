import pygame
import multiprocessing as mp
import numpy as np
import time
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), '../..')))
# for p in sys.path:
#     print(p)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


class Gamepad(mp.Process):
    def __init__(self, 
                 shm_manager, 
                 get_max_k=15, 
                 frequency=200,
                #  max_value=32767,  # Typical max value for gamepad axes
                 deadzone=(0,0,0,0,0,0), 
                 dtype=np.float16,
                 n_axes=5,
                 ):
        """ Continuously listen to gamepad events and update the latest state. """
        super().__init__()
        # if np.issubdtype(type(deadzone), np.number):
        #     deadzone = np.full(n_axes, fill_value=deadzone, dtype=dtype)
        # else:
        #     deadzone = np.array(deadzone, dtype=dtype)
        # assert (deadzone >= 0).all()

        self.frequency = frequency
        # self.max_value = max_value
        self.dtype = dtype
        # self.deadzone = deadzone
        self.n_axes = n_axes

        example = {
            'axis_event': np.zeros((n_axes,), dtype=np.float32),
            'receive_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(shm_manager, example, get_max_k)
        self.ring_buffer = ring_buffer

        self.stop_event = mp.Event()
        self.ready_event = mp.Event()

    def get_axis_state(self):
        return self.ring_buffer.get()['axis_event']

    #========== start stop API ===========

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

    # ========= main loop ==========
    def run(self):
        pygame.init()
        pygame.joystick.init()
        
        try:
            joystick_count = pygame.joystick.get_count()
            if joystick_count < 1:
                raise Exception("No gamepad connected")

            joystick = pygame.joystick.Joystick(0)
            joystick.init()

            axis_event = np.zeros((self.n_axes,), dtype=np.float32)
            self.ring_buffer.put({
                'axis_event': axis_event,
                'receive_timestamp': time.time()
            })
            self.ready_event.set()

            while not self.stop_event.is_set():
                for event in pygame.event.get():
                    if event.type == pygame.JOYAXISMOTION:
                        for axis_id in range(self.n_axes):
                            axis_value = joystick.get_axis(axis_id)
                            axis_event[axis_id] = axis_value

                  # invert y-axis for gamepads with left-handed layout
                axis_event = np.round(axis_event, 2)
                self.ring_buffer.put({
                    'axis_event': axis_event,
                    'receive_timestamp': time.time()
                })
                time.sleep(1 / self.frequency)
        finally:
            pygame.quit()

if __name__ == "__main__":
    shm_manager = None  # Replace with actual shared memory manager instance
    gamepad = Gamepad(shm_manager)
    gamepad.start()
    try:
        while True:
            axis_state = gamepad.get_axis_state()
            print("Current axis state:", axis_state)
            time.sleep(0.1)
    except KeyboardInterrupt:
        gamepad.stop()
