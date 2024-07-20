from diffusion_policy.real_world.gamepad_shared_memory  import Gamepad
import time
from multiprocessing.managers import SharedMemoryManager
if __name__ == "__main__":
    with SharedMemoryManager() as shm_manager:

        # Replace with actual shared memory manager instance
        gamepad = Gamepad(shm_manager)
        gamepad.start()
        try:
            while True:
                axis_state = gamepad.get_axis_state()
                print("Current axis state:", axis_state)
                time.sleep(0.1)
        except KeyboardInterrupt:
            gamepad.stop()
