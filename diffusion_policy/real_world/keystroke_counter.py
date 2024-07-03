from pynput.keyboard import Key, KeyCode, Listener
from collections import defaultdict
from threading import Lock

class KeystrokeCounter(Listener):
    def __init__(self):
        self.key_count_map = defaultdict(lambda:0)
        self.key_press_list = list()
        self.lock = Lock()
        super().__init__(on_press=self.on_press, on_release=self.on_release)
    
    def on_press(self, key):
        with self.lock:
            self.key_count_map[key] += 1
            self.key_press_list.append(key)
    
    def on_release(self, key):
        #那键盘控制运动时使用，否则用pass
        self.key_count_map[key] = 0
        # pass
    
    def clear(self):
        with self.lock:
            self.key_count_map = defaultdict(lambda:0)
            self.key_press_list = list()
    
    def __getitem__(self, key):
        with self.lock:
            return self.key_count_map[key]
    
    def get_press_events(self):
        with self.lock:
            events = list(self.key_press_list)
            self.key_press_list = list()
            return events

if __name__ == '__main__':
    import time
    with KeystrokeCounter() as key_counter:
        # try:
        #     while True:
        #         print('Space:', counter[Key.space])
        #         print('q:', counter[KeyCode(char='q')])
        #         time.sleep(1/60)
        # except KeyboardInterrupt:
        #     events = counter.get_press_events()
        #     print(events)
        stop = False
        try:
            while not stop:
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
                
                sm_state = [x_vel, y_vel, 0, 0, 0, 0]
                print(sm_state)

                # print('x_:', counter[Key.space])
                # print('q:', counter[KeyCode(char='q')])
                time.sleep(1/10)
        except KeyboardInterrupt:
            events = counter.get_press_events()
            print(events)