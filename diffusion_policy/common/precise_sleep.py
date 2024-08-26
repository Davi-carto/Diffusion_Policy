import time

# 这两个函数都适用于需要精确控制时间的场景。
# precise_sleep 用于在需要等待固定时间长度时保持精确，而 precise_wait 则用于确保事件在某个精确的时间点触发
# slack_time是 允许的时间抖动（也就三允许的精度误差）
def precise_sleep(dt: float, slack_time: float=0.001, time_func=time.monotonic):
    """
    Use hybrid of time.sleep and spinning to minimize jitter.
    Sleep dt - slack_time seconds first, then spin for the rest.
    """
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass
    return

def precise_wait(t_end: float, slack_time: float=0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return
