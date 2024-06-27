import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
##具体于相关机械臂的api, 此处为kinova机械臂的kortex api
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient ##用于control
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient##用于reveive
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2
#diffusion_policy中的共享内存管理类
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

class KortexInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller needs its separate process (due to python GIL)
    """
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 ##kinova机械臂开启所需的信息
                 robot_ip,
                 username,
                 password,

                 frequency=125,
                 lookahead_time=0.1,
                 gain=300,
                 max_pos_speed=0.25, # 5% of max speed
                 max_rot_speed=0.16, # 5% of max speed
                 launch_timeout=3,
                 tcp_offset_pose=None,
                 payload_mass=None,
                 payload_cog=None,
                 joints_init=None,
                 joints_init_speed=1.05,
                 soft_real_time=False,
                 verbose=False,
                 receive_keys=None,
                 get_max_k=128,
                 ):
        """
        frequency: update frequency for the controller
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: m/s
        max_rot_speed: rad/s
        tcp_offset_pose: 6d pose
        payload_mass: float
        payload_cog: 3d position, center of gravity
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        """
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="KortexPositionalController")
        self.robot_ip = robot_ip
        self.username = username
        self.password = password
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                'ActualTCPPose',
                'ActualTCPSpeed',
                'ActualQ',
                'ActualQd',

                'TargetTCPPose',
                'TargetTCPSpeed',
                'TargetQ',
                'TargetQd'
            ]
        example = dict()
        # You will need to initialize Kortex API and get the actual state for each receive key
        # Initialize the Kortex API
        example = {
            'ActualTCPPose': np.zeros(6),
            'ActualTCPSpeed': np.zeros(6),
            'ActualQ': np.zeros(6),
            'ActualQd': np.zeros(6),
            'TargetTCPPose': np.zeros(6),
            'TargetTCPSpeed': np.zeros(6),
            'TargetQ': np.zeros(6),
            'TargetQd': np.zeros(6),
            'robot_receive_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[KortexPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    # When using the `with` statement, the __enter__ method is called
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start kortex api
        from kortex_api.SessionManager import SessionManager
        from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
        from kortex_api.RouterClient import RouterClient
        from kortex_api.RouterClientSendOptions import RouterClientSendOptions
        from kortex_api.TCPTransport import TCPTransport
        from kortex_api.autogen.messages import DeviceConfig_pb2, Common_pb2

        # Setup connection to the robot
        transport = TCPTransport()
        router = RouterClient(transport, RouterClientSendOptions())
        transport.connect(self.robot_ip, 10000)
        session_info = Common_pb2.CreateSessionInfo()
        session_info.username = self.username
        session_info.password = self.password
        session_manager = SessionManager(router)
        session_manager.CreateSession(session_info)
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        try:
            if self.verbose:
                print(f"[KortexPositionalController] Connect to robot: {self.robot_ip}")

            # set parameters
            if self.tcp_offset_pose is not None:
                tool_pose = Base_pb2.ToolConfiguration()
                tool_pose.tool_z = self.tcp_offset_pose[2]
                tool_pose.tool_y = self.tcp_offset_pose[1]
                tool_pose.tool_x = self.tcp_offset_pose[0]
                base.SetToolConfiguration(tool_pose)

            # init pose
            if self.joints_init is not None:
                joint_cmd = Base_pb2.Action()
                joint_cmd.name = "init_position"
                joint_cmd.application_data = ""
                joint_waypoint = joint_cmd.reach_joint_angles.joint_angles
                for i, angle in enumerate(self.joints_init):
                    joint_waypoint.joint_angles.append(Base_pb2.JointAngle(joint_identifier=i, value=angle))
                base.ExecuteAction(joint_cmd)

            # main loop
            dt = 1. / self.frequency
            feedback = base_cyclic.RefreshFeedback()
            curr_pose = [
                feedback.base.tool_pose_x,
                feedback.base.tool_pose_y,
                feedback.base.tool_pose_z,
                feedback.base.tool_pose_theta_x,
                feedback.base.tool_pose_theta_y,
                feedback.base.tool_pose_theta_z,
            ]
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            iter_idx = 0
            keep_running = True
            while keep_running:
                # start control iteration
                t_start = time.time()

                # send command to robot
                t_now = time.monotonic()
                pose_command = pose_interp(t_now)

                servo_cmd = Base_pb2.TwistCommand()
                servo_cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
                servo_cmd.duration = 0
                servo_cmd.twist.linear_x = pose_command[0]
                servo_cmd.twist.linear_y = pose_command[1]
                servo_cmd.twist.linear_z = pose_command[2]
                servo_cmd.twist.angular_x = pose_command[3]
                servo_cmd.twist.angular_y = pose_command[4]
                servo_cmd.twist.angular_z = pose_command[5]
                base.SendTwistCommand(servo_cmd)

                # update robot state
                feedback = base_cyclic.RefreshFeedback()
                state = {
                    'ActualTCPPose': [
                        feedback.base.tool_pose_x,
                        feedback.base.tool_pose_y,
                        feedback.base.tool_pose_z,
                        feedback.base.tool_pose_theta_x,
                        feedback.base.tool_pose_theta_y,
                        feedback.base.tool_pose_theta_z,
                    ],
                    'ActualTCPSpeed': [
                        feedback.base.tool_twist_linear_x,
                        feedback.base.tool_twist_linear_y,
                        feedback.base.tool_twist_linear_z,
                        feedback.base.tool_twist_angular_x,
                        feedback.base.tool_twist_angular_y,
                        feedback.base.tool_twist_angular_z,
                    ],
                    'ActualQ': [feedback.actuators[i].position for i in range(6)],
                    'ActualQd': [feedback.actuators[i].velocity for i in range(6)],
                    'TargetTCPPose': pose_command,
                    'TargetTCPSpeed': [0] * 6,  # Not available in Kortex API, use zeros
                    'TargetQ': [0] * 6,  # Not available in Kortex API, use zeros
                    'TargetQd': [0] * 6,  # Not available in Kortex API, use zeros
                    'robot_receive_timestamp': time.time()
                }
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[KortexPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # regulate frequency
                elapsed_time = time.time() - t_start
                sleep_time = max(0, dt - elapsed_time)
                time.sleep(sleep_time)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[KortexPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")

        finally:
            # mandatory cleanup
            base.Stop()
            session_manager.CloseSession()
            router.Close()
            transport.disconnect()
            self.ready_event.set()

            if self.verbose:
                print(f"[KortexPositionalController] Disconnected from robot: {self.robot_ip}")
