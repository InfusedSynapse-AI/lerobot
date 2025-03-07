import torch
import time
import numpy as np
from lerobot.common.robot_devices.robots.configs import KinovaRobotConfig
from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, Base_pb2
import rospy
import cv2
from sensor_msgs.msg import JointState

class KinovaRobot:
    def __init__(
        self,
        config: KinovaRobotConfig,
    ):
        self.config = config
        self.robot_type = self.config.type
        self.arm = None
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.logs = {}
        self.robot_name = "my_gen3"
        self.qpos = [0.0] * 8
        self.max_gripper_width = 0.085 # GRIPPER_WIDTH_MAX
        self.max_gripper_joint = 0.79301 # GRIPPER_JOINT_MAX
        self.min_gripper_joint = 0.00698 # GRIPPER_JOINT_MIN

        # connect ros
        rospy.init_node("pi0", anonymous=True)
        rospy.Subscriber(f"/{self.robot_name}/joint_states", JointState, self.robot_state_cb)
    
    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper_width"]
        state_names = action_names
        return {
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": state_names,
            },
            "observation.state_ros": {
                "dtype": "float32",
                "shape": (8,),
                "names": state_names,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    def connect(self):
        if self.is_connected:
            print("kinovaRobot is already connected. Do not run `robot.connect()` twice.'")
            raise ConnectionError()

        # Connect the arm
        """Setup API"""
        error_callback = lambda kException: print("_________ callback error _________ {}".format(kException))
        self.transport = TCPTransport()
        router = RouterClient(self.transport, error_callback)
        self.transport.connect(self.config.ip, self.config.port)

        """Create session"""
        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = self.config.username
        session_info.password = self.config.password
        session_info.session_inactivity_timeout = 60000   # (milliseconds)
        session_info.connection_inactivity_timeout = 2000 # (milliseconds)

        print("Creating session for communication")
        self.session_manager = SessionManager(router)
        self.session_manager.CreateSession(session_info)
        print("Session created")

        self.arm = BaseClient(router)
        
        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True
    
    def robot_state_cb(self, data):
        self.qpos = list(data.position)[:8]
        self.qpos[-1] = (1-(abs(self.qpos[-1])-self.min_gripper_joint) / (self.max_gripper_joint-self.min_gripper_joint)) * self.max_gripper_width

    def parse_joint_angles(self, joint_states):
        return np.array([joint_angle.value for joint_angle in joint_states.joint_angles], dtype=np.float32)
    
    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise ConnectionError()

        # Get joint angles of arm
        before_lread_t = time.perf_counter()
        joint_states = self.parse_joint_angles(self.arm.GetMeasuredJointAngles())
        joint_states = torch.from_numpy(joint_states)
        self.logs[f"read_arm_pos_dt_s"] = time.perf_counter() - before_lread_t
        
        # Get gripper position
        before_lread_t = time.perf_counter()
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        gripper_measure = self.arm.GetMeasuredGripperMovement(gripper_request)
        gripper_position = gripper_measure.finger[0].value
        self.logs[f"read_gripper_pos_dt_s"] = time.perf_counter() - before_lread_t

        # Early exit when recording data is not requested
        if not record_data:
            return

        # TODO(rcadene): Add velocity and other info
        gripper_position_ros = torch.tensor([self.qpos[-1]], dtype=torch.float32)
        state = torch.cat((joint_states, gripper_position_ros), dim = 0)
        state_ros = torch.tensor(self.qpos, dtype=torch.float32)
        action = state

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        obs_dict["observation.state_ros"] = state_ros
        print(f"state:{state}, state_ros:{state_ros}")
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
            print("=" * 100)
            print(obs_dict[f"observation.images.{name}"].shape)
            print(obs_dict[f"observation.images.{name}"])
            save_img = cv2.cvtColor(obs_dict[f"observation.images.{name}"].numpy(), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"/home/cuhk/quebinbin/workspace/projects/lerobot/{name}.png", save_img)
            print("=" * 100)
        return obs_dict, action_dict

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise ConnectionError()

        # Get joint angles of arm
        before_lread_t = time.perf_counter()
        joint_states = self.parse_joint_angles(self.arm.GetMeasuredJointAngles())
        joint_states = torch.from_numpy(joint_states)
        self.logs[f"read_arm_pos_dt_s"] = time.perf_counter() - before_lread_t

        # Get gripper position
        before_lread_t = time.perf_counter()
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        gripper_measure = self.arm.GetMeasuredGripperMovement(gripper_request)
        gripper_position = gripper_measure.finger[0].value
        self.logs[f"read_gripper_pos_dt_s"] = time.perf_counter() - before_lread_t

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries and format to pytorch
        obs_dict = {}
        gripper_position_ros = torch.tensor([self.qpos[-1]])
        state = torch.cat((joint_states, gripper_position_ros), dim = 0)
        state_ros = torch.tensor(self.qpos)
        obs_dict["observation.state"] = state
        obs_dict["observation.state_ros"] = state_ros
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict

    def print_logs(self):
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self):
        if not self.is_connected:
            print("kinovaRobot is not connected. You need to run `robot.connect()` before disconnecting.'")
            raise ConnectionError()
        
        # Close API session
        self.session_manager.CloseSession()

        # Disconnect from the transport object
        self.transport.disconnect()

        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()