from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.robot_devices.robots.configs import KinovaRobotConfig
from lerobot.common.robot_devices.robots.kinova import KinovaRobot
from lerobot.common.robot_devices.utils import busy_wait
import time
import torch

inference_time_s = 60
fps = 10
device = "cuda"  # TODO: On Mac, use "mps" or "cpu"

ckpt_path = "outputs/train/pi0_kinova_v1_only_action_expert/checkpoints/060000/pretrained_model"
ckpt_path = "outputs/train/pi0_kinova_v1/checkpoints/060000/pretrained_model"
policy = PI0Policy.from_pretrained(ckpt_path)
policy.to(device)

robot_config = KinovaRobotConfig()
robot = KinovaRobot(robot_config)
robot.connect()
robot.back_home()

for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    observation = robot.capture_observation()

    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)

    observation["task"] = ["Pick up the orange and put it in the drawer."]

    # Compute the next action with the policy
    # based on the current observation
    action = policy.select_action(observation)
    # Remove batch dimension
    action = action.squeeze(0)
    # Move to cpu, if not already the case
    action = action.to("cpu")
    # Order the robot to move
    # print("action:", action)
    robot.send_action(action)

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)

robot.disconnect()