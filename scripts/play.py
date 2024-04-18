import os, shutil, sys
local_path = os.path.dirname(__file__)
log_path = os.path.join(local_path, 'log')
import isaacgym
assert isaacgym
import torch
import numpy as np
import glob
import pickle as pkl
from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from tqdm import tqdm


def load_policy(logdir):
    """
    从指定的日志目录加载策略模型。

    Attributes:
        logdir (str): 包含策略模型文件的目录路径。

    Methods：
        torch.jit.load：加载TorchScript模型。
        os：提供了一种使用操作系统依赖功能的方式。
        torch.cat：将张量在给定维度上进行连接。

    Returns:
        function: 返回一个函数，该函数接收观察值和信息字典，返回动作。

    Raises:
        FileNotFoundError: 如果策略模型文件不存在时抛出。

    Examples:
        >>> policy = load_policy('./logdir')
        >>> action = policy(observation, info={})
        >>> print(action)

    Note:
        - 策略函数使用了两个模型：一个主体模型和一个适应性模块。
        - 适应性模块用于生成潜在状态，主体模型将潜在状态和观察值合并后生成动作。
    """
    # 加载主体模型
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    # 加载适应性模块
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    # 定义策略函数
    def policy(obs, info={}):
        # 将观察历史转移到CPU并通过适应性模块生成潜在状态
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        # 将观察历史和潜在状态合并，通过主体模型生成动作
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        # 在信息字典中添加潜在状态
        info['latent'] = latent
        # 返回动作
        return action

    # 返回策略函数
    return policy


def load_env(label, headless=False):
    """
    加载指定标签的环境配置，并初始化环境。

    Attributes:
        label (str): 环境的标签，用于识别不同的运行配置。
        headless (bool): 是否以无头模式运行环境，无头模式下不显示图形界面。

    Methods：
        glob.glob：搜索匹配特定模式的文件路径。
        sorted：对列表进行排序。
        open：打开文件。
        pkl.load：从文件中加载pickle对象。
        hasattr：检查对象是否具有给定名称的属性。
        setattr：设置对象属性的值。
        getattr：获取对象属性的值。

    Returns:
        tuple: 包含初始化后的环境和策略模型的元组。

    Raises:
        FileNotFoundError: 如果指定标签的目录不存在或无法找到参数文件时抛出。

    Examples:
        >>> env, policy = load_env('example_label')
        >>> print(env, policy)

    Note:
        - 配置文件应包含环境和策略相关的所有必要配置信息。
        - 此函数还会关闭领域随机化（DR）设置，以便于评估脚本的执行。
    """
    # 使用glob模块搜索指定标签的所有运行目录
    dirs = glob.glob(f"../runs/{label}/*")
    # 对找到的目录进行排序，并选择第一个目录作为日志目录
    logdir = sorted(dirs)[-1]

    # 打开配置文件，加载配置信息
    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)  # 加载pickle文件
        print(pkl_cfg.keys())  # 打印配置键
        cfg = pkl_cfg["Cfg"]  # 获取配置信息
        print(cfg.keys())  # 打印配置信息的键

        # 更新Cfg类的属性
        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # 关闭领域随机化（DR）设置
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    # 设置环境和地形的配置
    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    # 导入环境包装器
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    # 初始化环境
    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)  # 应用历史包装器

    # 导入策略模型和日志记录器
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    # 加载策略模型
    policy = load_policy(logdir)

    # 返回环境和策略模型
    return env, policy


def play_go1(headless=True):
    """
    运行指定的环境配置，并使用加载的策略进行仿真，最后绘制仿真结果。

    Attributes:
        headless (bool): 是否以无头模式运行环境，无头模式下不显示图形界面。

    Methods：
        load_env：加载环境和策略模型。
        tqdm：提供一个进度条。
        torch.no_grad：禁用梯度计算。
        np.zeros：创建一个给定形状和类型的用0填充的数组。
        np.ones：创建一个给定形状和类型的用1填充的数组。
        plt.subplots：创建一个图形和一组子图。
        plt.show：显示所有打开的图形。

    Returns:
        None

    Raises:
        None

    Examples:
        >>> play_go1(headless=False)

    Note:
        - 仿真过程中，将根据预设的步态和指令控制机器人。
        - 最后绘制机器人的前进速度和关节位置随时间的变化。
    """
    # 导入必要的包
    from ml_logger import logger
    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    # 定义环境标签和加载环境
    label = "gait-conditioned-agility/pretrain-v0/train"
    env, policy = load_env(label, headless=headless)

    # 设置仿真步骤和步态
    num_eval_steps = 500
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "galloping": [0.25, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    # 定义控制指令
    gaits_c = "pacing"
    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.5, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits[gaits_c])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    # 初始化记录数组
    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    joint_positions = np.zeros((num_eval_steps, 12))

    # 重置环境
    env.start_recording()
    obs = env.reset()
    frames = []

    # 开始仿真循环
    for i in tqdm(range(num_eval_steps)):
        # 录制当前画面帧
        img = env.render(mode="rgb_array")
        with torch.no_grad():  # 禁用梯度计算
            actions = policy(obs)  # 获取动作
        # 更新环境控制指令
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)  # 执行一步仿真
        frames.append(img)  # 保存画面帧

        measured_x_vels[i] = env.base_lin_vel[0, 0]
        joint_positions[i] = env.dof_pos[0, :].cpu()

    # 绘制前进速度和关节位置图
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title(f"Forward Linear Velocity of {gaits_c}")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    axs[1].set_title(f"Joint Positions of {gaits_c}")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Joint Position (rad)")

    plt.tight_layout()
    # 保存视频
    logger.save_video(frames, f'{log_path}/{gaits_c}/{gaits_c}.mp4', fps=30)
    # 保存图像
    plt.savefig(f'{log_path}/{gaits_c}/{gaits_c}.svg')
    plt.show()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=True)
