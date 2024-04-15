from isaacgym import gymutil, gymapi
import torch
from params_proto import Meta
from typing import Union

from go1_gym.envs.base.legged_robot import LeggedRobot
from go1_gym.envs.base.legged_robot_config import Cfg


class VelocityTrackingEasyEnv(LeggedRobot):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                 cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX"):
        """
        VelocityTrackingEasyEnv类，用于创建一个简化的速度跟踪环境，继承自LeggedRobot类。

        该环境专为腿部机器人设计，用于跟踪特定的速度目标。它允许自定义仿真环境的数量、是否以无头模式运行、
        是否采用特定的物理引擎等配置。

        Attributes:
            cfg (Cfg): 环境配置对象，包含环境和仿真的所有配置参数。
            eval_cfg (Cfg): 评估配置对象，用于评估环境时的配置。
            initial_dynamics_dict (dict): 初始动力学参数字典。
            physics_engine (str): 使用的物理引擎名称。
            sim_device (str): 仿真运行的设备。
            headless (bool): 是否以无头模式运行仿真。
            num_envs (int, optional): 仿真环境的数量。
            prone (bool): 是否使用俯卧姿态初始化机器人。
            deploy (bool): 是否为部署模式。

        Methods:
            __init__: 构造函数，初始化VelocityTrackingEasyEnv环境。

        Examples:
            >>> cfg = Cfg()  # 假设Cfg是一个配置类
            >>> env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=True, num_envs=10, cfg=cfg)
            创建一个在CUDA设备上，以无头模式运行的，包含10个环境的速度跟踪环境。

        Note:
            - 该环境需要安装特定的物理引擎和仿真库。
            - 配置对象Cfg需要根据实际情况进行定义和初始化。
        """
        # 如果num_envs不为None，则更新配置文件中的环境数量
        if num_envs is not None:
            cfg.env.num_envs = num_envs

        # 创建仿真参数对象
        sim_params = gymapi.SimParams()
        # 解析仿真配置参数
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)

        # 调用父类构造函数，完成环境的初始化
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, eval_cfg, initial_dynamics_dict)

    def step(self, actions):
        """
        执行一个仿真步骤，包括应用动作、更新环境状态并返回相关信息。

        该方法首先调用父类的step方法来执行动作并获取环境的基本反馈信息。然后，它会计算并更新一些额外的信息，
        如脚的位置、关节的位置和速度等，这些信息对于高级控制和分析可能是必要的。

        Parameters:
            actions (torch.Tensor): 执行的动作，形状为(num_envs, num_actions)。

        Returns:
            tuple: 包含观察、奖励、重置信号和额外信息的元组。

        Examples:
            >>> env = VelocityTrackingEasyEnv(sim_device="cuda:0", headless=True, num_envs=4)
            >>> actions = torch.randn(4, env.action_space.shape[0])
            >>> observations, rewards, resets, extras = env.step(actions)

        Note:
            - `extras`字典中包含了大量的额外信息，如特权观察、关节位置和速度等，这些信息对于调试和分析非常有用。
        """
        # 调用父类的step方法，获取基本的环境反馈信息
        self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras = super().step(actions)

        # 获取脚的位置信息
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]

        # 更新额外信息字典，包括特权观察、关节位置和速度等
        self.extras.update({
            "privileged_obs": self.privileged_obs_buf,  # 特权观察信息
            "joint_pos": self.dof_pos.cpu().numpy(),  # 关节位置
            "joint_vel": self.dof_vel.cpu().numpy(),  # 关节速度
            "joint_pos_target": self.joint_pos_target.cpu().detach().numpy(),  # 目标关节位置
            "joint_vel_target": torch.zeros(12),  # 目标关节速度，初始化为全零
            "body_linear_vel": self.base_lin_vel.cpu().detach().numpy(),  # 基座线速度
            "body_angular_vel": self.base_ang_vel.cpu().detach().numpy(),  # 基座角速度
            "body_linear_vel_cmd": self.commands.cpu().numpy()[:, 0:2],  # 基座线速度命令
            "body_angular_vel_cmd": self.commands.cpu().numpy()[:, 2:],  # 基座角速度命令
            "contact_states": (self.contact_forces[:, self.feet_indices, 2] > 1.).detach().cpu().numpy().copy(),  # 接触状态
            "foot_positions": (self.foot_positions).detach().cpu().numpy().copy(),  # 脚的位置信息
            "body_pos": self.root_states[:, 0:3].detach().cpu().numpy(),  # 基座位置
            "torques": self.torques.detach().cpu().numpy()  # 关节力矩
        })

        # 返回观察、奖励、重置信号和额外信息
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        """
        重置所有环境到初始状态，并返回初始观测值。

        该方法首先重置所有环境的状态，然后执行一个全零动作的步骤来获取初始的观测值。这是在环境开始新的episode时
        常用的步骤，确保环境处于一个干净的初始状态。

        Returns:
            torch.Tensor: 所有环境的初始观测值，形状为(num_envs, observation_space)。

        Examples:
            >>> env = VelocityTrackingEasyEnv(sim_device="cuda:0", headless=True, num_envs=4)
            >>> initial_observations = env.reset()
            获取四个环境的初始观测值。

        Note:
            - 该方法会重置所有环境，不论其之前的状态如何。
            - 返回的观测值可以直接用于决策模型的输入。
        """
        # 重置所有环境到初始状态
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # 执行一个全零动作的步骤来获取初始观测值，不计算梯度
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))

        # 返回所有环境的初始观测值
        return obs
