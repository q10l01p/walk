# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
from typing import Dict

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch
import torch

from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.base.base_task import BaseTask
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from go1_gym.utils.terrain import Terrain
from .legged_robot_config import Cfg


class LeggedRobot(BaseTask):
    def __init__(self, cfg: Cfg, sim_params, physics_engine, sim_device, headless, eval_cfg=None,
                 initial_dynamics_dict=None):
        """
        初始化LeggedRobot任务的类。

        此类负责初始化腿式机器人任务，包括读取配置、创建仿真环境、初始化奖励函数等。
        它继承自BaseTask，扩展了特定于腿式机器人的初始化和配置解析逻辑。

        Methods:
            _parse_cfg(cfg): 根据给定的配置参数解析并设置任务相关的配置。
            _init_command_distribution(env_ids): 根据环境ID初始化命令分布。
            _init_buffers(): 初始化用于存储仿真数据的PyTorch缓冲区。
            _prepare_reward_function(): 根据配置准备奖励函数。
            set_camera(position, lookat): 设置仿真环境的相机位置和观察点。

        Attributes:
            cfg (Cfg): 存储任务配置参数。
            eval_cfg (Optional[Cfg]): 存储评估配置参数，可选。
            sim_params (dict): 存储仿真参数。
            height_samples (NoneType): 高度采样初始化为None。
            debug_viz (bool): 调试可视化标志，默认为False。
            init_done (bool): 初始化完成标志，默认为False。
            initial_dynamics_dict (Optional[dict]): 存储初始动力学参数字典，可选。
            record_now (bool): 当前记录状态，默认为False。
            record_eval_now (bool): 当前评估记录状态，默认为False。
            collecting_evaluation (bool): 是否正在收集评估状态，默认为False。
            num_still_evaluating (int): 仍在评估的数量，默认为0。

        Examples:
            >>> cfg = load_configuration("legged_robot_config.yaml")
            >>> sim_params = {"gravity": -9.81, "timestep": 0.01}
            >>> legged_robot = LeggedRobot(cfg, sim_params, "physx", "cuda:0", False)
            >>> print("LeggedRobot initialized with", legged_robot.num_envs, "environments")

        Note:
            - 此类的实例化将自动创建仿真环境并进行必要的初始化。
            - 特别适用于需要复杂配置和初始化步骤的腿式机器人任务。
            - 通过继承BaseTask，LeggedRobot类可以轻松集成到不同的仿真框架中。
        """
        # 存储任务配置参数
        self.cfg = cfg
        # 存储评估配置参数
        self.eval_cfg = eval_cfg
        # 存储仿真参数
        self.sim_params = sim_params
        # 初始化高度采样为None
        self.height_samples = None
        # 初始化调试可视化标志为False
        self.debug_viz = False
        # 初始化完成标志为False
        self.init_done = False
        # 存储初始动力学参数字典
        self.initial_dynamics_dict = initial_dynamics_dict
        if eval_cfg is not None:
            # 解析评估配置参数
            self._parse_cfg(eval_cfg)
        # 解析主配置参数
        self._parse_cfg(self.cfg)

        # 调用父类构造函数以创建仿真环境
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless, self.eval_cfg)

        # 初始化命令分布
        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))

        if not self.headless:
            # 如果非无头模式，设置相机参数
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        # 初始化PyTorch缓冲区
        self._init_buffers()

        # 准备奖励函数
        self._prepare_reward_function()
        # 设置初始化完成标志为True
        self.init_done = True
        # 初始化当前记录状态为False
        self.record_now = False
        # 初始化当前评估记录状态为False
        self.record_eval_now = False
        # 初始化是否正在收集评估状态为False
        self.collecting_evaluation = False
        # 初始化仍在评估的数量为0
        self.num_still_evaluating = 0

    def step(self, actions):
        """
        执行一步仿真，并处理动作、观测值以及其他仿真后的逻辑。

        此方法首先将输入的动作按配置裁剪至合适的范围，并将其应用到仿真环境中。
        随后，执行一定数量的仿真步骤（由配置决定），在每步中计算并应用扭矩。
        最后，对观测值进行裁剪并返回仿真的结果。

        Methods:
            _compute_torques(actions): 根据输入的动作计算应用到仿真环境中的扭矩。
            render_gui(): 如果配置允许，渲染仿真的GUI界面。
            post_physics_step(): 在每次物理仿真步骤后调用，用于处理仿真后的逻辑。

        Attributes:
            actions (torch.Tensor): 输入的动作。
            cfg (Config): 仿真配置，包含动作和观测值的裁剪范围等信息。
            device (torch.device): 仿真运行的设备（如CPU或GPU）。
            gym (gym.Env): 仿真环境。
            sim (Simulation): 仿真实例。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]: 返回观测值、特权观测值、奖励值、重置标志和额外信息的元组。

        Examples:
            >>> agent = YourAgentClass(cfg, device, gym, sim)  # 创建代理实例
            >>> actions = torch.tensor([[0.1, 0.2, -0.1]])  # 定义动作
            >>> obs, priv_obs, rewards, resets, extras = agent.step(actions)  # 执行一步仿真
            >>> print(obs)  # 打印观测值

        Note:
            - 动作被裁剪是为了防止过大的动作导致仿真不稳定。
            - 观测值被裁剪是为了确保它们在神经网络中的有效性。
            - 该方法适用于需要连续控制的仿真环境。
        """
        # 获取配置中的动作裁剪范围
        clip_actions = self.cfg.normalization.clip_actions
        # 裁剪动作并转移到设备上
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # 记录之前的基础位置、四元数、线速度和足部速度
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()
        # 渲染GUI界面
        self.render_gui()
        # 根据配置的减少因子进行仿真步骤
        for _ in range(self.cfg.control.decimation):
            # 计算扭矩
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # 应用扭矩
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            # 进行仿真
            self.gym.simulate(self.sim)
            # 如果设备为CPU，则获取仿真结果
            self.gym.fetch_results(self.sim, True)
            # 刷新关节状态张量
            self.gym.refresh_dof_state_tensor(self.sim)
        # 调用处理仿真后逻辑的方法
        self.post_physics_step()

        # 获取配置中的观测值裁剪范围
        clip_obs = self.cfg.normalization.clip_observations
        # 裁剪观测值
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            # 裁剪特权观测值
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        # 返回结果元组
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """
        在物理仿真步骤之后执行的更新操作。

        此方法负责更新仿真环境的状态，包括刷新状态张量、执行图形更新、更新位置和速度信息、检查终止条件、计算奖励和观测值等。

        Attributes:
            sim: 仿真环境的引用。
            record_now: 是否记录当前图形步骤的标志。
            episode_length_buf: 当前episode的长度缓冲区。
            common_step_counter: 全局步骤计数器。
            base_pos: 基础位置信息。
            base_quat: 基础四元数信息。
            base_lin_vel: 基础线速度。
            base_ang_vel: 基础角速度。
            projected_gravity: 投影重力。
            foot_velocities: 足部速度。
            foot_positions: 足部位置。
            reset_buf: 需要重置的环境ID缓冲区。
            viewer: 可视化查看器。
            enable_viewer_sync: 是否启用查看器同步。
            debug_viz: 是否启用调试可视化。

        Methods:
            gym.refresh_actor_root_state_tensor: 刷新根状态张量。
            gym.refresh_net_contact_force_tensor: 刷新接触力张量。
            gym.refresh_rigid_body_state_tensor: 刷新刚体状态张量。
            gym.step_graphics: 执行图形更新。
            gym.render_all_camera_sensors: 渲染所有相机传感器。
            quat_rotate_inverse: 通过四元数逆旋转计算。
            _post_physics_step_callback: 物理步骤后的回调函数。
            check_termination: 检查是否满足终止条件。
            compute_reward: 计算奖励。
            reset_idx: 根据环境ID重置环境。
            compute_observations: 计算观测值。
            _draw_debug_vis: 绘制调试信息。
            _render_headless: 执行无头渲染。

        Returns:
            None

        Examples:
            >>> env.post_physics_step()  # 在物理仿真步骤之后执行更新操作

        Note:
            - 该方法是仿真环境中的关键步骤之一，确保仿真的连贯性和数据的实时更新。
            - 需要在每个物理仿真步骤之后调用此方法。
        """
        # 刷新仿真环境中的根状态张量
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # 刷新仿真环境中的接触力张量
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # 刷新仿真环境中的刚体状态张量
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.record_now:
            # 如果记录当前图形步骤，则执行图形更新
            self.gym.step_graphics(self.sim)
            # 渲染所有相机传感器
            self.gym.render_all_camera_sensors(self.sim)

        # 更新当前episode的长度
        self.episode_length_buf += 1
        # 更新全局步骤计数器
        self.common_step_counter += 1

        # 更新基础位置信息
        self.base_pos[:] = self.root_states[:self.num_envs, 0:3]
        # 更新基础四元数信息
        self.base_quat[:] = self.root_states[:self.num_envs, 3:7]
        # 通过四元数逆旋转计算基础线速度
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 7:10])
        # 通过四元数逆旋转计算基础角速度
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 10:13])
        # 通过四元数逆旋转计算投影重力
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 更新足部速度和位置
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                               7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]

        # 调用物理步骤后的回调函数
        self._post_physics_step_callback()

        # 检查是否满足终止条件
        self.check_termination()
        # 计算奖励
        self.compute_reward()
        # 获取需要重置的环境ID
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # 根据环境ID重置环境
        self.reset_idx(env_ids)
        # 计算观测值
        self.compute_observations()

        # 更新动作和状态信息
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            # 如果启用了调试可视化，则绘制调试信息
            self._draw_debug_vis()

        # 执行无头渲染
        self._render_headless()

    def check_termination(self):
        """
        检查仿真环境是否满足终止条件。

        此方法负责检查三个可能的终止条件：
        - 接触力超过预设阈值。
        - 仿真的单个episode时长超过最大限制。
        - 如果启用，检查agent的身体高度是否低于终止阈值。

        Attributes:
            contact_forces: 仿真环境中的接触力张量。
            termination_contact_indices: 用于终止条件检查的接触力索引。
            episode_length_buf: 当前episode的长度缓冲区。
            cfg: 仿真环境的配置信息。
            root_states: 仿真环境中的根状态张量。
            measured_heights: 测量的高度值。

        Returns:
            None

        Examples:
            >>> env.check_termination()  # 检查仿真环境是否满足终止条件

        Note:
            - 终止条件的检查对于保持仿真环境的稳定性和安全性至关重要。
            - 根据配置，可能会启用不同的终止条件。
        """
        # 计算接触力是否超过阈值，并更新重置缓冲区
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        # 检查是否因为episode时长超过最大限制而需要重置
        self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length
        # 更新重置缓冲区，包含因时长超过限制的情况
        self.reset_buf |= self.time_out_buf
        # 如果启用了终止身体高度条件，检查并更新重置缓冲区
        if self.cfg.rewards.use_terminal_body_height:
            self.body_height_buf = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights,
                                              dim=1) < self.cfg.rewards.terminal_body_height
            self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)

    def reset_idx(self, env_ids):
        """
        重置指定环境的状态。
        此方法用于在仿真环境中重置指定ID的环境状态，包括机器人状态、缓冲区和额外信息。
        该方法还会根据配置随机化物理属性和地形特性。

        Attributes:
            env_ids (torch.Tensor): 需要重置的环境ID列表。

        Methods:
            _resample_commands: 重新采样控制命令。
            _call_train_eval: 对指定环境执行训练或评估模式下的方法。
            _randomize_dof_props: 随机化关节属性。
            _randomize_rigid_body_props: 随机化刚体属性。
            _reset_dofs: 重置关节状态。
            _reset_root_states: 重置根状态。

        Returns:
            None

        Examples:
            >>> env_ids = torch.tensor([0, 1, 2])
            >>> agent.reset_idx(env_ids)

        Note:
            - 该方法不返回任何值，但会直接修改仿真环境的内部状态。
            - 需要传入有效的环境ID，否则不会执行任何操作。
        """
        if len(env_ids) == 0:  # 如果没有指定环境ID，则不执行任何操作
            return

        # 重置机器人状态
        self._resample_commands(env_ids)  # 重新采样控制命令
        self._call_train_eval(self._randomize_dof_props, env_ids)  # 随机化关节属性
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._call_train_eval(self._randomize_rigid_body_props, env_ids)  # 随机化刚体属性
            self._call_train_eval(self.refresh_actor_rigid_shape_props, env_ids)  # 刷新刚体形状属性

        self._call_train_eval(self._reset_dofs, env_ids)  # 重置关节状态
        self._call_train_eval(self._reset_root_states, env_ids)  # 重置根状态

        # 重置缓冲区
        self.last_actions[env_ids] = 0.  # 重置最后的动作
        self.last_last_actions[env_ids] = 0.  # 重置上一次的动作
        self.last_dof_vel[env_ids] = 0.  # 重置关节速度
        self.feet_air_time[env_ids] = 0.  # 重置脚部空中时间
        self.episode_length_buf[env_ids] = 0  # 重置环境的episode长度
        self.reset_buf[env_ids] = 1  # 标记环境为需要重置

        # 填充额外信息
        train_env_ids = env_ids[env_ids < self.num_train_envs]  # 获取训练环境ID
        if len(train_env_ids) > 0:
            self.extras["train/episode"] = {}  # 初始化训练环境的额外信息
            for key in self.episode_sums.keys():  # 遍历所有键值
                self.extras["train/episode"]['rew_' + key] = torch.mean(
                    self.episode_sums[key][train_env_ids])  # 计算并存储平均奖励
                self.episode_sums[key][train_env_ids] = 0.  # 重置奖励和
        eval_env_ids = env_ids[env_ids >= self.num_train_envs]  # 获取评估环境ID
        if len(eval_env_ids) > 0:
            self.extras["eval/episode"] = {}  # 初始化评估环境的额外信息
            for key in self.episode_sums.keys():  # 遍历所有键值
                # 保存未设置的评估环境结果
                unset_eval_envs = eval_env_ids[self.episode_sums_eval[key][eval_env_ids] == -1]
                self.episode_sums_eval[key][unset_eval_envs] = self.episode_sums[key][unset_eval_envs]
                self.episode_sums[key][eval_env_ids] = 0.  # 重置奖励和

        # 记录额外的课程信息
        if self.cfg.terrain.curriculum:
            self.extras["train/episode"]["terrain_level"] = torch.mean(
                self.terrain_levels[:self.num_train_envs].float())  # 记录地形等级

        if self.cfg.commands.command_curriculum:  # 检查是否启用了命令课程功能
            # 记录环境命令分布
            self.extras["env_bins"] = torch.Tensor(self.env_command_bins)[:self.num_train_envs]  # 将环境命令分布存储到extras字典中
            # 记录命令参数的最小和最大值
            self.extras["train/episode"]["min_command_duration"] = torch.min(self.commands[:, 8])  # 记录命令持续时间的最小值
            self.extras["train/episode"]["max_command_duration"] = torch.max(self.commands[:, 8])  # 记录命令持续时间的最大值
            self.extras["train/episode"]["min_command_bound"] = torch.min(self.commands[:, 7])  # 记录命令边界的最小值
            self.extras["train/episode"]["max_command_bound"] = torch.max(self.commands[:, 7])  # 记录命令边界的最大值
            self.extras["train/episode"]["min_command_offset"] = torch.min(self.commands[:, 6])  # 记录命令偏移的最小值
            self.extras["train/episode"]["max_command_offset"] = torch.max(self.commands[:, 6])  # 记录命令偏移的最大值
            self.extras["train/episode"]["min_command_phase"] = torch.min(self.commands[:, 5])  # 记录命令相位的最小值
            self.extras["train/episode"]["max_command_phase"] = torch.max(self.commands[:, 5])  # 记录命令相位的最大值
            self.extras["train/episode"]["min_command_freq"] = torch.min(self.commands[:, 4])  # 记录命令频率的最小值
            self.extras["train/episode"]["max_command_freq"] = torch.max(self.commands[:, 4])  # 记录命令频率的最大值
            self.extras["train/episode"]["min_command_x_vel"] = torch.min(self.commands[:, 0])  # 记录x轴速度的最小值
            self.extras["train/episode"]["max_command_x_vel"] = torch.max(self.commands[:, 0])  # 记录x轴速度的最大值
            self.extras["train/episode"]["min_command_y_vel"] = torch.min(self.commands[:, 1])  # 记录y轴速度的最小值
            self.extras["train/episode"]["max_command_y_vel"] = torch.max(self.commands[:, 1])  # 记录y轴速度的最大值
            self.extras["train/episode"]["min_command_yaw_vel"] = torch.min(self.commands[:, 2])  # 记录偏航速度的最小值
            self.extras["train/episode"]["max_command_yaw_vel"] = torch.max(self.commands[:, 2])  # 记录偏航速度的最大值
            # 如果命令数量超过9，记录摆动高度的最小和最大值
            if self.cfg.commands.num_commands > 9:
                self.extras["train/episode"]["min_command_swing_height"] = torch.min(self.commands[:, 9])  # 记录摆动高度的最小值
                self.extras["train/episode"]["max_command_swing_height"] = torch.max(self.commands[:, 9])  # 记录摆动高度的最大值
            # 记录每个命令区域的权重总和
            for curriculum, category in zip(self.curricula, self.category_names):
                self.extras["train/episode"][f"command_area_{category}"] = np.sum(curriculum.weights) / \
                                                                           curriculum.weights.shape[
                                                                               0]  # 计算并记录每个命令区域的权重总和

            # 记录动作的最小和最大值
            self.extras["train/episode"]["min_action"] = torch.min(self.actions)  # 记录动作的最小值
            self.extras["train/episode"]["max_action"] = torch.max(self.actions)  # 记录动作的最大值

            # 更新课程分布信息
            self.extras["curriculum/distribution"] = {}  # 初始化课程分布信息字典
            for curriculum, category in zip(self.curricula, self.category_names):
                self.extras[f"curriculum/distribution"][f"weights_{category}"] = curriculum.weights  # 记录每个类别的权重
                self.extras[f"curriculum/distribution"][f"grid_{category}"] = curriculum.grid  # 记录每个类别的网格信息

        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf[:self.num_train_envs]  # 记录超时信息

        self.gait_indices[env_ids] = 0  # 重置步态索引

        for i in range(len(self.lag_buffer)):  # 清空延迟缓冲区
            self.lag_buffer[i][env_ids, :] = 0

    def set_idx_pose(self, env_ids, dof_pos, base_state):
        """
        设置指定环境ID的关节位置和基础状态。
        此方法用于在仿真环境中为指定的环境ID设置关节位置（dof_pos）和基础状态（base_state）。

        Attributes:
            env_ids (torch.Tensor): 需要设置状态的环境ID列表。
            dof_pos (torch.Tensor): 目标关节位置。
            base_state (torch.Tensor): 目标基础状态。

        Methods:
            gym.set_dof_state_tensor_indexed: 设置指定索引的关节状态。
            gym.set_actor_root_state_tensor_indexed: 设置指定索引的演员根状态。

        Returns:
            None

        Examples:
            >>> env_ids = torch.tensor([0, 1, 2])
            >>> dof_pos = torch.tensor([...])
            >>> base_state = torch.tensor([...])
            >>> agent.set_idx_pose(env_ids, dof_pos, base_state)

        Note:
            - 如果env_ids为空，则不执行任何操作。
            - dof_pos和base_state必须是适当的张量，其尺寸与env_ids匹配。
        """
        if len(env_ids) == 0:  # 如果没有指定环境ID，则不执行任何操作
            return

        env_ids_int32 = env_ids.to(dtype=torch.int32).to(self.device)  # 将环境ID转换为int32类型，并移动到指定设备

        # 设置关节状态
        if dof_pos is not None:  # 如果提供了目标关节位置
            self.dof_pos[env_ids] = dof_pos  # 更新关节位置
            self.dof_vel[env_ids] = 0.  # 将关节速度设置为0

            # 使用Gym API更新指定环境的关节状态
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # 设置基础状态
        self.root_states[env_ids] = base_state.to(self.device)  # 更新基础状态，并确保其在正确的设备上

        # 使用Gym API更新指定环境的基础状态
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def compute_reward(self):
        """
        计算并更新奖励缓冲区。
        此方法遍历所有定义的奖励函数，计算奖励，并根据配置更新正奖励和负奖励缓冲区。
        同时，它还会更新每个奖励名称对应的累计奖励和命令相关的统计信息。

        Attributes:
            None

        Methods:
            reward_functions: 定义的奖励函数列表。
            _reward_termination: 终止奖励的计算函数。

        Returns:
            None

        Examples:
            >>> agent.compute_reward()

        Note:
            - 奖励缓冲区会根据配置项`only_positive_rewards`和`only_positive_rewards_ji22_style`进行调整。
            - 该方法同时更新了奖励的累计和和命令相关的统计信息。
        """
        self.rew_buf[:] = 0.  # 重置奖励缓冲区
        self.rew_buf_pos[:] = 0.  # 重置正奖励缓冲区
        self.rew_buf_neg[:] = 0.  # 重置负奖励缓冲区
        for i in range(len(self.reward_functions)):  # 遍历所有定义的奖励函数
            name = self.reward_names[i]  # 获取奖励名称
            rew = self.reward_functions[i]() * self.reward_scales[name]  # 计算奖励值并应用奖励缩放
            self.rew_buf += rew  # 更新总奖励缓冲区
            if torch.sum(rew) >= 0:  # 如果奖励为正
                self.rew_buf_pos += rew  # 更新正奖励缓冲区
            elif torch.sum(rew) <= 0:  # 如果奖励为负
                self.rew_buf_neg += rew  # 更新负奖励缓冲区
            self.episode_sums[name] += rew  # 更新奖励名称对应的累计奖励
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:  # 特殊处理某些奖励
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew  # 更新命令相关的统计信息
        if self.cfg.rewards.only_positive_rewards:  # 如果只考虑正奖励
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        elif self.cfg.rewards.only_positive_rewards_ji22_style:  # 如果使用特殊的正奖励处理方式
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)
        self.episode_sums["total"] += self.rew_buf  # 更新总奖励的累计和
        # 添加终止奖励
        if "termination" in self.reward_scales:  # 如果定义了终止奖励
            rew = self.reward_container._reward_termination() * self.reward_scales["termination"]  # 计算终止奖励
            self.rew_buf += rew  # 更新奖励缓冲区
            self.episode_sums["termination"] += rew  # 更新终止奖励的累计和
            self.command_sums["termination"] += rew  # 更新命令相关的统计信息

        # 更新命令相关的统计信息
        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]  # 线速度原始值
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]  # 角速度原始值
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2  # 线速度残差
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2  # 角速度残差
        self.command_sums["ep_timesteps"] += 1  # 更新时间步数

    def compute_observations(self):
        """
        计算观测值，包括普通观测值和特权观测值。

        此方法根据配置选项，组合不同的观测信息，包括重力投影、关节位置、关节速度、动作、命令等，并对它们进行缩放处理。
        同时，还会根据配置添加噪声、计算特权观测值。

        Attributes:
            obs_buf (torch.Tensor): 普通观测值缓冲区。
            privileged_obs_buf (torch.Tensor): 特权观测值缓冲区。
            next_privileged_obs_buf (torch.Tensor): 下一步的特权观测值缓冲区。

        Methods:
            - 组合普通观测值。
            - 添加噪声（如果需要）。
            - 计算特权观测值。

        Returns:
            None

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> agent.compute_observations()  # 计算观测值

        Note:
            - 特权观测值用于训练时提供额外信息，但在评估或实际部署时不可用。
            - 观测值的组合和缩放取决于配置文件中的设置。
        """
        self.obs_buf = torch.cat((self.projected_gravity,
                                  (self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:,
                                                                             :self.num_actuated_dof]) * self.obs_scales.dof_pos,
                                  self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel,
                                  self.actions
                                  ), dim=-1)
        # if self.cfg.env.observe_command and not self.cfg.env.observe_height_command:
        #     self.obs_buf = torch.cat((self.projected_gravity,
        #                               self.commands[:, :3] * self.commands_scale,
        #                               (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
        #                               self.dof_vel * self.obs_scales.dof_vel,
        #                               self.actions
        #                               ), dim=-1)

        if self.cfg.env.observe_command:
            self.obs_buf = torch.cat((self.projected_gravity,
                                      self.commands * self.commands_scale,
                                      (self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:,
                                                                                 :self.num_actuated_dof]) * self.obs_scales.dof_pos,
                                      self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel,
                                      self.actions
                                      ), dim=-1)

        if self.cfg.env.observe_two_prev_actions:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.last_actions), dim=-1)

        if self.cfg.env.observe_timing_parameter:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.gait_indices.unsqueeze(1)), dim=-1)

        if self.cfg.env.observe_clock_inputs:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.clock_inputs), dim=-1)

        # if self.cfg.env.observe_desired_contact_states:
        #     self.obs_buf = torch.cat((self.obs_buf,
        #                               self.desired_contact_states), dim=-1)

        if self.cfg.env.observe_vel:
            if self.cfg.commands.global_reference:
                self.obs_buf = torch.cat((self.root_states[:self.num_envs, 7:10] * self.obs_scales.lin_vel,
                                          self.base_ang_vel * self.obs_scales.ang_vel,
                                          self.obs_buf), dim=-1)
            else:
                self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                          self.base_ang_vel * self.obs_scales.ang_vel,
                                          self.obs_buf), dim=-1)

        if self.cfg.env.observe_only_ang_vel:
            self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                      self.obs_buf), dim=-1)

        if self.cfg.env.observe_only_lin_vel:
            self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                      self.obs_buf), dim=-1)

        if self.cfg.env.observe_yaw:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
            # heading_error = torch.clip(0.5 * wrap_to_pi(heading), -1., 1.).unsqueeze(1)
            self.obs_buf = torch.cat((self.obs_buf,
                                      heading), dim=-1)

        if self.cfg.env.observe_contact_states:
            self.obs_buf = torch.cat((self.obs_buf, (self.contact_forces[:, self.feet_indices, 2] > 1.).view(
                self.num_envs,
                -1) * 1.0), dim=1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # build privileged obs

        self.privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)
        self.next_privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)

        if self.cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.friction_coeffs[:, 0].unsqueeze(
                                                     1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.friction_coeffs[:, 0].unsqueeze(
                                                          1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_ground_friction:
            self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
            ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(
                self.cfg.normalization.ground_friction_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.ground_friction_coeffs.unsqueeze(
                                                     1) - ground_friction_coeffs_shift) * ground_friction_coeffs_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.ground_friction_coeffs.unsqueeze(
                                                          1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.normalization.restitution_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.restitutions[:, 0].unsqueeze(
                                                     1) - restitutions_shift) * restitutions_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.restitutions[:, 0].unsqueeze(
                                                          1) - restitutions_shift) * restitutions_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_base_mass:
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.normalization.added_mass_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_com_displacement:
            com_displacements_scale, com_displacements_shift = get_scale_shift(
                self.cfg.normalization.com_displacement_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (
                                                         self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (
                                                              self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_motor_strength:
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.normalization.motor_strength_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (
                                                         self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (
                                                              self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_motor_offset:
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.normalization.motor_offset_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (
                                                         self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                      (
                                                              self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_body_height:
            body_height_scale, body_height_shift = get_scale_shift(self.cfg.normalization.body_height_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 ((self.root_states[:self.num_envs, 2]).view(
                                                     self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      ((self.root_states[:self.num_envs, 2]).view(
                                                          self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_body_velocity:
            body_velocity_scale, body_velocity_shift = get_scale_shift(self.cfg.normalization.body_velocity_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 ((self.base_lin_vel).view(self.num_envs,
                                                                           -1) - body_velocity_shift) * body_velocity_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      ((self.base_lin_vel).view(self.num_envs,
                                                                                -1) - body_velocity_shift) * body_velocity_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_gravity:
            gravity_scale, gravity_shift = get_scale_shift(self.cfg.normalization.gravity_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.gravities - gravity_shift) / gravity_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.gravities - gravity_shift) / gravity_scale), dim=1)

        if self.cfg.env.priv_observe_clock_inputs:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.clock_inputs), dim=-1)

        if self.cfg.env.priv_observe_desired_contact_states:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.desired_contact_states), dim=-1)

        assert self.privileged_obs_buf.shape[
                   1] == self.cfg.env.num_privileged_obs, f"num_privileged_obs ({self.cfg.env.num_privileged_obs}) != the number of privileged observations ({self.privileged_obs_buf.shape[1]}), you will discard data from the student!"

    def create_sim(self):
        """
        创建仿真环境及地形。

        根据配置文件中的地形类型，创建相应的仿真环境和地形。支持的地形类型包括平面（plane）、高度场（heightfield）和三角网格（trimesh）。
        如果配置了评估环境，同时为训练和评估环境创建地形实例。

        Attributes:
            up_axis_idx (int): 重力方向的索引，2代表z轴，1代表y轴。
            sim (gym.Env): 创建的仿真环境实例。
            terrain (Terrain): 地形实例。

        Methods:
            _create_ground_plane(): 创建平面地形。
            _create_heightfield(): 创建高度场地形。
            _create_trimesh(): 创建三角网格地形。
            _create_envs(): 创建环境实例。

        Raises:
            ValueError: 如果地形网格类型不被识别。

        Examples:
            >>> simulator = YourSimulatorClass()  # 仿真器类的实例化
            >>> simulator.create_sim()  # 创建仿真环境和地形

        Note:
            - 支持的地形类型有 'plane', 'heightfield', 'trimesh'。
            - 如果配置了评估环境，将同时为训练和评估环境创建地形实例。
        """
        self.up_axis_idx = 2  # 设置重力方向的索引，2代表z轴，1代表y轴
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)  # 创建仿真环境实例

        mesh_type = self.cfg.terrain.mesh_type  # 获取地形的网格类型
        if mesh_type in ['heightfield', 'trimesh']:
            if self.eval_cfg is not None:
                # 为训练和评估环境创建地形实例
                self.terrain = Terrain(self.cfg.terrain, self.num_train_envs, self.eval_cfg.terrain, self.num_eval_envs)
            else:
                # 仅为训练环境创建地形实例
                self.terrain = Terrain(self.cfg.terrain, self.num_train_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()  # 创建平面地形
        elif mesh_type == 'heightfield':
            self._create_heightfield()  # 创建高度场地形
        elif mesh_type == 'trimesh':
            self._create_trimesh()  # 创建三角网格地形
        elif mesh_type is not None:
            # 如果地形网格类型不被识别，抛出错误
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

        self._create_envs()  # 创建环境实例

    def set_camera(self, position, lookat):
        """
        设置仿真环境中的相机位置和朝向。

        通过指定相机的位置和它所指向的目标点来设置相机的视角。这对于观察仿真环境中的特定场景非常有用。

        Attributes:
            position (list of float): 相机的位置，格式为 [x, y, z]。
            lookat (list of float): 相机目标点的位置，格式为 [x, y, z]。

        Methods:
            gymapi.Vec3: 构造三维向量。
            gym.viewer_camera_look_at: 设置相机的位置和朝向。

        Examples:
            >>> simulator = YourSimulatorClass()  # 仿真器类的实例化
            >>> simulator.set_camera([0, 1, 2], [3, 4, 5])  # 设置相机位置和朝向

        Note:
            - position 和 lookat 都是包含三个浮点数的列表，分别代表x、y、z坐标。
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])  # 构造相机位置向量
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])  # 构造相机目标点向量
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)  # 设置相机位置和朝向

    def set_main_agent_pose(self, loc, quat):
        """
        设置主代理的位置和姿态。

        通过指定位置和姿态（四元数）来更新仿真环境中主代理的根状态。这对于初始化代理的起始位置或在仿真过程中改变其位置非常有用。

        Attributes:
            loc (list of float): 代理的位置，格式为 [x, y, z]。
            quat (list of float): 代理的姿态（四元数），格式为 [x, y, z, w]。

        Methods:
            torch.Tensor: 创建一个新的张量。
            gym.set_actor_root_state_tensor: 更新代理的根状态张量。

        Examples:
            >>> simulator = YourSimulatorClass()  # 仿真器类的实例化
            >>> simulator.set_main_agent_pose([0, 1, 2], [1, 0, 0, 0])  # 设置主代理的位置和姿态

        Note:
            - loc 是包含三个浮点数的列表，代表代理的x、y、z坐标。
            - quat 是包含四个浮点数的列表，代表代理的姿态四元数。
        """
        self.root_states[0, 0:3] = torch.Tensor(loc)  # 设置代理位置
        self.root_states[0, 3:7] = torch.Tensor(quat)  # 设置代理姿态
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))  # 更新代理的根状态张量

    # ------------- Callbacks --------------
    def _call_train_eval(self, func, env_ids):
        """
        对训练和评估环境执行指定函数，并合并结果。

        此方法将环境ID分为训练和评估两组，分别对它们执行给定的函数。如果两组环境都有返回值，则将这些值沿最后一个维度合并。

        Attributes:
            func (callable): 需要执行的函数，该函数应接受环境ID列表和配置对象作为输入。
            env_ids (torch.Tensor): 包含环境ID的张量。

        Returns:
            torch.Tensor: 如果训练和评估环境都有返回值，则返回合并后的张量；否则，返回其中一个非空的返回值。

        Examples:
            >>> simulator = YourSimulatorClass()  # 仿真器类的实例化
            >>> env_ids = torch.tensor([0, 1, 2, 3, 4])  # 假设有5个环境，其中3个用于训练，2个用于评估
            >>> result = simulator._call_train_eval(your_function, env_ids)  # 对这些环境执行your_function函数

        Note:
            - func函数需要能够接受两个参数：环境ID列表和配置对象。
            - 此方法用于处理同时存在训练和评估环境的情况。
        """
        env_ids_train = env_ids[env_ids < self.num_train_envs]  # 分离出训练环境的ID
        env_ids_eval = env_ids[env_ids >= self.num_train_envs]  # 分离出评估环境的ID

        ret, ret_eval = None, None  # 初始化返回值

        if len(env_ids_train) > 0:
            ret = func(env_ids_train, self.cfg)  # 对训练环境执行函数
        if len(env_ids_eval) > 0:
            ret_eval = func(env_ids_eval, self.eval_cfg)  # 对评估环境执行函数
            if ret is not None and ret_eval is not None:  # 如果两组环境都有返回值，则合并这些值
                ret = torch.cat((ret, ret_eval), axis=-1)

        return ret  # 返回最终结果

    def _randomize_gravity(self, external_force=None):
        """
        随机化或设置仿真环境中的重力。

        如果提供了外部力，则使用该力作为新的重力向量。否则，如果配置了重力随机化，将在指定范围内随机生成新的重力向量。
        更新后的重力向量将被标准化并应用到仿真环境中。

        Attributes:
            external_force (torch.Tensor, optional): 指定的外部力向量。默认为None。

        Methods:
            torch.rand: 生成随机数张量。
            torch.unsqueeze: 增加张量的一个维度。
            torch.norm: 计算张量的范数。
            gym.get_sim_params: 获取仿真参数。
            gym.set_sim_params: 设置仿真参数。

        Examples:
            >>> simulator = YourSimulatorClass()  # 仿真器类的实例化
            >>> simulator._randomize_gravity(torch.Tensor([0, 0, -9.8]))  # 设置特定的重力向量

        Note:
            - 如果没有提供external_force且配置了重力随机化，将在配置的gravity_range范围内随机选择重力。
        """
        if external_force is not None:
            # 如果提供了外部力，则使用该力作为新的重力向量
            self.gravities[:, :] = external_force.unsqueeze(0)
        elif self.cfg.domain_rand.randomize_gravity:
            # 如果配置了重力随机化，随机生成新的重力向量
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity
            self.gravities[:, :] = external_force.unsqueeze(0)

        sim_params = self.gym.get_sim_params(self.sim)
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device)
        # 更新重力向量，标准化后应用到仿真环境
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)

    def _process_rigid_shape_props(self, props, env_id):
        """
        处理并更新刚体形状的物理属性。

        遍历每个刚体形状，根据环境ID更新其摩擦系数和恢复系数。这允许在不同环境中对物体的物理行为进行定制。

        Attributes:
            props (list of gymapi.RigidShapeProperties): 刚体形状属性的列表。
            env_id (int): 环境的ID，用于索引特定环境的物理属性值。

        Returns:
            list of gymapi.RigidShapeProperties: 更新后的刚体形状属性列表。

        Examples:
            >>> simulator = YourSimulatorClass()  # 仿真器类的实例化
            >>> props = [gymapi.RigidShapeProperties(), gymapi.RigidShapeProperties()]  # 创建刚体形状属性列表
            >>> updated_props = simulator._process_rigid_shape_props(props, 0)  # 更新刚体形状的物理属性

        Note:
            - 摩擦系数和恢复系数是影响物体碰撞后行为的重要物理属性。
        """
        for s in range(len(props)):
            props[s].friction = self.friction_coeffs[env_id, 0]  # 更新摩擦系数
            props[s].restitution = self.restitutions[env_id, 0]  # 更新恢复系数

        return props  # 返回更新后的属性列表

    def _process_dof_props(self, props, env_id):
        """
        处理和设置关节属性。

        为第一个环境初始化关节的位置限制、速度限制和扭矩限制。如果是第一个环境（env_id == 0），则基于提供的属性字典更新关节的限制。
        此外，根据配置调整关节位置的软限制。

        Attributes:
            props (dict): 包含关节属性的字典，键包括"lower", "upper", "velocity", "effort"。
            env_id (int): 环境的ID。

        Returns:
            dict: 更新后的关节属性字典。

        Examples:
            >>> simulator = YourSimulatorClass()  # 仿真器类的实例化
            >>> props = {"lower": torch.tensor([-1.0]), "upper": torch.tensor([1.0]), "velocity": torch.tensor([0.5]), "effort": torch.tensor([1.0])}
            >>> updated_props = simulator._process_dof_props(props, 0)  # 更新第一个环境的关节属性

        Note:
            - 仅在env_id为0时更新关节限制，以避免重复设置相同的限制。
            - 关节位置的软限制是通过配置中的`soft_dof_pos_limit`参数调整的。
        """
        if env_id == 0:
            # 初始化关节的位置限制、速度限制和扭矩限制张量
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                # 更新关节的位置限制、速度限制和扭矩限制
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # 调整关节位置的软限制
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        return props

    def _randomize_rigid_body_props(self, env_ids, cfg):
        """
        随机化刚体属性。

        根据配置，为指定的环境ID随机化质量、质心偏移、摩擦系数和恢复系数。这些随机化的属性有助于提高训练的泛化能力。

        Attributes:
            env_ids (torch.Tensor): 需要随机化属性的环境ID列表。
            cfg (Config): 包含随机化参数的配置对象。

        Methods:
            torch.rand: 生成随机数张量。

        Examples:
            >>> simulator = YourSimulatorClass()  # 仿真器类的实例化
            >>> env_ids = torch.tensor([0, 1, 2])  # 指定需要随机化属性的环境ID
            >>> simulator._randomize_rigid_body_props(env_ids, config)  # 根据配置随机化刚体属性

        Note:
            - 随机化的属性包括质量、质心偏移、摩擦系数和恢复系数。
            - 这些属性的随机化范围由配置对象中的domain_rand部分指定。
        """
        if cfg.domain_rand.randomize_base_mass:
            # 随机化基础质量
            min_payload, max_payload = cfg.domain_rand.added_mass_range
            self.payloads[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (max_payload - min_payload) + min_payload

        if cfg.domain_rand.randomize_com_displacement:
            # 随机化质心偏移
            min_com_displacement, max_com_displacement = cfg.domain_rand.com_displacement_range
            self.com_displacements[env_ids, :] = torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
                                                            requires_grad=False) * (
                                                             max_com_displacement - min_com_displacement) + min_com_displacement

        if cfg.domain_rand.randomize_friction:
            # 随机化摩擦系数
            min_friction, max_friction = cfg.domain_rand.friction_range
            self.friction_coeffs[env_ids, :] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                          requires_grad=False) * (
                                                           max_friction - min_friction) + min_friction

        if cfg.domain_rand.randomize_restitution:
            # 随机化恢复系数
            min_restitution, max_restitution = cfg.domain_rand.restitution_range
            self.restitutions[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                    requires_grad=False) * (
                                                     max_restitution - min_restitution) + min_restitution

    def refresh_actor_rigid_shape_props(self, env_ids, cfg):
        """
        刷新指定环境中的演员刚体形状属性。

        遍历给定的环境ID列表，更新每个环境中第一个演员的刚体形状属性，包括摩擦系数和恢复系数。这些属性的值从之前随机化或设置的属性中获取。

        Attributes:
            env_ids (torch.Tensor): 需要更新刚体形状属性的环境ID列表。
            cfg (Config): 配置对象，虽然此函数中未直接使用，但保留以便未来可能的配置需求。

        Methods:
            gym.get_actor_rigid_shape_properties: 获取演员的刚体形状属性。
            gym.set_actor_rigid_shape_properties: 设置演员的刚体形状属性。

        Examples:
            >>> simulator = YourSimulatorClass()  # 仿真器类的实例化
            >>> env_ids = torch.tensor([0, 1, 2])  # 指定需要更新刚体形状属性的环境ID
            >>> simulator.refresh_actor_rigid_shape_props(env_ids, config)  # 更新指定环境中演员的刚体形状属性

        Note:
            - 此方法用于在随机化或手动设置刚体属性后，将这些属性应用到仿真环境中的演员上。
        """
        for env_id in env_ids:
            # 获取指定环境中第一个演员的刚体形状属性
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(self.num_dof):
                # 更新摩擦系数和恢复系数
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitutions[env_id, 0]

            # 将更新后的刚体形状属性应用到演员上
            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _randomize_dof_props(self, env_ids, cfg):
        """
        根据配置随机化关节属性，包括电机强度、电机偏移、比例增益因子和微分增益因子。

        根据域随机化配置，此方法可以随机化电机的强度、偏移量以及控制器的Kp和Kd因子。

        Attributes:
            env_ids (list[int]): 需要随机化属性的环境ID列表。
            cfg (object): 包含域随机化设置的配置对象。

        Methods:
            torch.rand: 生成指定形状的随机数张量。
            unsqueeze: 增加张量的一个维度。

        Examples:
            >>> env_ids = [0, 1, 2]
            >>> cfg = cfg.domain_rand
            >>> _randomize_dof_props(env_ids, cfg)

        Note:
            - 域随机化有助于提高模型的泛化能力，通过在训练过程中引入物理参数的变化来实现。
            - 本方法支持对多个环境进行批量操作。
        """
        # 如果配置中指定了随机化电机强度
        if cfg.domain_rand.randomize_motor_strength:
            # 从配置中获取电机强度的最小值和最大值
            min_strength, max_strength = cfg.domain_rand.motor_strength_range
            # 随机生成新的电机强度，并赋值给相应的环境
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                          requires_grad=False).unsqueeze(1) * (
                                                       max_strength - min_strength) + min_strength
        # 如果配置中指定了随机化电机偏移
        if cfg.domain_rand.randomize_motor_offset:
            # 从配置中获取电机偏移的最小值和最大值
            min_offset, max_offset = cfg.domain_rand.motor_offset_range
            # 随机生成新的电机偏移，并赋值给相应的环境
            self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float,
                                                        device=self.device, requires_grad=False) * (
                                                     max_offset - min_offset) + min_offset
        # 如果配置中指定了随机化Kp因子
        if cfg.domain_rand.randomize_Kp_factor:
            # 从配置中获取Kp因子的最小值和最大值
            min_Kp_factor, max_Kp_factor = cfg.domain_rand.Kp_factor_range
            # 随机生成新的Kp因子，并赋值给相应的环境
            self.Kp_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kp_factor - min_Kp_factor) + min_Kp_factor
        # 如果配置中指定了随机化Kd因子
        if cfg.domain_rand.randomize_Kd_factor:
            # 从配置中获取Kd因子的最小值和最大值
            min_Kd_factor, max_Kd_factor = cfg.domain_rand.Kd_factor_range
            # 随机生成新的Kd因子，并赋值给相应的环境
            self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kd_factor - min_Kd_factor) + min_Kd_factor

    def _process_rigid_body_props(self, props, env_id):
        """
        处理刚体属性的调整。
        根据环境ID调整刚体的质量和质心位置，以模拟不同的载荷和平衡条件。

        Attributes:
            default_body_mass (float): 默认的刚体质量。
            payloads (list): 每个环境对应的额外载荷质量。
            com_displacements (numpy.ndarray): 每个环境对应的质心偏移量。

        Args:
            props (list): 包含刚体属性的列表。
            env_id (int): 当前环境的ID。

        Returns:
            list: 更新后的刚体属性列表。

        Examples:
            >>> props = [RigidBodyProperties(mass=1.0, com=gymapi.Vec3(0, 0, 0))]
            >>> env_id = 0
            >>> updated_props = _process_rigid_body_props(props, env_id)
            >>> print(updated_props[0].mass)
            >>> print(updated_props[0].com.x, updated_props[0].com.y, updated_props[0].com.z)

        Note:
            - 该方法假设`props`列表中至少有一个元素，并且`env_id`在`payloads`和`com_displacements`的索引范围内。
            - 质量的调整是通过在默认质量上加上对应环境的额外载荷来实现的。
            - 质心(com)的调整是通过设置新的gymapi.Vec3对象来实现的，其值由`com_displacements`数组提供。
        """
        # 保存默认的刚体质量
        self.default_body_mass = props[0].mass
        # 根据环境ID调整第一个刚体的质量
        props[0].mass = self.default_body_mass + self.payloads[env_id]
        # 根据环境ID调整第一个刚体的质心位置
        props[0].com = gymapi.Vec3(self.com_displacements[env_id, 0],
                                   self.com_displacements[env_id, 1],
                                   self.com_displacements[env_id, 2])
        # 返回更新后的刚体属性列表
        return props

    def _post_physics_step_callback(self):
        """
        物理步骤后的回调函数。

        在每个物理步骤后执行的一系列操作，包括防止机器人掉落边缘的传送、命令的重采样、地形高度的测量、机器人的推动、关节属性的随机化以及重力的随机化。

        Methods:
            _call_train_eval: 对训练和评估环境执行指定函数。
            _teleport_robots: 传送机器人以防止掉落。
            _resample_commands: 重采样控制命令。
            _step_contact_targets: 更新接触目标。
            _get_heights: 测量地形高度。
            _push_robots: 推动机器人。
            _randomize_dof_props: 随机化关节属性。
            _randomize_gravity: 随机化重力。
            _randomize_rigid_body_props: 随机化刚体属性。
            refresh_actor_rigid_shape_props: 刷新演员刚体形状属性。

        Examples:
            >>> simulator = YourSimulatorClass()  # 仿真器类的实例化
            >>> simulator._post_physics_step_callback()  # 执行物理步骤后的回调函数

        Note:
            - 此回调函数在仿真的每个物理步骤后自动调用，用于执行仿真环境的维护和更新操作。
        """
        # 防止机器人掉落边缘的传送
        self._call_train_eval(self._teleport_robots, torch.arange(self.num_envs, device=self.device))

        # 重采样控制命令
        sample_interval = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % sample_interval == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._step_contact_targets()

        # 测量地形高度
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights(torch.arange(self.num_envs, device=self.device), self.cfg)

        # 推动机器人
        self._call_train_eval(self._push_robots, torch.arange(self.num_envs, device=self.device))

        # 随机化关节属性
        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval) == 0).nonzero(
            as_tuple=False).flatten()
        self._call_train_eval(self._randomize_dof_props, env_ids)

        # 随机化重力
        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        if int(self.common_step_counter - self.cfg.domain_rand.gravity_rand_duration) % int(
                self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity(torch.tensor([0, 0, 0]))

        # 如果配置了在开始后随机化刚体属性，则执行随机化并刷新演员刚体形状属性
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._call_train_eval(self._randomize_rigid_body_props, env_ids)
            self._call_train_eval(self.refresh_actor_rigid_shape_props, env_ids)

    def _resample_commands(self, env_ids):

        if len(env_ids) == 0: return

        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.cfg.env.max_episode_length, timesteps)

        # update curricula based on terminated environment bins and categories
        for i, (category, curriculum) in enumerate(zip(self.category_names, self.curricula)):
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            elif len(env_ids_in_category) == 0:
                continue

            env_ids_in_category = env_ids[env_ids_in_category]

            task_rewards, success_thresholds = [], []
            for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force",
                        "tracking_contacts_shaped_vel"]:
                if key in self.command_sums.keys():
                    task_rewards.append(self.command_sums[key][env_ids_in_category] / ep_len)
                    success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])

            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]
            if len(success_thresholds) > 0:
                curriculum.update

        # assign resampled environments to new categories
        random_env_floats = torch.rand(len(env_ids), device=self.device)
        probability_per_category = 1. / len(self.category_names)
        category_env_ids = [env_ids[torch.logical_and(probability_per_category * i <= random_env_floats,
                                                      random_env_floats < probability_per_category * (i + 1))] for i in
                            range(len(self.category_names))]

        # sample from new category curricula
        for i, (category, env_ids_in_category, curriculum) in enumerate(
                zip(self.category_names, category_env_ids, self.curricula)):

            batch_size = len(env_ids_in_category)
            if batch_size == 0: continue

            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)

            self.env_command_bins[env_ids_in_category.cpu().numpy()] = new_bin_inds
            self.env_command_categories[env_ids_in_category.cpu().numpy()] = i

            self.commands[env_ids_in_category, :] = torch.Tensor(new_commands[:, :self.cfg.commands.num_commands]).to(
                self.device)

        if self.cfg.commands.num_commands > 5:
            if self.cfg.commands.gaitwise_curricula:
                for i, (category, env_ids_in_category) in enumerate(zip(self.category_names, category_env_ids)):
                    if category == "pronk":  # pronking
                        self.commands[env_ids_in_category, 5] = (self.commands[env_ids_in_category, 5] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 6] = (self.commands[env_ids_in_category, 6] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 7] = (self.commands[env_ids_in_category, 7] / 2 - 0.25) % 1
                    elif category == "trot":  # trotting
                        self.commands[env_ids_in_category, 5] = self.commands[env_ids_in_category, 5] / 2 + 0.25
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "pace":  # pacing
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = self.commands[env_ids_in_category, 6] / 2 + 0.25
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "bound":  # bounding
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = self.commands[env_ids_in_category, 7] / 2 + 0.25

            elif self.cfg.commands.exclusive_phase_offset:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                trotting_envs = env_ids[random_env_floats < 0.34]
                pacing_envs = env_ids[torch.logical_and(0.34 <= random_env_floats, random_env_floats < 0.67)]
                bounding_envs = env_ids[0.67 <= random_env_floats]
                self.commands[pacing_envs, 5] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[trotting_envs, 6] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 7] = 0

            elif self.cfg.commands.balance_gait_distribution:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                pronking_envs = env_ids[random_env_floats <= 0.25]
                trotting_envs = env_ids[torch.logical_and(0.25 <= random_env_floats, random_env_floats < 0.50)]
                pacing_envs = env_ids[torch.logical_and(0.50 <= random_env_floats, random_env_floats < 0.75)]
                bounding_envs = env_ids[0.75 <= random_env_floats]
                self.commands[pronking_envs, 5] = (self.commands[pronking_envs, 5] / 2 - 0.25) % 1
                self.commands[pronking_envs, 6] = (self.commands[pronking_envs, 6] / 2 - 0.25) % 1
                self.commands[pronking_envs, 7] = (self.commands[pronking_envs, 7] / 2 - 0.25) % 1
                self.commands[trotting_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 5] = 0
                self.commands[pacing_envs, 7] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 5] = self.commands[trotting_envs, 5] / 2 + 0.25
                self.commands[pacing_envs, 6] = self.commands[pacing_envs, 6] / 2 + 0.25
                self.commands[bounding_envs, 7] = self.commands[bounding_envs, 7] / 2 + 0.25

            if self.cfg.commands.binary_phases:
                self.commands[env_ids, 5] = (torch.round(2 * self.commands[env_ids, 5])) / 2.0 % 1
                self.commands[env_ids, 6] = (torch.round(2 * self.commands[env_ids, 6])) / 2.0 % 1
                self.commands[env_ids, 7] = (torch.round(2 * self.commands[env_ids, 7])) / 2.0 % 1

        # setting the smaller commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.

    def _step_contact_targets(self):
        """
        根据命令计算步态目标，包括脚部指标、时钟输入和期望接触状态。

        此方法根据给定的命令（如频率、相位、偏移量等），计算每只脚的步态指标，并据此生成时钟输入和期望的接触状态。
        这些计算支持不同的步态模式，并可通过配置文件进行调整。

        Attributes:
            gait_indices (torch.Tensor): 步态指数，表示当前步态周期中的位置。
            foot_indices (torch.Tensor): 每只脚的步态指标。
            clock_inputs (torch.Tensor): 基于步态指标的时钟输入。
            doubletime_clock_inputs (torch.Tensor): 倍频时钟输入。
            halftime_clock_inputs (torch.Tensor): 半频时钟输入。
            desired_contact_states (torch.Tensor): 期望的接触状态。

        Methods:
            - 计算步态指数和脚部指标。
            - 生成时钟输入和期望接触状态。
            - 根据配置调整步态模式和接触状态。

        Returns:
            None

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> agent._step_contact_targets()  # 计算步态目标

        Note:
            - 此方法依赖于环境配置和给定的命令。
            - 时钟输入和期望接触状态用于控制和奖励计算。
        """
        # 如果配置中指定观察步态命令
        if self.cfg.env.observe_gait_commands:
            # 从命令中提取步态相关参数
            frequencies = self.commands[:, 4]
            phases = self.commands[:, 5]
            offsets = self.commands[:, 6]
            bounds = self.commands[:, 7]
            durations = self.commands[:, 8]
            # 更新步态指数
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

            # 根据配置选择步态偏移量计算方法
            if self.cfg.commands.pacing_offset:
                foot_indices = [self.gait_indices + phases + offsets + bounds,
                                self.gait_indices + bounds,
                                self.gait_indices + offsets,
                                self.gait_indices + phases]
            else:
                foot_indices = [self.gait_indices + phases + offsets + bounds,
                                self.gait_indices + offsets,
                                self.gait_indices + bounds,
                                self.gait_indices + phases]

            # 计算脚部指标
            self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

            # 根据脚部指标和持续时间计算接触状态
            for idxs in foot_indices:
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                            0.5 / (1 - durations[swing_idxs]))

            # 生成时钟输入
            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

            # 生成倍频和半频时钟输入
            self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
            self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
            self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
            self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

            self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
            self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
            self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
            self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

            # 使用von Mises分布计算期望接触状态的平滑过渡
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

            smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                        1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) + smoothing_cdf_start(
                torch.remainder(foot_indices[0], 1.0) - 1) * (1 - smoothing_cdf_start(
                torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
            smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                        1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) + smoothing_cdf_start(
                torch.remainder(foot_indices[1], 1.0) - 1) * (1 - smoothing_cdf_start(
                torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                        1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) + smoothing_cdf_start(
                torch.remainder(foot_indices[2], 1.0) - 1) * (1 - smoothing_cdf_start(
                torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                        1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) + smoothing_cdf_start(
                torch.remainder(foot_indices[3], 1.0) - 1) * (1 - smoothing_cdf_start(
                torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

            # 设置期望接触状态
            self.desired_contact_states[:, 0] = smoothing_multiplier_FL
            self.desired_contact_states[:, 1] = smoothing_multiplier_FR
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR

        # 如果命令数量超过9，设置期望的脚摆高度
        if self.cfg.commands.num_commands > 9:
            self.desired_footswing_height = self.commands[:, 9]

    def _compute_torques(self, actions):
        """
        根据给定的动作计算关节扭矩。

        此方法根据配置的控制类型（如PD控制器或自定义的执行器网络）计算关节扭矩。动作首先被缩放并应用到关节目标位置，然后根据关节位置和速度误差计算扭矩。

        Attributes:
            actions (torch.Tensor): 输入动作张量，形状为(batch_size, action_dim)。

        Returns:
            torch.Tensor: 计算得到的关节扭矩张量。

        Examples:
            >>> simulator = YourSimulatorClass()  # 仿真器类的实例化
            >>> actions = torch.tensor([[0.5, -0.5, ...]])  # 示例动作输入
            >>> torques = simulator._compute_torques(actions)  # 计算关节扭矩

        Note:
            - 动作输入首先被缩放，特定动作（如髋关节屈曲）可能会进一步缩小。
            - 控制类型包括"actuator_net"和"P"，分别对应自定义的执行器网络和P控制器。
        """
        # PD控制器
        actions_scaled = actions[:, :12] * self.cfg.control.action_scale
        # 缩小髋关节屈曲范围
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction

        if self.cfg.domain_rand.randomize_lag_timesteps:
            # 如果配置了随机化延迟时间步，更新延迟缓冲区并计算目标关节位置
            self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
            self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
        else:
            # 否则直接计算目标关节位置
            self.joint_pos_target = actions_scaled + self.default_dof_pos

        control_type = self.cfg.control.control_type

        if control_type == "actuator_net":
            # 如果控制类型为执行器网络
            self.joint_pos_err = self.dof_pos - self.joint_pos_target + self.motor_offsets
            self.joint_vel = self.dof_vel
            # 通过执行器网络计算扭矩
            torques = self.actuator_network(self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last,
                                            self.joint_vel, self.joint_vel_last, self.joint_vel_last_last)
            # 更新误差和速度的历史记录
            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_pos_err_last = torch.clone(self.joint_pos_err)
            self.joint_vel_last_last = torch.clone(self.joint_vel_last)
            self.joint_vel_last = torch.clone(self.joint_vel)
        elif control_type == "P":
            # 如果控制类型为P控制器
            torques = self.p_gains * self.Kp_factors * (
                    self.joint_pos_target - self.dof_pos + self.motor_offsets) - self.d_gains * self.Kd_factors * self.dof_vel
        else:
            # 如果控制类型未知，抛出异常
            raise NameError(f"Unknown controller type: {control_type}")

        # 根据电机强度缩放扭矩，并限制在最大扭矩范围内
        torques = torques * self.motor_strengths
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids, cfg):
        """
        重置指定环境的关节位置和速度。

        对于给定的环境ID列表，将关节位置设置为默认位置的随机倍数（在0.5到1.5之间），速度设置为0。然后，使用这些新的状态更新仿真环境中的关节状态。

        Attributes:
            env_ids (torch.Tensor): 需要重置关节状态的环境ID列表。
            cfg (Config): 配置对象，虽然此函数中未直接使用，但保留以便未来可能的配置需求。

        Methods:
            torch_rand_float: 生成指定范围和形状的随机浮点数张量。
            gym.set_dof_state_tensor_indexed: 根据索引设置关节状态张量。

        Examples:
            >>> simulator = YourSimulatorClass()  # 仿真器类的实例化
            >>> env_ids = torch.tensor([0, 1, 2])  # 指定需要重置关节状态的环境ID
            >>> simulator._reset_dofs(env_ids, config)  # 重置指定环境的关节状态

        Note:
            - 关节位置被设置为默认位置的随机倍数，以增加初始化状态的多样性。
            - 关节速度被重置为0，表示静止状态。
        """
        # 将关节位置设置为默认位置的随机倍数，并将速度设置为0
        self.dof_pos[env_ids] = (self.default_dof_pos *
                                 torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof),
                                                                        device=self.device))
        self.dof_vel[env_ids] = 0.

        # 将env_ids转换为int32类型，以符合gym接口的要求
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # 使用新的关节状态更新仿真环境
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids, cfg):
        """
        重置选定环境的根状态位置和速度。

        根据课程设置基础位置，并在-0.5到0.5[m/s, rad/s]范围内随机选择基础速度。如果启用了自定义起点，将根据配置调整位置和偏航角。

        Attributes:
            env_ids (List[int]): 需要重置根状态的环境ID列表。
            cfg (Config): 配置对象。

        Methods:
            torch_rand_float: 生成指定范围和形状的随机浮点数张量。
            quat_from_angle_axis: 根据角度和轴生成四元数。
            gym.set_actor_root_state_tensor_indexed: 根据索引设置演员的根状态张量。

        Examples:
            >>> simulator = YourSimulatorClass()  # 仿真器类的实例化
            >>> env_ids = [0, 1, 2]  # 指定需要重置根状态的环境ID
            >>> simulator._reset_root_states(env_ids, config)  # 重置选定环境的根状态

        Note:
            - 如果启用了自定义起点，位置将根据环境原点和配置的初始化范围进行调整。
            - 偏航角和基础速度将随机选择。
        """
        # 如果启用了自定义起点
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state  # 设置基础状态
            self.root_states[env_ids, :3] += self.env_origins[env_ids]  # 加上环境原点
            # 随机调整x和y位置
            self.root_states[env_ids, 0:1] += torch_rand_float(-cfg.terrain.x_init_range,
                                                               cfg.terrain.x_init_range, (len(env_ids), 1),
                                                               device=self.device)
            self.root_states[env_ids, 1:2] += torch_rand_float(-cfg.terrain.y_init_range,
                                                               cfg.terrain.y_init_range, (len(env_ids), 1),
                                                               device=self.device)
            # 加上x和y的偏移
            self.root_states[env_ids, 0] += cfg.terrain.x_init_offset
            self.root_states[env_ids, 1] += cfg.terrain.y_init_offset
        else:
            self.root_states[env_ids] = self.base_init_state  # 设置基础状态
            self.root_states[env_ids, :3] += self.env_origins[env_ids]  # 加上环境原点

        # 随机选择偏航角
        init_yaws = torch_rand_float(-cfg.terrain.yaw_init_range,
                                     cfg.terrain.yaw_init_range, (len(env_ids), 1),
                                     device=self.device)
        quat = quat_from_angle_axis(init_yaws, torch.Tensor([0, 0, 1]).to(self.device))[:, 0, :]
        self.root_states[env_ids, 3:7] = quat  # 设置偏航角

        # 随机选择基础速度
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                           device=self.device)  # [7:10]: 线速度, [10:13]: 角速度
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # 更新仿真环境中的演员根状态
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # 如果配置了视频记录，并且环境ID包含0或训练环境数量，则处理视频帧
        if cfg.env.record_video and 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []  # 初始化完整视频帧列表
            else:
                self.complete_video_frames = self.video_frames[:]  # 保存当前视频帧
            self.video_frames = []  # 重置视频帧列表

        if cfg.env.record_video and self.eval_cfg is not None and self.num_train_envs in env_ids:
            if self.complete_video_frames_eval is None:
                self.complete_video_frames_eval = []  # 初始化评估环境的完整视频帧列表
            else:
                self.complete_video_frames_eval = self.video_frames_eval[:]  # 保存当前评估环境的视频帧
            self.video_frames_eval = []  # 重置评估环境的视频帧列表

    def _push_robots(self, env_ids, cfg):
        """
        对选定的环境中的机器人施加随机推力。

        通过设置随机的基础速度来模拟冲击力，以此来模拟对机器人的随机推力。这有助于增加训练过程中的鲁棒性。

        Attributes:
            env_ids (torch.Tensor): 需要施加推力的环境ID列表。
            cfg (Config): 配置对象，包含推力相关的配置参数。

        Methods:
            torch_rand_float: 生成指定范围和形状的随机浮点数张量。
            gym.set_actor_root_state_tensor: 设置演员的根状态张量。

        Examples:
            >>> simulator = YourSimulatorClass()  # 仿真器类的实例化
            >>> env_ids = torch.tensor([0, 1, 2])  # 指定需要施加推力的环境ID
            >>> simulator._push_robots(env_ids, config)  # 对选定环境中的机器人施加随机推力

        Note:
            - 只有在配置中启用了推力功能，并且当前步骤符合推力间隔时，才会施加推力。
        """
        # 检查是否启用了推力功能，并且当前步骤符合推力间隔
        if cfg.domain_rand.push_robots:
            env_ids = env_ids[self.episode_length_buf[env_ids] % int(cfg.domain_rand.push_interval) == 0]

            # 获取最大推力速度
            max_vel = cfg.domain_rand.max_push_vel_xy
            # 为选定的环境随机生成线速度x/y
            self.root_states[env_ids, 7:9] = torch_rand_float(-max_vel, max_vel, (len(env_ids), 2),
                                                              device=self.device)
            # 更新仿真环境中的演员根状态
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _teleport_robots(self, env_ids, cfg):
        """
        将靠近边缘的机器人传送到另一侧。

        如果机器人的位置靠近地形的边缘，此方法将机器人传送到地形的另一侧，以防止它们掉出地形边界。

        Attributes:
            env_ids (torch.Tensor): 需要检查和可能传送的环境ID列表。
            cfg (Config): 配置对象，包含地形和传送相关的配置参数。

        Methods:
            gym.set_actor_root_state_tensor: 设置演员的根状态张量。
            gym.refresh_actor_root_state_tensor: 刷新演员的根状态张量。

        Examples:
            >>> simulator = YourSimulatorClass()  # 仿真器类的实例化
            >>> env_ids = torch.tensor([0, 1, 2])  # 指定需要检查和可能传送的环境ID
            >>> simulator._teleport_robots(env_ids, config)  # 将靠近边缘的机器人传送到另一侧

        Note:
            - 只有在配置中启用了机器人传送功能时，此方法才会执行。
            - 传送阈值和地形尺寸由配置对象中的参数决定。
        """
        # 检查是否启用了机器人传送功能
        if cfg.terrain.teleport_robots:
            thresh = cfg.terrain.teleport_thresh  # 获取传送阈值

            x_offset = int(cfg.terrain.x_offset * cfg.terrain.horizontal_scale)  # 计算x方向的偏移

            # 寻找x坐标低于阈值的环境ID
            low_x_ids = env_ids[self.root_states[env_ids, 0] < thresh + x_offset]
            # 将这些环境中的机器人传送到x方向的另一侧
            self.root_states[low_x_ids, 0] += cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            # 寻找x坐标高于阈值的环境ID
            high_x_ids = env_ids[
                self.root_states[env_ids, 0] > cfg.terrain.terrain_length * cfg.terrain.num_rows - thresh + x_offset]
            # 将这些环境中的机器人传送到x方向的另一侧
            self.root_states[high_x_ids, 0] -= cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1)

            # 寻找y坐标低于阈值的环境ID
            low_y_ids = env_ids[self.root_states[env_ids, 1] < thresh]
            # 将这些环境中的机器人传送到y方向的另一侧
            self.root_states[low_y_ids, 1] += cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            # 寻找y坐标高于阈值的环境ID
            high_y_ids = env_ids[
                self.root_states[env_ids, 1] > cfg.terrain.terrain_width * cfg.terrain.num_cols - thresh]
            # 将这些环境中的机器人传送到y方向的另一侧
            self.root_states[high_y_ids, 1] -= cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1)

            # 更新仿真环境中的演员根状态
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            # 刷新演员的根状态张量，确保变更生效
            self.gym.refresh_actor_root_state_tensor(self.sim)

    def _get_noise_scale_vec(self, cfg):
        """
        根据配置生成噪声比例向量。

        此函数根据配置中指定的不同观测类型和噪声设置，生成一个噪声比例向量。该向量用于在仿真过程中向观测添加噪声，以提高模型的鲁棒性。

        Attributes:
            cfg (Config): 配置对象，包含噪声相关的配置参数。

        Returns:
            torch.Tensor: 噪声比例向量。

        Examples:
            >>> simulator = YourSimulatorClass()  # 仿真器类的实例化
            >>> noise_scale_vec = simulator._get_noise_scale_vec(config)  # 生成噪声比例向量

        Note:
            - 噪声比例向量的长度和内容取决于配置中指定的观测类型。
            - 该向量用于在仿真过程中向观测添加噪声，以提高模型的鲁棒性。
        """
        # 确定是否添加噪声
        self.add_noise = self.cfg.noise.add_noise
        # 获取噪声比例配置
        noise_scales = self.cfg.noise_scales
        # 获取噪声等级
        noise_level = self.cfg.noise.noise_level
        # 初始化噪声比例向量，包括重力、关节位置、关节速度的噪声
        noise_vec = torch.cat((torch.ones(3) * noise_scales.gravity * noise_level,
                               torch.ones(
                                   self.num_actuated_dof) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                               torch.ones(
                                   self.num_actuated_dof) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                               torch.zeros(self.num_actions),
                               ), dim=0)

        # 如果观测包括命令，更新噪声比例向量
        if self.cfg.env.observe_command:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.gravity * noise_level,
                                   torch.zeros(self.cfg.commands.num_commands),
                                   torch.ones(
                                       self.num_actuated_dof) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                                   torch.ones(
                                       self.num_actuated_dof) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                                   torch.zeros(self.num_actions),
                                   ), dim=0)
        # 如果观测包括两个之前的动作，扩展噪声比例向量
        if self.cfg.env.observe_two_prev_actions:
            noise_vec = torch.cat((noise_vec, torch.zeros(self.num_actions)), dim=0)
        # 如果观测包括时间参数，扩展噪声比例向量
        if self.cfg.env.observe_timing_parameter:
            noise_vec = torch.cat((noise_vec, torch.zeros(1)), dim=0)
        # 如果观测包括时钟输入，扩展噪声比例向量
        if self.cfg.env.observe_clock_inputs:
            noise_vec = torch.cat((noise_vec, torch.zeros(4)), dim=0)
        # 如果观测包括线速度和角速度，更新噪声比例向量
        if self.cfg.env.observe_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                                   torch.ones(3) * noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel,
                                   noise_vec
                                   ), dim=0)
        # 如果只观测线速度，更新噪声比例向量
        if self.cfg.env.observe_only_lin_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                                   noise_vec
                                   ), dim=0)
        # 如果观测包括偏航角，扩展噪声比例向量
        if self.cfg.env.observe_yaw:
            noise_vec = torch.cat((noise_vec, torch.zeros(1)), dim=0)
        # 如果观测包括接触状态，更新噪声比例向量
        if self.cfg.env.observe_contact_states:
            noise_vec = torch.cat((noise_vec, torch.ones(4) * noise_scales.contact_states * noise_level), dim=0)

        # 将噪声比例向量移动到指定的设备上
        noise_vec = noise_vec.to(self.device)

        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """
        初始化各种状态和控制缓冲区。
        此方法主要负责获取和初始化与仿真环境相关的状态张量，包括动作器、接触力、刚体状态等，并对这些张量进行封装处理，便于后续操作。
        同时，根据配置初始化控制相关的张量，如目标位置、速度、力矩等。

        Attributes:
            root_states (torch.Tensor): 封装后的根状态张量。
            dof_state (torch.Tensor): 封装后的关节状态张量。
            net_contact_forces (torch.Tensor): 封装后的接触力张量。
            dof_pos (torch.Tensor): 关节位置。
            base_pos (torch.Tensor): 基座位置。
            dof_vel (torch.Tensor): 关节速度。
            base_quat (torch.Tensor): 基座四元数。
            rigid_body_state (torch.Tensor): 封装后的刚体状态张量。
            foot_velocities (torch.Tensor): 足部速度。
            foot_positions (torch.Tensor): 足部位置。
            prev_base_pos (torch.Tensor): 上一时刻的基座位置。
            prev_foot_velocities (torch.Tensor): 上一时刻的足部速度。
            lag_buffer (list): 延迟缓冲区。
            contact_forces (torch.Tensor): 接触力。
            common_step_counter (int): 步骤计数器。
            extras (dict): 额外信息。
            height_points (torch.Tensor): 高度点。
            measured_heights (int): 测量高度。
            noise_scale_vec (torch.Tensor): 噪声比例向量。
            gravity_vec (torch.Tensor): 重力向量。
            forward_vec (torch.Tensor): 前向向量。
            torques (torch.Tensor): 力矩。
            p_gains (torch.Tensor): 比例增益。
            d_gains (torch.Tensor): 微分增益。
            actions (torch.Tensor): 动作。
            last_actions (torch.Tensor): 上一动作。
            last_last_actions (torch.Tensor): 上上一动作。
            joint_pos_target (torch.Tensor): 关节位置目标。
            last_joint_pos_target (torch.Tensor): 上一关节位置目标。
            last_last_joint_pos_target (torch.Tensor): 上上一关节位置目标。
            last_dof_vel (torch.Tensor): 上一关节速度。
            last_root_vel (torch.Tensor): 上一根部速度。
            commands_value (torch.Tensor): 命令值。
            commands (torch.Tensor): 命令。
            commands_scale (torch.Tensor): 命令比例。
            desired_contact_states (torch.Tensor): 期望接触状态。
            feet_air_time (torch.Tensor): 足部空中时间。
            last_contacts (torch.Tensor): 上一接触。
            last_contact_filt (torch.Tensor): 上一接触过滤。
            base_lin_vel (torch.Tensor): 基座线速度。
            base_ang_vel (torch.Tensor): 基座角速度。
            projected_gravity (torch.Tensor): 投影重力。
            default_dof_pos (torch.Tensor): 默认关节位置。
            actuator_network (function): 执行器网络函数。
            joint_pos_err_last_last (torch.Tensor): 上上一关节位置误差。
            joint_pos_err_last (torch.Tensor): 上一关节位置误差。
            joint_vel_last_last (torch.Tensor): 上上一关节速度。
            joint_vel_last (torch.Tensor): 上一关节速度。
        """
        # 获取仿真环境中的状态张量
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)  # 获取根状态张量
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)  # 获取关节状态张量
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)  # 获取接触力张量
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)  # 获取刚体状态张量
        # 刷新状态张量以确保数据是最新的
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)  # 渲染所有相机传感器

        # 创建一些封装张量，用于不同的数据片段
        self.root_states = gymtorch.wrap_tensor(actor_root_state)  # 封装根状态张量
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)  # 封装关节状态张量
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies,
                                  :]  # 封装接触力张量，并调整形状
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]  # 提取关节位置
        self.base_pos = self.root_states[:self.num_envs, 0:3]  # 提取基座位置
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]  # 提取关节速度
        self.base_quat = self.root_states[:self.num_envs, 3:7]  # 提取基座四元数
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs * self.num_bodies,
                                :]  # 封装刚体状态张量，并调整形状
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                               7:10]  # 提取足部速度
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]  # 提取足部位置
        self.prev_base_pos = self.base_pos.clone()  # 复制基座位置作为上一时刻的位置
        self.prev_foot_velocities = self.foot_velocities.clone()  # 复制足部速度作为上一时刻的速度

        # 初始化延迟缓冲区
        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps + 1)]

        # 封装接触力张量，并调整形状为(num_envs, num_bodies, xyz axis)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :].view(
            self.num_envs, -1, 3)

        # 初始化一些后续使用的数据
        self.common_step_counter = 0  # 步骤计数器
        self.extras = {}  # 额外信息字典

        # 如果配置中指定了测量高度，则初始化高度点
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points(torch.arange(self.num_envs, device=self.device), self.cfg)
        self.measured_heights = 0  # 测量高度初始化为0

        # 初始化噪声比例向量、重力向量、前向向量等控制相关的张量
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)  # 获取噪声比例向量
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))  # 重力向量
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))  # 前向向量
        # 初始化力矩、PD增益、动作等张量
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                   requires_grad=False)  # 力矩
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)  # 比例增益
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)  # 微分增益
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)  # 动作
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)  # 上一动作
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                             requires_grad=False)  # 上上一动作
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                            requires_grad=False)  # 关节位置目标
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                                 requires_grad=False)  # 上一关节位置目标
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                                      device=self.device, requires_grad=False)  # 上上一关节位置目标
        self.last_dof_vel = torch.zeros_like(self.dof_vel)  # 上一关节速度
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])  # 上一根部速度

        # 初始化命令相关的张量
        self.commands_value = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                          device=self.device, requires_grad=False)  # 命令值
        self.commands = torch.zeros_like(self.commands_value)  # 命令
        # 命令比例
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
                                            self.obs_scales.body_height_cmd, self.obs_scales.gait_freq_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.footswing_height_cmd, self.obs_scales.body_pitch_cmd,
                                            self.obs_scales.body_roll_cmd, self.obs_scales.stance_width_cmd,
                                            self.obs_scales.stance_length_cmd, self.obs_scales.aux_reward_cmd],
                                           device=self.device, requires_grad=False)[:self.cfg.commands.num_commands]
        # 期望接触状态
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False)

        # 初始化足部空中时间和接触状态相关的张量
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)  # 足部空中时间
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)  # 上一接触
        self.last_contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool,
                                             device=self.device, requires_grad=False)  # 上一接触过滤
        # 基座线速度和角速度
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 7:10])  # 基座线速度
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 10:13])  # 基座角速度
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)  # 投影重力

        # 初始化默认关节位置和PD增益
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device,
                                           requires_grad=False)  # 默认关节位置
        for i in range(self.num_dofs):
            name = self.dof_names[i]  # 关节名称
            angle = self.cfg.init_state.default_joint_angles[name]  # 默认角度
            self.default_dof_pos[i] = angle  # 设置默认关节位置
            found = False  # 标记是否找到对应的PD增益
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]  # 设置比例增益
                    self.d_gains[i] = self.cfg.control.damping[dof_name]  # 设置微分增益
                    found = True
            if not found:
                self.p_gains[i] = 0.  # 如果未找到，则设置为0
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")  # 打印警告信息
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)  # 增加一个维度

        # 如果控制类型为actuator_net，则加载执行器网络
        if self.cfg.control.control_type == "actuator_net":
            actuator_path = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../../resources/actuator_nets/unitree_go1.pt'  # 执行器网络路径
            actuator_network = torch.jit.load(actuator_path).to(self.device)  # 加载执行器网络

            def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_vel, joint_vel_last,
                                      joint_vel_last_last):
                """
                执行器网络评估函数。
                根据当前、上一时刻和上上一时刻的关节位置和速度，通过执行器网络计算力矩。

                Parameters:
                    joint_pos (torch.Tensor): 当前关节位置。
                    joint_pos_last (torch.Tensor): 上一时刻关节位置。
                    joint_pos_last_last (torch.Tensor): 上上一时刻关节位置。
                    joint_vel (torch.Tensor): 当前关节速度。
                    joint_vel_last (torch.Tensor): 上一时刻关节速度。
                    joint_vel_last_last (torch.Tensor): 上上一时刻关节速度。

                Returns:
                    torch.Tensor: 计算得到的力矩。
                """
                xs = torch.cat((joint_pos.unsqueeze(-1),
                                joint_pos_last.unsqueeze(-1),
                                joint_pos_last_last.unsqueeze(-1),
                                joint_vel.unsqueeze(-1),
                                joint_vel_last.unsqueeze(-1),
                                joint_vel_last_last.unsqueeze(-1)), dim=-1)  # 拼接输入张量
                torques = actuator_network(xs.view(self.num_envs * 12, 6))  # 通过执行器网络计算力矩
                return torques.view(self.num_envs, 12)  # 调整形状并返回

            self.actuator_network = eval_actuator_network  # 设置执行器网络评估函数

            # 初始化关节位置误差和速度相关的张量
            self.joint_pos_err_last_last = torch.zeros((self.num_envs, 12), device=self.device)  # 上上一关节位置误差
            self.joint_pos_err_last = torch.zeros((self.num_envs, 12), device=self.device)  # 上一关节位置误差
            self.joint_vel_last_last = torch.zeros((self.num_envs, 12), device=self.device)  # 上上一关节速度
            self.joint_vel_last = torch.zeros((self.num_envs, 12), device=self.device)  # 上一关节速度

    def _init_custom_buffers__(self):
        """
        初始化命令分布。
        该函数用于初始化与域随机化相关的属性，包括摩擦系数、恢复系数、负载、质心偏移、电机强度、电机偏移、比例增益因子、微分增益因子和重力向量。
        这些张量都被初始化为特定的形状和类型，并放置在指定的设备上，不需要梯度。

        Attributes:
            env_ids (list[int]): 环境ID列表，用于指定需要初始化的环境。

        Raises:
            None

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> agent._init_command_distribution(env_ids=[0, 1, 2])  # 初始化指定环境的命令分布

        Note:
            - 如果提供了自定义的初始动态参数字典，将会在这里设置它们。
            - 步态索引、时钟输入、双倍时间的时钟输入和半倍时间的时钟输入也会被初始化。
        """
        # domain randomization properties
        # 初始化摩擦系数张量，形状为(num_envs, 4)，类型为float，放置在指定的设备上，不需要梯度
        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, 4, dtype=torch.float,
                                                                  device=self.device,
                                                                  requires_grad=False)
        # 初始化恢复系数张量，形状为(num_envs, 4)，类型为float，放置在指定的设备上，不需要梯度
        self.restitutions = self.default_restitution * torch.ones(self.num_envs, 4, dtype=torch.float,
                                                                  device=self.device,
                                                                  requires_grad=False)
        # 初始化负载张量，形状为(num_envs,)，类型为float，放置在指定的设备上，不需要梯度
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # 初始化质心偏移张量，形状为(num_envs, 3)，类型为float，放置在指定的设备上，不需要梯度
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        # 初始化电机强度张量，形状为(num_envs, num_dof)，类型为float，放置在指定的设备上，不需要梯度
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        # 初始化电机偏移张量，形状为(num_envs, num_dof)，类型为float，放置在指定的设备上，不需要梯度
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        # 初始化比例增益因子张量，形状为(num_envs, num_dof)，类型为float，放置在指定的设备上，不需要梯度
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        # 初始化微分增益因子张量，形状为(num_envs, num_dof)，类型为float，放置在指定的设备上，不需要梯度
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        # 初始化重力向量张量，形状为(num_envs, 3)，类型为float，放置在指定的设备上，不需要梯度
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        # 根据上轴索引计算重力向量，并将其重复扩展到每个环境
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))

        # if custom initialization values were passed in, set them here
        dynamics_params = ["friction_coeffs", "restitutions", "payloads", "com_displacements", "motor_strengths",
                           "Kp_factors", "Kd_factors"]  # 定义动态参数的名称列表
        if self.initial_dynamics_dict is not None:  # 如果存在初始动态参数字典
            for k, v in self.initial_dynamics_dict.items():  # 遍历字典中的每个项目
                if k in dynamics_params:  # 如果键是动态参数之一
                    setattr(self, k, v.to(self.device))  # 设置属性，并将值转移到指定的设备

        # 初始化步态索引张量，形状为(num_envs,)，类型为float，放置在指定的设备上，不需要梯度
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        # 初始化时钟输入张量，形状为(num_envs, 4)，类型为float，放置在指定的设备上，不需要梯度
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        # 初始化双倍时间的时钟输入张量，形状为(num_envs, 4)，类型为float，放置在指定的设备上，不需要梯度
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
        # 初始化半倍时间的时钟输入张量，形状为(num_envs, 4)，类型为float，放置在指定的设备上，不需要梯度
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                 requires_grad=False)

    def _init_command_distribution(self, env_ids):
        """
        初始化命令分布和课程学习策略。
        根据配置文件中的设置，初始化不同的命令类别和相应的课程学习策略。支持根据奖励阈值和利普希茨连续性来调整命令分布。

        Parameters:
            env_ids (list): 环境ID列表，用于初始化环境特定的命令分布。

        Attributes:
            category_names (list): 命令类别名称列表。
            curricula (list): 课程学习策略实例列表。
            env_command_bins (numpy.ndarray): 环境命令分布的索引。
            env_command_categories (numpy.ndarray): 环境命令类别的索引。
        """
        # 默认命令类别设置为'nominal'
        self.category_names = ['nominal']
        # 如果配置启用了步态课程学习，更新命令类别列表
        if self.cfg.commands.gaitwise_curricula:
            self.category_names = ['pronk', 'trot', 'pace', 'bound']

        # 根据课程学习类型选择相应的课程学习类
        if self.cfg.commands.curriculum_type == "RewardThresholdCurriculum":
            from .curriculum import RewardThresholdCurriculum  # 导入奖励阈值课程学习类
            CurriculumClass = RewardThresholdCurriculum  # 设置课程学习类为奖励阈值课程学习

        self.curricula = []  # 初始化课程学习策略列表
        # 遍历命令类别，为每个类别创建课程学习实例
        for category in self.category_names:
            # 创建课程学习实例，并配置其参数
            self.curricula += [CurriculumClass(seed=self.cfg.commands.curriculum_seed,
                                               x_vel=(
                                               self.cfg.commands.limit_vel_x[0], self.cfg.commands.limit_vel_x[1],
                                               self.cfg.commands.num_bins_vel_x),
                                               y_vel=(
                                               self.cfg.commands.limit_vel_y[0], self.cfg.commands.limit_vel_y[1],
                                               self.cfg.commands.num_bins_vel_y),
                                               yaw_vel=(
                                               self.cfg.commands.limit_vel_yaw[0], self.cfg.commands.limit_vel_yaw[1],
                                               self.cfg.commands.num_bins_vel_yaw),
                                               body_height=(self.cfg.commands.limit_body_height[0],
                                                            self.cfg.commands.limit_body_height[1],
                                                            self.cfg.commands.num_bins_body_height),
                                               gait_frequency=(self.cfg.commands.limit_gait_frequency[0],
                                                               self.cfg.commands.limit_gait_frequency[1],
                                                               self.cfg.commands.num_bins_gait_frequency),
                                               gait_phase=(self.cfg.commands.limit_gait_phase[0],
                                                           self.cfg.commands.limit_gait_phase[1],
                                                           self.cfg.commands.num_bins_gait_phase),
                                               gait_offset=(self.cfg.commands.limit_gait_offset[0],
                                                            self.cfg.commands.limit_gait_offset[1],
                                                            self.cfg.commands.num_bins_gait_offset),
                                               gait_bounds=(self.cfg.commands.limit_gait_bound[0],
                                                            self.cfg.commands.limit_gait_bound[1],
                                                            self.cfg.commands.num_bins_gait_bound),
                                               gait_duration=(self.cfg.commands.limit_gait_duration[0],
                                                              self.cfg.commands.limit_gait_duration[1],
                                                              self.cfg.commands.num_bins_gait_duration),
                                               footswing_height=(self.cfg.commands.limit_footswing_height[0],
                                                                 self.cfg.commands.limit_footswing_height[1],
                                                                 self.cfg.commands.num_bins_footswing_height),
                                               body_pitch=(self.cfg.commands.limit_body_pitch[0],
                                                           self.cfg.commands.limit_body_pitch[1],
                                                           self.cfg.commands.num_bins_body_pitch),
                                               body_roll=(self.cfg.commands.limit_body_roll[0],
                                                          self.cfg.commands.limit_body_roll[1],
                                                          self.cfg.commands.num_bins_body_roll),
                                               stance_width=(self.cfg.commands.limit_stance_width[0],
                                                             self.cfg.commands.limit_stance_width[1],
                                                             self.cfg.commands.num_bins_stance_width),
                                               stance_length=(self.cfg.commands.limit_stance_length[0],
                                                              self.cfg.commands.limit_stance_length[1],
                                                              self.cfg.commands.num_bins_stance_length),
                                               aux_reward_coef=(self.cfg.commands.limit_aux_reward_coef[0],
                                                                self.cfg.commands.limit_aux_reward_coef[1],
                                                                self.cfg.commands.num_bins_aux_reward_coef),
                                               )]

        # 如果课程学习类型为LipschitzCurriculum，为每个课程设置利普希茨阈值和二进制阶段参数
        if self.cfg.commands.curriculum_type == "LipschitzCurriculum":
            for curriculum in self.curricula:
                curriculum.set_params(lipschitz_threshold=self.cfg.commands.lipschitz_threshold,
                                      binary_phases=self.cfg.commands.binary_phases)

        # 初始化环境命令分布和类别的索引数组
        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int)
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int)

        # 设置命令分布的上下限数组
        low = np.array(
            [self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_y[0], self.cfg.commands.ang_vel_yaw[0],
             self.cfg.commands.body_height_cmd[0],
             self.cfg.commands.gait_frequency_cmd_range[0], self.cfg.commands.gait_phase_cmd_range[0],
             self.cfg.commands.gait_offset_cmd_range[0],
             self.cfg.commands.gait_bound_cmd_range[0], self.cfg.commands.gait_duration_cmd_range[0],
             self.cfg.commands.footswing_height_range[0],
             self.cfg.commands.body_pitch_range[0], self.cfg.commands.body_roll_range[0],
             self.cfg.commands.stance_width_range[0],
             self.cfg.commands.stance_length_range[0], self.cfg.commands.aux_reward_coef_range[0]])
        high = np.array(
            [self.cfg.commands.lin_vel_x[1], self.cfg.commands.lin_vel_y[1], self.cfg.commands.ang_vel_yaw[1],
             self.cfg.commands.body_height_cmd[1],
             self.cfg.commands.gait_frequency_cmd_range[1], self.cfg.commands.gait_phase_cmd_range[1],
             self.cfg.commands.gait_offset_cmd_range[1],
             self.cfg.commands.gait_bound_cmd_range[1], self.cfg.commands.gait_duration_cmd_range[1],
             self.cfg.commands.footswing_height_range[1],
             self.cfg.commands.body_pitch_range[1], self.cfg.commands.body_roll_range[1],
             self.cfg.commands.stance_width_range[1],
             self.cfg.commands.stance_length_range[1], self.cfg.commands.aux_reward_coef_range[1]])
        # 为每个课程学习策略设置命令分布的上下限
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)

    def _prepare_reward_function(self):
        """
        准备奖励函数，用于计算和管理奖励。

        此方法初始化奖励容器，处理奖励比例，准备奖励函数列表，并初始化奖励和命令的累计和。

        Attributes:
            reward_container (object): 根据配置选定的奖励容器实例。
            reward_scales (dict): 经过处理的奖励比例字典，移除了比例为零的项并调整了非零项。
            reward_functions (list): 奖励函数的列表。
            reward_names (list): 有效奖励名称的列表。
            episode_sums (dict): 每个环境中每种奖励的累计和。
            episode_sums_eval (dict): 评估模式下，每个环境中每种奖励的累计和。
            command_sums (dict): 每个环境中命令的累计和。

        Methods:
            - 初始化奖励容器实例。
            - 处理奖励比例，移除比例为零的项并调整非零项。
            - 准备奖励函数列表，排除终止奖励并检查奖励函数的存在性。
            - 初始化奖励和命令的累计和。

        Returns:
            None

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> agent._prepare_reward_function()  # 准备奖励函数

        Note:
            - 奖励容器根据配置文件中的 'rewards.reward_container_name' 选择。
            - 奖励比例为零的奖励不会被包含在计算中。
        """
        # 导入奖励容器
        from go1_gym.envs.rewards.corl_rewards import CoRLRewards
        reward_containers = {"CoRLRewards": CoRLRewards}
        # 根据配置初始化奖励容器实例
        self.reward_container = reward_containers[self.cfg.rewards.reward_container_name](self)

        # 移除奖励比例为零的项，并将非零项乘以dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        # 准备奖励函数列表
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue

            if not hasattr(self.reward_container, '_reward_' + name):
                print(f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!")
            else:
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))

        # 初始化每个环境中每种奖励的累计和
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        # 评估模式下的奖励累计和初始化
        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                      requires_grad=False)
        # 初始化每个环境中命令的累计和
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}

    def _create_ground_plane(self):
        """
        在仿真环境中添加地面平面，并根据配置设置摩擦系数和恢复系数。

        此方法创建一个地面平面实例，并使用配置文件中指定的静态摩擦系数、动态摩擦系数和恢复系数进行配置。
        地面平面是仿真中所有物理交互的基础。

        Attributes:
            cfg (object): 包含地形配置的对象。

        Methods:
            gym.add_ground: 向仿真环境中添加地面平面。

        Returns:
            None

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> agent._create_ground_plane()  # 在仿真环境中添加地面平面

        Note:
            - 此方法在仿真环境初始化时调用。
            - 地面平面的物理属性由cfg对象提供。
        """
        # 初始化地面平面参数
        plane_params = gymapi.PlaneParams()
        # 设置地面平面的法线向量，指向z轴方向，表示水平地面
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        # 设置地面平面的静态摩擦系数
        plane_params.static_friction = self.cfg.terrain.static_friction
        # 设置地面平面的动态摩擦系数
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        # 设置地面平面的恢复系数
        plane_params.restitution = self.cfg.terrain.restitution
        # 向仿真环境中添加配置好的地面平面
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """
        在仿真环境中添加高度场地形，并根据配置设置相关参数。

        此方法创建一个高度场地形实例，使用配置文件中指定的水平和垂直缩放系数、行列数、边界大小以及摩擦系数和恢复系数。

        Attributes:
            terrain.cfg (object): 包含地形配置的对象。
            cfg.terrain (object): 包含地形摩擦和恢复系数配置的对象。

        Methods:
            gym.add_heightfield: 向仿真环境中添加高度场地形。

        Returns:
            None

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> agent._create_heightfield()  # 在仿真环境中添加高度场地形

        Note:
            - 此方法在仿真环境初始化时调用，用于添加地形。
            - 高度场地形的具体参数由cfg对象提供。
        """
        # 初始化高度场参数
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale  # 设置列的缩放系数
        hf_params.row_scale = self.terrain.cfg.horizontal_scale  # 设置行的缩放系数
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale  # 设置垂直缩放系数
        hf_params.nbRows = self.terrain.tot_cols  # 设置总列数
        hf_params.nbColumns = self.terrain.tot_rows  # 设置总行数
        # 设置高度场的位置
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        # 设置摩擦系数和恢复系数
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        # 打印高度样本的形状和高度场的行列数，用于调试
        print(self.terrain.heightsamples.shape, hf_params.nbRows, hf_params.nbColumns)

        # 向仿真环境中添加高度场地形
        self.gym.add_heightfield(self.sim, self.terrain.heightsamples.T, hf_params)
        # 将高度样本转换为张量并存储
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """
        在仿真环境中添加三角形网格地形，并根据配置设置参数。

        此方法创建一个三角形网格地形实例，并使用配置文件中指定的静态摩擦系数、动态摩擦系数和恢复系数进行配置。
        三角形网格地形提供了一种更复杂和详细的地形表示方法，适用于需要高度不规则地形的仿真。

        Attributes:
            cfg (object): 包含地形配置的对象。
            terrain (object): 包含地形顶点和三角形数据的对象。

        Methods:
            gym.add_triangle_mesh: 向仿真环境中添加三角形网格地形。

        Returns:
            None

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> agent._create_trimesh()  # 在仿真环境中添加三角形网格地形

        Note:
            - 此方法在仿真环境初始化时调用，用于创建更复杂的地形。
        """
        # 初始化三角形网格地形参数
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]  # 设置顶点数量
        tm_params.nb_triangles = self.terrain.triangles.shape[0]  # 设置三角形数量

        # 设置地形的位置和摩擦、恢复系数
        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution

        # 向仿真环境中添加三角形网格地形
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        # 将地形高度样本转换为张量并存储
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """
        创建仿真环境实例，并加载机器人模型。

        此方法执行以下步骤：
        1. 加载机器人URDF/MJCF资产。
        2. 对于每个环境实例：
           2.1 创建环境。
           2.2 调用自由度(DOF)和刚体形状属性回调。
           2.3 使用这些属性创建演员并将其添加到环境中。
        3. 存储机器人不同部件的索引。

        Attributes:
            cfg (object): 包含环境配置的对象。

        Methods:
            gym.load_asset: 加载机器人资产。
            gym.create_env: 创建环境实例。
            gym.create_actor: 创建演员实例。
            gym.set_actor_dof_properties: 设置演员的DOF属性。
            gym.set_actor_rigid_body_properties: 设置演员的刚体属性。
            gym.create_camera_sensor: 创建摄像头传感器。
            gym.set_camera_location: 设置摄像头位置。

        Returns:
            None

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> agent._create_envs()  # 创建仿真环境实例并加载机器人模型

        Note:
            - 此方法在仿真环境初始化时调用。
            - 机器人模型和环境配置由cfg对象提供。
        """
        # 解析资产路径
        # 使用配置文件中指定的路径模板和全局变量MINI_GYM_ROOT_DIR来构建完整的资产路径
        asset_path = self.cfg.asset.file.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)
        # 获取资产文件的根目录路径
        asset_root = os.path.dirname(asset_path)
        # 获取资产文件的名称
        asset_file = os.path.basename(asset_path)

        # 设置资产选项
        # 创建一个新的AssetOptions对象，用于配置资产的加载选项
        asset_options = gymapi.AssetOptions()
        # 设置默认的自由度(DOF)驱动模式
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        # 是否合并固定关节
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        # 是否将圆柱体替换为胶囊体
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        # 是否翻转视觉附件
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        # 是否固定基座链接
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        # 设置密度
        asset_options.density = self.cfg.asset.density
        # 设置角阻尼
        asset_options.angular_damping = self.cfg.asset.angular_damping
        # 设置线性阻尼
        asset_options.linear_damping = self.cfg.asset.linear_damping
        # 设置最大角速度
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        # 设置最大线速度
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        # 设置骨架
        asset_options.armature = self.cfg.asset.armature
        # 设置厚度
        asset_options.thickness = self.cfg.asset.thickness
        # 是否禁用重力
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        # 加载资产
        # 使用之前设置的资产路径和选项在仿真中加载机器人模型
        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # 获取DOF和刚体数量
        # 查询加载的资产以获取其自由度（DOF）的数量
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        # 设置可激活的DOF数量，通常与仿真中的动作数量相等
        self.num_actuated_dof = self.num_actions
        # 查询资产以获取其刚体的数量
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)

        # 获取DOF和刚体属性
        # 获取资产的DOF属性，这些属性包括关节的限制、驱动器的配置等
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        # 获取资产的刚体形状属性，这些属性包括形状的大小、质量、摩擦系数等
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)

        # 保存资产中的刚体名称
        # 获取并保存资产中所有刚体的名称
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        # 获取并保存资产中所有DOF的名称
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        # 更新刚体和DOF的数量，确保与实际加载的资产匹配
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)

        # 筛选出脚部和需要惩罚接触的刚体名称
        # 根据配置中指定的脚部名称，从所有刚体名称中筛选出脚部刚体
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        # 根据配置中指定的需要惩罚接触的刚体名称，从所有刚体名称中筛选出相应刚体
        penalized_contact_names = [s for name in self.cfg.asset.penalize_contacts_on for s in body_names if name in s]
        # 根据配置中指定的在接触后终止仿真的刚体名称，从所有刚体名称中筛选出相应刚体
        termination_contact_names = [s for name in self.cfg.asset.terminate_after_contacts_on for s in body_names if
                                     name in s]

        # 初始化基础状态
        # 将配置文件中定义的初始位置、旋转、线速度和角速度合并为一个列表
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        # 将基础状态列表转换为张量，并指定设备和是否需要梯度
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        # 创建一个Transform对象，并设置其位置部分为基础状态中的位置信息
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # 初始化环境原点和地形相关的张量
        # 为每个环境创建一个原点位置的零张量
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # 创建一个表示地形级别的零张量
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        # 创建一个表示地形原点的零张量
        self.terrain_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # 创建一个表示地形类型的零张量
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        # 调用一个方法来获取或计算每个环境的原点位置，可能会根据是训练还是评估阶段进行不同的处理
        self._call_train_eval(self._get_env_origins, torch.arange(self.num_envs, device=self.device))
        # 设置环境的边界大小，这里初始化为零向量，可能会在后续步骤中根据需要进行调整
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        # 初始化用于存储演员句柄和IMU传感器句柄的列表
        self.actor_handles = []
        self.imu_sensor_handles = []
        # 初始化用于存储环境句柄的列表
        self.envs = []

        # 初始化默认摩擦和恢复系数
        # 从资产的刚体形状属性中获取默认的摩擦系数和恢复系数，用于后续可能的自定义或重置
        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution

        # 初始化自定义缓冲区
        # 调用一个自定义方法来初始化或重置用于存储仿真过程中生成的特定数据的缓冲区
        self._init_custom_buffers__()

        # 随机化刚体属性
        # 调用一个方法，该方法根据当前是训练阶段还是评估阶段，对刚体的物理属性进行随机化处理
        # 这有助于增加训练过程中的多样性，提高模型的泛化能力
        self._call_train_eval(self._randomize_rigid_body_props, torch.arange(self.num_envs, device=self.device))

        # 随机化重力
        # 调用一个方法来随机化仿真环境中的重力设置，这同样是为了增加训练过程中的多样性和挑战性
        self._randomize_gravity()

        for i in range(self.num_envs):  # 遍历需要创建的环境数量
            # 创建环境实例
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()  # 克隆环境原点坐标
            # 随机调整环境的x坐标
            pos[0:1] += torch_rand_float(-self.cfg.terrain.x_init_range, self.cfg.terrain.x_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            # 随机调整环境的y坐标
            pos[1:2] += torch_rand_float(-self.cfg.terrain.y_init_range, self.cfg.terrain.y_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)  # 设置机器人的初始位置

            # 处理并设置刚体形状属性
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            # 创建演员并设置DOF属性
            anymal_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, "anymal", i,
                                                  self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)  # 处理DOF属性
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)  # 设置DOF属性
            # 获取并处理刚体属性，然后重新设置
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)  # 处理刚体属性
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)  # 将环境句柄添加到列表
            self.actor_handles.append(anymal_handle)  # 将演员句柄添加到列表

        # 初始化脚部、惩罚接触和终止接触的索引
        # 创建零张量以存储脚部、惩罚接触和终止接触刚体的索引
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(feet_names):
            # 为每个脚部名称找到对应的刚体句柄，并存储索引
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i, name in enumerate(penalized_contact_names):
            # 为每个需要惩罚接触的刚体名称找到对应的句柄，并存储索引
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0], name)

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i, name in enumerate(termination_contact_names):
            # 为每个终止接触的刚体名称找到对应的句柄，并存储索引
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0], name)

        # 如果配置了视频录制，设置摄像头
        if self.cfg.env.record_video:
            # 配置摄像头属性
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 256
            self.camera_props.height = 256
            # 为第一个环境创建摄像头传感器，并设置位置
            self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], self.camera_props)
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(1.5, 1, 3.0),
                                         gymapi.Vec3(0, 0, 0))

            # 如果存在评估配置，为评估环境创建摄像头传感器，并设置位置
            if self.eval_cfg is not None:
                self.rendering_camera_eval = self.gym.create_camera_sensor(self.envs[self.num_train_envs],
                                                                           self.camera_props)
                self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
                                             gymapi.Vec3(1.5, 1, 3.0), gymapi.Vec3(0, 0, 0))

        # 初始化视频录制相关的变量
        self.video_writer = None
        self.video_frames = []
        self.video_frames_eval = []
        self.complete_video_frames = []
        self.complete_video_frames_eval = []

    def render(self, mode="rgb_array"):
        """
        渲染环境的视觉表示。

        该方法用于生成环境的RGB图像，通过设置相机的位置和朝向来捕捉特定的视角。

        Attributes:
            mode (str): 渲染模式，本函数仅支持 "rgb_array" 模式。

        Returns:
            numpy.ndarray: 返回一个形状为 (宽度, 高度//4, 4) 的数组，代表渲染的RGB图像。

        Examples:
            >>> environment = YourEnvironmentClass()  # 环境类的实例化
            >>> image = environment.render(mode="rgb_array")  # 获取RGB图像
            >>> print(image.shape)

        Note:
            - 本方法假设相机已经在环境中被正确初始化。
            - 返回的图像数组包含RGBA通道，但宽度被缩减为原来的四分之一。
        """
        assert mode == "rgb_array"  # 确保渲染模式为 "rgb_array"
        # 获取根状态的x, y, z坐标
        bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
        # 设置相机的位置和朝向，以捕捉环境的视角
        self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                     gymapi.Vec3(bx, by, bz))
        self.gym.step_graphics(self.sim)  # 更新图形
        self.gym.render_all_camera_sensors(self.sim)  # 渲染所有相机传感器
        # 获取相机捕捉的图像
        img = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera, gymapi.IMAGE_COLOR)
        w, h = img.shape  # 获取图像的宽度和高度
        # 重塑图像数组的形状，返回处理后的图像
        return img.reshape([w, h // 4, 4])

    def _render_headless(self):
        """
        在无头模式下渲染视频帧并记录。

        该方法在无头模式下（没有图形界面的情况下）渲染并记录训练和评估过程的视频帧。
        当满足特定条件时，从仿真环境中捕获当前视角的图像，并将其添加到视频帧序列中。

        Attributes:
            record_now (bool): 是否记录训练过程的标志。
            complete_video_frames (list): 存储训练过程视频帧的列表。
            record_eval_now (bool): 是否记录评估过程的标志。
            complete_video_frames_eval (list): 存储评估过程视频帧的列表。
            root_states (numpy.ndarray): 存储根状态信息的数组。
            gym (gymapi.Gym): 仿真接口实例。
            sim (gymapi.Sim): 仿真实例。
            envs (list): 环境实例列表。
            rendering_camera (int): 渲染用的相机ID。
            rendering_camera_eval (int): 评估用的相机ID。
            camera_props (object): 相机属性对象。
            num_train_envs (int): 训练环境的数量。
            video_frames (list): 存储训练视频帧的列表。
            video_frames_eval (list): 存储评估视频帧的列表。

        Methods:
            _get_env_origins: 设置环境原点。
            _randomize_rigid_body_props: 随机化刚体属性。
            _randomize_gravity: 随机化重力向量。
            _process_rigid_shape_props: 处理刚体形状属性。
            _process_dof_props: 处理自由度属性。
            _process_rigid_body_props: 处理刚体属性。
            _init_custom_buffers: 初始化自定义缓冲区。
            _call_train_eval: 调用训练或评估过程的函数。
            _create_envs: 创建环境。

        Examples:
            >>> environment = YourEnvironmentClass()
            >>> environment._render_headless()

        Note:
            - 该方法仅在需要记录视频帧时调用。
            - 支持分别记录训练和评估过程的视频帧。
        """
        # 如果当前是记录训练过程且视频帧列表为空
        if self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0:
            # 获取根状态的位置信息
            bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
            # 设置相机位置
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                         gymapi.Vec3(bx, by, bz))
            # 获取当前相机视角的图像
            self.video_frame = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera,
                                                         gymapi.IMAGE_COLOR)
            # 调整图像形状以匹配视频帧的格式
            self.video_frame = self.video_frame.reshape((self.camera_props.height, self.camera_props.width, 4))
            # 将视频帧添加到列表中
            self.video_frames.append(self.video_frame)

        # 如果当前是记录评估过程且视频帧列表为空
        if self.record_eval_now and self.complete_video_frames_eval is not None and len(
                self.complete_video_frames_eval) == 0:
            # 如果评估配置存在
            if self.eval_cfg is not None:
                # 获取评估环境的根状态位置信息
                bx, by, bz = self.root_states[self.num_train_envs, 0], self.root_states[self.num_train_envs, 1], \
                    self.root_states[self.num_train_envs, 2]
                # 设置评估用相机的位置
                self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
                                             gymapi.Vec3(bx, by - 1.0, bz + 1.0),
                                             gymapi.Vec3(bx, by, bz))
                # 获取当前评估相机视角的图像
                self.video_frame_eval = self.gym.get_camera_image(self.sim, self.envs[self.num_train_envs],
                                                                  self.rendering_camera_eval,
                                                                  gymapi.IMAGE_COLOR)
                # 调整图像形状以匹配视频帧的格式
                self.video_frame_eval = self.video_frame_eval.reshape(
                    (self.camera_props.height, self.camera_props.width, 4))
                # 将视频帧添加到评估视频帧列表中
                self.video_frames_eval.append(self.video_frame_eval)

    def start_recording(self):
        """
        开始录制视频帧。

        此方法用于初始化或重置视频帧的存储，并启动录制过程。调用此方法后，所有捕获的视频帧将被存储，直到调用停止录制的方法。

        Attributes:
            complete_video_frames (NoneType or list): 存储完整视频帧的列表。如果未开始录制，则为None。
            record_now (bool): 指示是否当前正在录制的标志。

        Examples:
            >>> recorder = YourClassName()  # 实例化录制器
            >>> recorder.start_recording()  # 开始录制
            >>> # 进行一些操作，如捕获视频帧
            >>> recorder.stop_recording()  # 停止录制并保存视频

        Note:
            - 在开始录制之前，请确保已经创建了实例。
            - 调用此方法将重置之前的录制内容。
        """
        self.complete_video_frames = None  # 初始化视频帧存储为None，表示当前没有视频帧被存储
        self.record_now = True  # 设置录制标志为True，开始录制视频帧

    def start_recording_eval(self):
        """
        开始记录评估阶段的视频帧。

        当调用此方法时，将初始化用于存储评估视频帧的变量，并设置标志以开始记录视频帧。

        Attributes:
            complete_video_frames_eval (list or None): 存储评估阶段视频帧的列表，初始化为None。
            record_eval_now (bool): 标志位，用于指示是否开始记录评估阶段的视频帧。

        Methods:
            无

        Returns:
            None

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> agent.start_recording_eval()  # 开始记录评估阶段的视频帧

        Note:
            - 此方法仅在需要记录评估阶段的视频帧时调用。
            - 记录开始后，需要通过其他方法将视频帧添加到`complete_video_frames_eval`列表中。
        """
        # 初始化用于存储评估视频帧的列表为None，表示开始新的视频录制
        self.complete_video_frames_eval = None
        # 设置标志位为True，开始记录评估阶段的视频帧
        self.record_eval_now = True

    def pause_recording(self):
        """
        暂停录制过程，并清空当前录制的视频帧缓存。

        该方法用于在录制过程中暂停录制，同时清空已经录制的视频帧，以便重新开始录制。
        通常在录制过程中需要暂停或者在录制完成后准备下一次录制时使用。

        Attributes:
            complete_video_frames (list): 存储完整视频帧的列表。
            video_frames (list): 存储当前录制视频帧的列表。
            record_now (bool): 标记是否正在录制的布尔值。

        Methods:
            None

        Returns:
            None

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> agent.pause_recording()  # 暂停录制并清空视频帧缓存

        Note:
            - 调用此方法后，之前录制的视频帧将被清空，需要重新开始录制。
        """
        # 清空存储完整视频帧的列表，准备下一次录制
        self.complete_video_frames = []
        # 清空当前录制视频帧的列表，准备下一次录制
        self.video_frames = []
        # 设置录制标记为False，暂停录制过程
        self.record_now = False

    def pause_recording_eval(self):
        """
        暂停评估阶段的视频录制，并清空已录制的视频帧。

        此方法用于在评估过程中暂停视频录制，同时清空已经录制的视频帧列表，以便于重新开始录制。

        Attributes:
            complete_video_frames_eval (list): 存储评估阶段完整视频帧的列表，调用此方法后被清空。
            video_frames_eval (list): 存储评估阶段当前视频帧的列表，调用此方法后被清空。
            record_eval_now (bool): 标志位，用于指示是否正在记录评估阶段的视频帧，调用此方法后设置为False。

        Methods:
            无

        Returns:
            None

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> agent.pause_recording_eval()  # 暂停评估阶段的视频录制并清空已录制的视频帧

        Note:
            - 此方法仅在需要暂停评估阶段的视频录制时使用。
            - 调用此方法后，如果需要重新开始录制，应再次调用开始录制的方法。
        """
        # 清空存储评估阶段完整视频帧的列表
        self.complete_video_frames_eval = []
        # 清空存储评估阶段当前视频帧的列表
        self.video_frames_eval = []
        # 设置标志位为False，暂停评估阶段的视频录制
        self.record_eval_now = False

    def get_complete_frames(self):
        """
        获取已完成录制的视频帧列表。

        如果视频帧列表尚未初始化（即为None），则返回一个空列表。
        否则，返回存储已完成录制视频帧的列表。

        Attributes:
            complete_video_frames (list or None): 存储已完成录制视频帧的列表，可能未初始化。

        Methods:
            None

        Returns:
            list: 已完成录制的视频帧列表，如果未初始化则返回空列表。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> frames = agent.get_complete_frames()  # 获取已完成录制的视频帧列表

        Note:
            - 在调用此方法前，确保已经开始并可能完成了视频帧的录制。
        """
        # 检查是否已经初始化了视频帧列表
        if self.complete_video_frames is None:
            # 如果未初始化，则返回一个空列表
            return []
        # 如果已初始化，返回存储的视频帧列表
        return self.complete_video_frames

    def get_complete_frames_eval(self):
        """
        获取评估阶段已完成录制的视频帧列表。

        如果评估阶段的视频帧列表尚未初始化（即为None），则返回一个空列表。
        否则，返回存储评估阶段已完成录制视频帧的列表。

        Attributes:
            complete_video_frames_eval (list or None): 存储评估阶段已完成录制视频帧的列表，可能未初始化。

        Methods:
            None

        Returns:
            list: 评估阶段已完成录制的视频帧列表，如果未初始化则返回空列表。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> frames_eval = agent.get_complete_frames_eval()  # 获取评估阶段已完成录制的视频帧列表

        Note:
            - 在调用此方法前，确保已经开始并可能完成了评估阶段的视频帧录制。
        """
        # 检查是否已经初始化了评估阶段的视频帧列表
        if self.complete_video_frames_eval is None:
            # 如果未初始化，则返回一个空列表
            return []
        # 如果已初始化，返回存储的评估阶段视频帧列表
        return self.complete_video_frames_eval

    def _get_env_origins(self, env_ids, cfg):
        """
        根据环境配置初始化环境原点位置。

        该方法根据地形类型和配置，为每个环境设置初始位置。支持基于地形的自定义初始位置和基于网格的标准初始位置。

        Attributes:
            env_ids (list): 需要设置原点的环境ID列表。
            cfg (object): 包含环境和地形配置的对象。

        Methods:
            torch.randint: 生成指定范围内的随机整数。
            torch.div: 对张量进行整数除法。
            torch.arange: 生成一系列整数。
            torch.from_numpy: 将NumPy数组转换为torch张量。
            np.floor: 向下取整。
            np.ceil: 向上取整。
            torch.meshgrid: 生成网格点坐标矩阵。

        Returns:
            None

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> agent._get_env_origins(env_ids=[0, 1, 2], cfg=config)  # 初始化指定环境的原点位置

        Note:
            - 支持的地形类型有 'heightfield' 和 'trimesh'。
            - 对于 'heightfield' 和 'trimesh' 类型，位置基于地形配置动态生成。
            - 对于其他类型，位置基于网格布局静态生成。
        """
        # 检查地形类型是否为 'heightfield' 或 'trimesh'
        if cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True  # 标记为使用自定义原点
            # 根据地形配置设置机器人的初始位置
            max_init_level = cfg.terrain.max_init_terrain_level
            min_init_level = cfg.terrain.min_init_terrain_level
            # 如果未启用课程学习，调整初始位置的范围
            if not cfg.terrain.curriculum: max_init_level = cfg.terrain.num_rows - 1
            if not cfg.terrain.curriculum: min_init_level = 0
            # 如果配置中心化机器人，计算中心化的位置范围
            if cfg.terrain.center_robots:
                min_terrain_level = cfg.terrain.num_rows // 2 - cfg.terrain.center_span
                max_terrain_level = cfg.terrain.num_rows // 2 + cfg.terrain.center_span - 1
                min_terrain_type = cfg.terrain.num_cols // 2 - cfg.terrain.center_span
                max_terrain_type = cfg.terrain.num_cols // 2 + cfg.terrain.center_span - 1
                # 为每个环境随机生成地形级别和类型
                self.terrain_levels[env_ids] = torch.randint(min_terrain_level, max_terrain_level + 1, (len(env_ids),),
                                                             device=self.device)
                self.terrain_types[env_ids] = torch.randint(min_terrain_type, max_terrain_type + 1, (len(env_ids),),
                                                            device=self.device)
            else:
                # 为每个环境随机生成地形级别
                self.terrain_levels[env_ids] = torch.randint(min_init_level, max_init_level + 1, (len(env_ids),),
                                                             device=self.device)
                # 为每个环境分配地形类型
                self.terrain_types[env_ids] = torch.div(torch.arange(len(env_ids), device=self.device),
                                                        (len(env_ids) / cfg.terrain.num_cols),
                                                        rounding_mode='floor').to(
                    torch.long)
            # 更新地形的最大级别和原点位置
            cfg.terrain.max_terrain_level = cfg.terrain.num_rows
            cfg.terrain.terrain_origins = torch.from_numpy(cfg.terrain.env_origins).to(self.device).to(torch.float)
            # 设置环境原点位置
            self.env_origins[env_ids] = cfg.terrain.terrain_origins[
                self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        else:
            self.custom_origins = False  # 标记为不使用自定义原点
            # 基于网格创建机器人的初始位置
            num_cols = np.floor(np.sqrt(len(env_ids)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = cfg.env.env_spacing
            # 设置环境原点的x, y坐标
            self.env_origins[env_ids, 0] = spacing * xx.flatten()[:len(env_ids)]
            self.env_origins[env_ids, 1] = spacing * yy.flatten()[:len(env_ids)]
            # 设置环境原点的z坐标为0
            self.env_origins[env_ids, 2] = 0.

    def _parse_cfg(self, cfg):
        """
        解析并初始化配置参数。

        此方法用于根据传入的配置对象初始化仿真环境的各项参数，包括时间步长、观测和奖励的缩放因子、课程学习阈值等。
        同时，根据地形类型调整课程学习的可用性，并计算最大仿真步数和各种随机事件的间隔。

        Attributes:
            cfg (object): 包含仿真环境配置的对象。

        Methods:
            np.ceil: 向上取整。

        Returns:
            None

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> agent._parse_cfg(config)  # 解析并初始化配置参数

        Note:
            - 该方法在仿真环境初始化时调用，用于配置环境参数。
            - 根据地形类型，可能会禁用课程学习功能。
        """
        # 计算控制时间步长
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        # 初始化观测的缩放因子
        self.obs_scales = self.cfg.obs_scales
        # 初始化奖励的缩放因子
        self.reward_scales = vars(self.cfg.reward_scales)
        # 初始化课程学习的阈值
        self.curriculum_thresholds = vars(self.cfg.curriculum_thresholds)
        # 解析命令范围
        cfg.command_ranges = vars(cfg.commands)
        # 如果地形类型不是 'heightfield' 或 'trimesh'，则禁用课程学习
        if cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            cfg.terrain.curriculum = False
        # 计算最大仿真步数
        max_episode_length_s = cfg.env.episode_length_s
        cfg.env.max_episode_length = np.ceil(max_episode_length_s / self.dt)
        self.max_episode_length = cfg.env.max_episode_length
        # 计算推力间隔、随机事件间隔和重力随机事件间隔
        cfg.domain_rand.push_interval = np.ceil(cfg.domain_rand.push_interval_s / self.dt)
        cfg.domain_rand.rand_interval = np.ceil(cfg.domain_rand.rand_interval_s / self.dt)
        cfg.domain_rand.gravity_rand_interval = np.ceil(cfg.domain_rand.gravity_rand_interval_s / self.dt)
        # 计算重力随机事件的持续时间
        cfg.domain_rand.gravity_rand_duration = np.ceil(
            cfg.domain_rand.gravity_rand_interval * cfg.domain_rand.gravity_impulse_duration)

    def _draw_debug_vis(self):
        """
        在仿真环境中绘制用于调试的视觉效果，特别是标出高度测量点。

        当启用测量高度的配置时，此方法通过在每个高度测量点的位置绘制小球体来可视化这些点。
        这有助于调试和验证高度测量功能的正确性。

        Attributes:
            terrain.cfg.measure_heights (bool): 是否启用高度测量的配置项。
            gym (gymapi.Gym): 使用的Gym接口实例。
            viewer (gymapi.Viewer): Gym视图渲染器。
            sim (gymapi.Sim): Gym仿真器实例。
            num_envs (int): 环境数量。
            root_states (torch.Tensor): 各环境中根部件的状态。
            measured_heights (torch.Tensor): 测量到的高度值。
            base_quat (torch.Tensor): 各环境中根部件的四元数。
            height_points (torch.Tensor): 高度测量点的位置。

        Methods:
            gym.clear_lines: 清除之前的绘制内容。
            gym.refresh_rigid_body_state_tensor: 刷新刚体状态张量。
            gymutil.WireframeSphereGeometry: 创建线框球体几何体。
            quat_apply_yaw: 应用偏航角旋转到四元数。
            gymutil.draw_lines: 在指定位置绘制球体。

        Returns:
            None

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> agent._draw_debug_vis()  # 绘制高度测量点的调试视觉效果

        Note:
            - 仅在配置中启用测量高度时有效。
            - 通过在每个高度测量点位置绘制小球体来可视化这些点。
        """
        # 检查是否启用了高度测量配置，如果未启用，则直接返回
        if not self.terrain.cfg.measure_heights:
            return
        # 清除之前在视图中绘制的所有线条
        self.gym.clear_lines(self.viewer)
        # 刷新仿真器中的刚体状态张量，以确保获取的状态是最新的
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # 创建一个黄色的线框球体几何体，用于在视图中表示高度测量点
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        # 遍历所有环境
        for i in range(self.num_envs):
            # 从根状态中获取当前环境的根部件位置
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            # 获取当前环境测量到的高度值
            heights = self.measured_heights[i].cpu().numpy()
            # 计算高度测量点的位置
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                           self.height_points[i]).cpu().numpy()
            # 遍历所有高度测量点
            for j in range(heights.shape[0]):
                # 计算球体的x, y, z坐标
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                # 创建球体的位姿，用于确定球体在仿真环境中的位置
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                # 在视图中绘制球体，以可视化高度测量点
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self, env_ids, cfg):
        """
        初始化用于测量地形高度的点。

        根据配置文件中指定的x和y坐标，创建一个网格点集合，用于后续测量地形高度。

        Attributes:
            env_ids (list): 需要初始化高度点的环境ID列表。
            cfg (object): 包含地形和环境配置信息的配置对象。

        Returns:
            torch.Tensor: 初始化的高度点，形状为 (len(env_ids), num_height_points, 3)，其中每个点包含(x, y, z)坐标。

        Examples:
            >>> env_ids = [0, 1, 2]
            >>> cfg = YourConfigClass()  # 配置类的实例化
            >>> points = agent._init_height_points(env_ids, cfg)
            >>> print(points.shape)
            >>> # 输出: torch.Size([3, num_height_points, 3])

        Note:
            - 这里假设z坐标（高度）初始为0，后续将根据地形数据更新。
            - x和y坐标是根据配置文件中的measured_points_x和measured_points_y生成的网格点。
        """
        # 根据配置文件中的y坐标创建一个张量
        y = torch.tensor(cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        # 根据配置文件中的x坐标创建一个张量
        x = torch.tensor(cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        # 使用x和y坐标生成一个网格
        grid_x, grid_y = torch.meshgrid(x, y)

        # 计算网格中点的总数，并更新配置对象中的高度点数量
        cfg.env.num_height_points = grid_x.numel()
        # 初始化一个用于存储高度点的张量，形状为(len(env_ids), num_height_points, 3)
        points = torch.zeros(len(env_ids), cfg.env.num_height_points, 3, device=self.device, requires_grad=False)
        # 设置所有高度点的x坐标
        points[:, :, 0] = grid_x.flatten()
        # 设置所有高度点的y坐标
        points[:, :, 1] = grid_y.flatten()
        # 返回初始化的高度点张量
        return points

    def _get_heights(self, env_ids, cfg):
        """
        获取指定环境中的高度值。

        根据环境ID和配置，计算并返回每个环境中指定点的高度值。支持平面和自定义地形类型。

        Attributes:
            env_ids (list): 环境ID列表。
            cfg (object): 配置对象，包含地形和环境的配置信息。

        Returns:
            torch.Tensor: 指定环境和点的高度值，形状为 (len(env_ids), cfg.env.num_height_points)。

        Raises:
            NameError: 如果配置中的地形类型为 'none'，即没有指定地形类型时抛出。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> heights = agent._get_heights(env_ids=[0, 1, 2], cfg=config)  # 获取指定环境中的高度值
            >>> print(heights)

        Note:
            - 支持的地形类型有 'plane' 和自定义类型。
            - 当地形类型为 'plane' 时，返回全零张量。
            - 对于自定义地形，通过仿真环境的地形配置计算高度值。
        """
        # 根据地形类型为 'plane'，返回全零张量
        if cfg.terrain.mesh_type == 'plane':
            return torch.zeros(len(env_ids), cfg.env.num_height_points, device=self.device, requires_grad=False)
        # 如果地形类型为 'none'，抛出异常
        elif cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        # 计算每个环境中指定点的位置
        points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, cfg.env.num_height_points),
                                self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)

        # 调整点的位置以适应地形边界和比例
        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()

        # 将点的坐标转换为一维索引
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)

        # 限制点的坐标范围以防越界
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        # 获取指定点的高度值
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]

        # 计算最终的高度值
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        # 返回调整后的高度值
        return heights.view(len(env_ids), -1) * self.terrain.cfg.vertical_scale
