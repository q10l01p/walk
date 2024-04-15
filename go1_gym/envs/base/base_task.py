# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import sys

import gym
import torch
from isaacgym import gymapi, gymutil

from gym import spaces
import numpy as np


# Base class for RL tasks
class BaseTask(gym.Env):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, eval_cfg=None):
        """
        初始化BaseTask类。

        该类是一个环境的基类，用于创建模拟环境、场景和查看器，并分配PyTorch缓冲区。它接收配置参数，并根据这些参数初始化仿真环境。

        Attributes:
            gym (gymapi.Gym): gymapi的句柄。
            sim_params (gymapi.SimParams): 模拟参数。
            physics_engine (gymapi.SimType): 物理引擎类型。
            sim_device (string): 模拟设备，'cuda'或'cpu'。
            headless (bool): 如果为True，则无渲染模式运行。
            eval_cfg (Dict, optional): 评估配置文件，默认为None。
            device (string): PyTorch设备类型。
            num_obs (int): 观测空间维度。
            num_privileged_obs (int): 特权观测空间维度。
            num_actions (int): 动作空间维度。
            num_eval_envs (int): 评估环境数。
            num_train_envs (int): 训练环境数。
            num_envs (int): 总环境数。
            obs_buf (torch.Tensor): 观测缓冲区。
            rew_buf (torch.Tensor): 奖励缓冲区。
            rew_buf_pos (torch.Tensor): 正奖励缓冲区。
            rew_buf_neg (torch.Tensor): 负奖励缓冲区。
            reset_buf (torch.Tensor): 重置缓冲区。
            episode_length_buf (torch.Tensor): 每个环境的episode长度缓冲区。
            time_out_buf (torch.Tensor): 超时缓冲区。
            privileged_obs_buf (torch.Tensor): 特权观测缓冲区。
            extras (dict): 存储额外信息的字典。
            enable_viewer_sync (bool): 是否启用查看器同步。
            viewer (gymapi.Viewer): 查看器实例。

        Methods:
            create_sim: 创建模拟环境的方法。

        Examples:
            >>> cfg = load_config()  # 加载配置文件
            >>> sim_params = get_sim_params()  # 获取模拟参数
            >>> task = BaseTask(cfg, sim_params, 'SIM_PHYSX', 'cuda:0', False)  # 创建BaseTask实例

        Note:
            - 如果headless为True，则不会创建查看器。
            - 如果sim_device为'cuda'且sim_params.use_gpu_pipeline为True，则设备设置为GPU。
            - 如果headless为False，则会创建查看器并订阅键盘事件。
        """
        self.gym = gymapi.acquire_gym()  # 获取gymapi的句柄

        # 如果physics_engine是字符串"SIM_PHYSX",则将其转换为gymapi.SIM_PHYSX
        if isinstance(physics_engine, str) and physics_engine == "SIM_PHYSX":
            physics_engine = gymapi.SIM_PHYSX

        self.sim_params = sim_params  # 模拟参数
        self.physics_engine = physics_engine  # 物理引擎类型
        self.sim_device = sim_device  # 模拟设备
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)  # 解析设备字符串
        self.headless = headless  # 是否无渲染模式

        # 如果模拟在GPU上且use_gpu_pipeline为True,则环境设备为GPU,否则返回的张量会被physX复制到CPU
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id  # 渲染用的图形设备,-1表示不渲染
        if self.headless == True:
            self.graphics_device_id = self.sim_device_id  # 如果是无渲染模式,则设置图形设备为-1

        self.num_obs = cfg.env.num_observations  # 观测空间维度
        self.num_privileged_obs = cfg.env.num_privileged_obs  # 特权观测空间维度
        self.num_actions = cfg.env.num_actions  # 动作空间维度

        # 如果提供了评估配置文件,则按照评估和训练环境数量分别设置
        if eval_cfg is not None:
            self.num_eval_envs = eval_cfg.env.num_envs  # 评估环境数
            self.num_train_envs = cfg.env.num_envs  # 训练环境数
            self.num_envs = self.num_eval_envs + self.num_train_envs  # 总环境数
        else:
            self.num_eval_envs = 0
            self.num_train_envs = cfg.env.num_envs
            self.num_envs = cfg.env.num_envs

        # 设置PyTorch JIT的优化标志
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # 分配PyTorch缓冲区
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_pos = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_neg = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,
                                              dtype=torch.float)
        # self.num_privileged_obs = self.num_obs

        self.extras = {}  # 存储额外信息的字典

        self.create_sim()  # 创建模拟环境
        self.gym.prepare_sim(self.sim)  # 准备模拟

        self.enable_viewer_sync = True  # 启用查看器同步(todo:从配置文件读取)
        self.viewer = None

        if self.headless == False:  # 如果不是无渲染模式
            # 创建查看器
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

            # 订阅查看器的键盘事件
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    def get_observations(self):
        """
        获取当前所有环境的观测值。

        该方法用于返回当前所有环境的观测值张量。观测值是理解和交互环境状态的基础，对于智能体的决策过程至关重要。

        Methods:
            使用 self.obs_buf 直接返回当前所有环境的观测值张量。

        Returns:
            torch.Tensor: 当前所有环境的观测值张量。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> observations = agent.get_observations()  # 获取当前所有环境的观测值
            >>> print(observations)

        Note:
            观测值可能包含环境的位置、速度、角度等信息，这些信息是智能体学习和决策的基础。
        """
        return self.obs_buf  # 返回当前所有环境的观测值张量

    def get_privileged_observations(self):
        """
        获取当前所有环境的特权观测值。

        该方法用于返回当前所有环境的特权观测值张量。特权观测值通常包含了一些额外的信息，这些信息在标准观测值中不可用，但对于训练更高级的模型来说可能很有用。

        Methods:
            使用 self.privileged_obs_buf 直接返回当前所有环境的特权观测值张量。

        Returns:
            torch.Tensor: 当前所有环境的特权观测值张量。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> privileged_obs = agent.get_privileged_observations()  # 获取特权观测值
            >>> print(privileged_obs)

        Note:
            特权观测值可能包括但不限于环境的内部状态、隐藏参数等，这些信息对于一些特定的训练场景非常重要。
        """
        return self.privileged_obs_buf  # 返回当前所有环境的特权观测值张量

    def reset_idx(self, env_ids):
        """
        重置选定的环境中的机器人。

        该方法的目的是在仿真环境中重置指定ID的环境。这通常发生在环境达到终止状态或者需要重新开始仿真时。调用此方法可以确保环境被重置到初始状态。

        Args:
            env_ids (iterable): 需要重置的环境的ID列表。

        Raises:
            NotImplementedError: 当方法未被实现时抛出。这是一个提示，表明子类需要提供具体的实现细节。

        Methods:
            该方法预期在子类中被实现，以提供具体的重置逻辑。

        Examples:
            >>> env_manager = YourEnvManagerClass()  # 环境管理器的实例化
            >>> env_manager.reset_idx([0, 1, 2])  # 重置ID为0, 1, 2的环境

        Note:
            该方法是一个抽象方法，需要在继承该类的子类中实现具体的重置逻辑。
        """
        raise NotImplementedError  # 抛出未实现异常

    def reset(self):
        """
        重置所有环境中的机器人。

        该方法用于重置仿真环境中的所有机器人到初始状态。这是在仿真开始或者需要重新开始仿真时非常有用的一个步骤。它首先通过调用 `reset_idx` 方法重置所有环境，随后通过执行一个空动作来获取重置后的观测值和特权观测值。

        Returns:
            tuple: 包含观测值和特权观测值的元组。观测值用于普通的决策过程，而特权观测值可能包含了一些额外的信息，这些信息在标准观测值中不可用，但对于训练更高级的模型来说可能很有用。

        Methods:
            - `reset_idx`: 调用此方法以重置所有环境。
            - `step`: 执行一步空动作，用于获取重置后的观测值和特权观测值。

        Examples:
            >>> env_manager = YourEnvManagerClass()  # 环境管理器的实例化
            >>> obs, privileged_obs = env_manager.reset()  # 重置所有环境并获取观测值
            >>> print(obs, privileged_obs)

        Note:
            该方法确保所有环境被重置并准备好接受新的指令。返回的观测值和特权观测值对于初始化智能体的状态非常重要。
        """
        self.reset_idx(torch.arange(self.num_envs, device=self.device))  # 调用reset_idx方法重置所有环境中的机器人
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        )  # 执行一步空动作以获取重置后的观测值和特权观测值
        return obs, privileged_obs  # 返回观测值和特权观测值

    def step(self, actions):
        """
        对所有环境执行一步动作。

        该方法的目标是在仿真环境中对所有环境执行指定的动作。这是环境与智能体交互的基础，允许智能体通过其动作影响环境。该方法预期在具体的仿真环境实现中被覆写，以提供具体的动作执行逻辑。

        Args:
            actions (torch.Tensor): 所有环境的动作张量。这个张量包含了对每个环境应该执行的动作。

        Raises:
            NotImplementedError: 当方法未被实现时抛出。这是一个提示，表明子类需要提供具体的实现细节。

        Methods:
            该方法预期在子类中被实现，以提供具体的动作执行逻辑。

        Examples:
            >>> env_manager = YourEnvManagerClass()  # 环境管理器的实例化
            >>> actions = torch.tensor([[1, 0], [0, 1]])  # 假设的动作张量，适用于两个环境
            >>> env_manager.step(actions)  # 对所有环境执行动作

        Note:
            该方法是仿真环境与智能体交互的核心部分，需要根据具体的仿真环境和智能体能力来实现。
        """
        raise NotImplementedError  # 抛出未实现异常

    def render_gui(self, sync_frame_time=True):
        """
        渲染图形用户界面。

        此方法在有可视化窗口时负责处理窗口事件、同步帧时间，并渲染图形界面。根据sync_frame_time参数决定是否同步帧时间。
        如果启用了视图同步，将同步图形步骤和绘制视图器；如果未启用，则仅处理视图器事件。

        Attributes:
            sync_frame_time (bool): 是否同步帧时间，默认为True。

        Methods:
            gym.query_viewer_has_closed: 查询视图器是否已关闭。
            gym.query_viewer_action_events: 查询视图器的动作事件。
            gym.fetch_results: 从模拟器中获取结果。
            gym.step_graphics: 同步图形步骤。
            gym.draw_viewer: 绘制视图器。
            gym.sync_frame_time: 同步帧时间。
            gym.poll_viewer_events: 轮询视图器事件。

        Raises:
            SystemExit: 如果检测到视图器已关闭或用户触发退出事件，则退出程序。

        Examples:
            >>> simulator = YourSimulatorClass()  # 创建模拟器实例
            >>> simulator.render_gui()  # 调用render_gui方法来渲染图形用户界面

        Note:
            - 如果设备不是CPU，会从模拟器中获取结果。
            - 视图同步开关可以通过键盘事件来控制。
        """

        if self.viewer:  # 检查是否存在可视化窗口
            if self.gym.query_viewer_has_closed(self.viewer):  # 查询视图器是否已关闭
                sys.exit()  # 如果视图器已关闭，退出程序

            for evt in self.gym.query_viewer_action_events(self.viewer):  # 查询视图器的动作事件
                if evt.action == "QUIT" and evt.value > 0:  # 如果用户触发退出事件
                    sys.exit()  # 退出程序
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:  # 如果触发视图同步开关事件
                    self.enable_viewer_sync = not self.enable_viewer_sync  # 切换视图同步状态

            if self.device != 'cpu':  # 如果设备不是CPU
                self.gym.fetch_results(self.sim, True)  # 从模拟器中获取结果

            if self.enable_viewer_sync:  # 如果启用了视图同步
                self.gym.step_graphics(self.sim)  # 同步图形步骤
                self.gym.draw_viewer(self.viewer, self.sim, True)  # 绘制视图器
                if sync_frame_time:  # 如果设置为同步帧时间
                    self.gym.sync_frame_time(self.sim)  # 进行帧时间同步
            else:  # 如果未启用视图同步
                self.gym.poll_viewer_events(self.viewer)  # 轮询视图器事件

    def close(self):
        """
        关闭模拟器和视图器。

        当程序结束运行时，此方法用于关闭并释放模拟器和视图器资源。如果程序在有图形界面的模式下运行，
        它还将负责关闭视图器。

        Methods:
            gym.destroy_viewer: 销毁创建的视图器实例。
            gym.destroy_sim: 销毁创建的模拟器实例。

        Examples:
            >>> simulator = YourSimulatorClass(headless=False)  # 创建模拟器实例
            >>> simulator.close()  # 调用close方法来关闭模拟器和视图器

        Note:
            - 无头模式是指程序运行时不包含图形用户界面的模式。
            - 如果程序在无头模式下运行，不需要销毁视图器。
        """

        if self.headless == False:  # 检查是否为有图形界面的模式
            self.gym.destroy_viewer(self.viewer)  # 如果不是无头模式，则关闭视图器
        self.gym.destroy_sim(self.sim)  # 关闭并释放模拟器资源
