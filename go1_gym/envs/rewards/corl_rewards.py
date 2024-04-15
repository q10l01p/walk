import torch
import numpy as np
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi


class CoRLRewards:
    def __init__(self, env):
        """
        CoRLRewards 类用于初始化与环境相关的奖励系统。

        该类的主要职责是存储对环境的引用，以便于后续计算奖励时可以访问环境的状态和属性。

        Attributes:
            env: 传入的环境对象，用于后续计算奖励时访问环境的状态和属性。

        Methods:
            __init__(self, env): 类的构造函数，用于初始化CoRLRewards类的实例。

        Examples:
            >>> env = YourEnvironmentClass()  # 假设有一个环境类的实例
            >>> rewards_system = CoRLRewards(env)  # 使用环境实例初始化奖励系统

        Note:
            - 这个类目前只包含了构造函数，后续可能会添加用于计算具体奖励的方法。
            - 传入的环境对象应该包含必要的状态和属性，以供奖励计算使用。
        """
        self.env = env  # 存储传入的环境对象引用

    def load_env(self, env):
        """
        在CoRLRewards类中添加一个方法用于加载新的环境对象。

        该方法允许在实例化后更新或更换CoRLRewards类实例所引用的环境对象。

        Attributes:
            env (object): 新的环境对象，用于替换原有的环境对象。

        Methods:
            load_env(self, env): 用于加载新的环境对象到CoRLRewards类实例中。

        Examples:
            >>> env = YourEnvironmentClass()  # 假设有一个环境类的实例
            >>> rewards_system = CoRLRewards(env)  # 使用环境实例初始化奖励系统
            >>> new_env = AnotherEnvironmentClass()  # 假设有另一个环境类的实例
            >>> rewards_system.load_env(new_env)  # 更新奖励系统中的环境对象

        Note:
            - 使用此方法可以在不重新创建CoRLRewards类实例的情况下，更新其内部使用的环境对象。
            - 确保新的环境对象与原有的环境对象具有相同的接口和属性，以便于无缝切换。
        """
        self.env = env  # 更新存储的环境对象引用

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        """
        在CoRLRewards类中添加一个私有方法，用于计算线性速度跟踪的奖励。

        该方法通过比较环境中的命令速度与实际基座线性速度之间的误差，来计算奖励值。使用指数函数对误差进行缩放，以得到最终的奖励值。

        Attributes:
            无额外属性。

        Methods:
            _reward_tracking_lin_vel(self): 计算线性速度跟踪的奖励值。

        Returns:
            torch.Tensor: 计算得到的线性速度跟踪奖励值，为一维张量。

        Examples:
            >>> rewards_system = CoRLRewards(env)  # 假设已有CoRLRewards类的实例
            >>> lin_vel_reward = rewards_system._reward_tracking_lin_vel()  # 计算线性速度跟踪奖励

        Note:
            - 该方法假设环境对象env中已经包含了必要的属性：commands（命令速度），base_lin_vel（基座线性速度），
                以及cfg.rewards.tracking_sigma（速度跟踪误差的缩放系数）。
            - 该方法是私有的，意味着它仅在CoRLRewards类内部使用，不应该被外部直接调用。
        """
        # 计算命令速度与基座线性速度在xy轴上的误差的平方和
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        # 使用指数函数对误差进行缩放，得到奖励值
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        计算基于角速度跟踪的奖励值。

        该方法旨在计算代理在跟踪给定的角速度命令（特别是偏航角速度）时的性能。通过比较期望的偏航角速度和实际的偏航角速度，
        并根据二者之间的误差计算奖励值。奖励值通过一个指数衰减函数计算，以确保误差越小，奖励越高。

        Attributes:
            env: 环境对象，包含了执行任务所需的所有信息和配置。
            env.commands: 一个张量，包含了所有环境的命令，其中包括期望的角速度。
            env.base_ang_vel: 一个张量，包含了所有环境中代理的基础角速度。
            env.cfg.rewards.tracking_sigma_yaw: 用于计算奖励值的偏航角速度跟踪误差的标准差。

        Returns:
            torch.Tensor: 代表每个环境基于角速度跟踪性能的奖励值。

        Examples:
            >>> env = YourEnvironmentClass()  # 环境类的实例化
            >>> reward = env._reward_tracking_ang_vel()  # 计算角速度跟踪奖励
            >>> print(reward)

        Note:
            - 该方法假设env对象已经包含了所有必要的配置和状态信息。
            - 奖励计算使用的指数衰减函数可以根据具体任务进行调整。
        """
        # 计算期望偏航角速度与实际偏航角速度之间的误差
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        # 根据误差计算奖励值，使用指数衰减函数确保误差越小，奖励越高
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)

    def _reward_lin_vel_z(self):
        """
        计算基于Z轴线速度的惩罚值。

        该方法旨在惩罚代理在Z轴（垂直方向）上的线速度，以鼓励代理保持稳定的飞行或移动状态，避免在垂直方向上的快速移动。
        通过计算代理在Z轴上的线速度的平方，来生成一个惩罚值，该值随着速度的增加而增加，从而鼓励代理减少在垂直方向上的速度。

        Attributes:
            env: 环境对象，包含了执行任务所需的所有信息和配置。
            env.base_lin_vel: 一个张量，包含了所有环境中代理的基础线速度。

        Returns:
            torch.Tensor: 代表每个环境基于Z轴线速度的惩罚值。

        Examples:
            >>> env = YourEnvironmentClass()  # 环境类的实例化
            >>> penalty = env._reward_lin_vel_z()  # 计算Z轴线速度惩罚
            >>> print(penalty)

        Note:
            - 该方法假设env对象已经包含了所有必要的配置和状态信息。
            - 该惩罚值可以与其他奖励值结合使用，以形成复合的奖励/惩罚机制。
        """
        # 计算代理在Z轴上的线速度的平方，作为惩罚值
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        """
        计算基于XY轴角速度的惩罚值。

        该方法旨在惩罚代理在XY平面（水平方向）上的角速度，以鼓励代理保持稳定的旋转状态，避免在水平方向上的快速旋转。
        通过计算代理在XY轴上的角速度的平方和，来生成一个惩罚值，该值随着角速度的增加而增加，从而鼓励代理减少在水平方向上的旋转速度。

        Attributes:
            env: 环境对象，包含了执行任务所需的所有信息和配置。
            env.base_ang_vel: 一个张量，包含了所有环境中代理的基础角速度。

        Returns:
            torch.Tensor: 代表每个环境基于XY轴角速度的惩罚值。

        Examples:
            >>> env = YourEnvironmentClass()  # 环境类的实例化
            >>> penalty = env._reward_ang_vel_xy()  # 计算XY轴角速度惩罚
            >>> print(penalty)

        Note:
            - 该方法假设env对象已经包含了所有必要的配置和状态信息。
            - 该惩罚值可以与其他奖励值结合使用，以形成复合的奖励/惩罚机制。
        """
        # 计算代理在XY轴上的角速度的平方和，作为惩罚值
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """
        计算基于非平面基座姿态的惩罚值。

        该方法旨在惩罚代理的基座如果没有保持在一个平面姿态上。通过计算代理相对于重力方向的投影在XY平面上的分量的平方和，
        来生成一个惩罚值，该值随着代理偏离平面姿态的程度增加而增加，从而鼓励代理保持平面姿态。

        Attributes:
            env: 环境对象，包含了执行任务所需的所有信息和配置。
            env.projected_gravity: 一个张量，包含了所有环境中代理相对于重力方向的投影。

        Returns:
            torch.Tensor: 代表每个环境基于非平面基座姿态的惩罚值。

        Examples:
            >>> env = YourEnvironmentClass()  # 环境类的实例化
            >>> penalty = env._reward_orientation()  # 计算非平面基座姿态惩罚
            >>> print(penalty)

        Note:
            - 该方法假设env对象已经包含了所有必要的配置和状态信息。
            - 该惩罚值可以与其他奖励值结合使用，以形成复合的奖励/惩罚机制。
        """
        # 计算代理相对于重力方向的投影在XY平面上的分量的平方和，作为惩罚值
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        """
        计算并返回由于扭矩产生的惩罚值。

        该方法旨在通过对环境中的扭矩应用平方和操作来计算惩罚值，以此来鼓励agent使用较小的扭矩值，从而提高其运动的效率和平滑性。

        Attributes:
            env (Environment): 代表当前agent所处的仿真环境，其中包含了扭矩信息。

        Returns:
            torch.Tensor: 所有环境中因扭矩产生的惩罚值，形状为 (num_envs,)，其中num_envs是环境的数量。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> penalty = agent._reward_torques()  # 计算扭矩惩罚
            >>> print(penalty)

        Note:
            - 该方法假设环境属性已经包含了扭矩信息。
            - 返回的惩罚值越大，表示使用的扭矩越大，相应的效率和平滑性越低。
        """
        # 对环境中的扭矩应用平方操作，并在所有环境上求和，以计算扭矩的惩罚值
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_dof_acc(self):
        """
        计算并返回关节速度变化的平方和，用于评估动作的平滑性。

        该函数通过计算当前时间步与上一时间步之间的关节速度差异的平方和，来评估动作的平滑性。这种方法可以用来鼓励agent执行更平滑的动作，减少突然的速度变化，从而在某些任务中获得更好的性能。

        Attributes:
            env (object): 环境对象，包含当前和上一时间步的关节速度(dof_vel和last_dof_vel)以及时间步长(dt)。

        Returns:
            torch.Tensor: 关节速度变化的平方和，形状为(batch_size,)，其中batch_size为环境的数量。

        Examples:
            >>> reward = self._reward_velocity_change()
            >>> print(reward)

        Note:
            - 该函数假设env对象已经有了last_dof_vel（上一时间步的关节速度）、dof_vel（当前时间步的关节速度）和dt（时间步长）这三个属性。
            - 返回的奖励值越小，表示动作越平滑。
        """
        # 计算当前时间步与上一时间步之间的关节速度差异，然后除以时间步长，得到速度变化率
        velocity_change = (self.env.last_dof_vel - self.env.dof_vel) / self.env.dt
        # 计算速度变化率的平方和，dim=1表示沿着批次的维度进行求和
        return torch.sum(torch.square(velocity_change), dim=1)

    def _reward_action_rate(self):
        """
        计算并返回动作变化率的平方和，用于评估动作的连续性。

        该方法通过计算当前动作与上一动作之间的差异的平方和，来鼓励agent执行更连续、更平滑的动作。这有助于避免执行剧烈或突然的动作变化，从而可能在某些任务中获得更好的性能。

        Attributes:
            env (object): 环境对象，包含当前和上一时间步的动作(actions和last_actions)。

        Returns:
            torch.Tensor: 动作变化率的平方和，形状为(batch_size,)，其中batch_size为环境的数量。

        Examples:
            >>> reward = self._reward_action_rate()
            >>> print(reward)

        Note:
            - 该函数假设env对象已经有了last_actions（上一时间步的动作）和actions（当前时间步的动作）这两个属性。
            - 返回的奖励值越小，表示动作变化越平滑。
        """
        # 计算当前动作与上一动作之间的差异
        action_diff = self.env.last_actions - self.env.actions
        # 计算动作差异的平方和，dim=1表示沿着批次的维度进行求和
        return torch.sum(torch.square(action_diff), dim=1)

    def _reward_collision(self):
        """
        计算并返回由于选定部位发生碰撞产生的惩罚值。

        该方法通过检查环境中指定部位的接触力是否超过预设阈值来判断是否发生碰撞，并对发生碰撞的情况进行计数，以此来鼓励agent避免与环境中的对象发生不必要的接触。

        Attributes:
            env (object): 环境对象，包含了接触力(contact_forces)信息以及需要被惩罚的接触部位索引(penalised_contact_indices)。

        Returns:
            torch.Tensor: 每个环境中发生碰撞的次数，形状为(batch_size,)，其中batch_size为环境的数量。

        Examples:
            >>> collision_penalty = self._reward_collision()
            >>> print(collision_penalty)

        Note:
            - 该方法假设env对象已经有了contact_forces（接触力）和penalised_contact_indices（需要被惩罚的接触部位索引）这两个属性。
            - 返回的惩罚值越大，表示发生碰撞的次数越多。
        """
        # 计算选定部位的接触力的范数，并判断是否超过0.1的阈值，超过则认为发生碰撞
        collision_detection = 1. * (
                    torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1)
        # 对发生碰撞的情况进行计数，dim=1表示沿着批次的维度进行求和
        return torch.sum(collision_detection, dim=1)

    def _reward_dof_pos_limits(self):
        """
        计算关节位置超出限制的惩罚。

        该方法用于计算机器人关节位置超出其预设限制时的惩罚值。通过比较当前关节位置与其限制范围，
        来确定是否有关节位置超出限制，并对这些超出限制的关节位置进行累加惩罚。

        Attributes:
            env (Environment): 仿真环境对象，包含关节位置(dof_pos)和关节位置限制(dof_pos_limits)信息。

        Returns:
            torch.Tensor: 所有环境中每个关节位置超出限制的累加惩罚值，形状为(batch_size,)。

        Examples:
            >>> env = YourEnvironmentClass()  # 环境类的实例化
            >>> penalty = env._reward_dof_pos_limits()  # 计算关节位置超出限制的惩罚
            >>> print(penalty)

        Note:
            - 关节位置限制由两部分组成，分别是最小限制和最大限制。
            - 超出最小限制和最大限制的关节位置将分别计算惩罚，并进行累加。
        """
        # 计算关节位置低于最小限制的部分，并将其限制在0以上，表示未超出限制的部分不计入惩罚
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)
        # 计算关节位置高于最大限制的部分，并将其限制在0以下，表示未超出限制的部分不计入惩罚
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        # 对所有超出限制的关节位置进行累加，得到总的惩罚值
        return torch.sum(out_of_limits, dim=1)

    def _reward_jump(self):
        """
        计算跳跃高度的奖励。

        此方法用于根据agent的跳跃高度与目标跳跃高度之间的差异来计算奖励。目标跳跃高度是通过环境中的指令和配置文件中设定的基础高度目标来确定的。
        奖励值是负的跳跃高度差的平方，意味着跳得越接近目标高度，奖励越高（损失越小）。

        Attributes:
            env (Environment): 仿真环境对象，包含基座位置(base_pos)、指令(commands)和配置(cfg)信息。

        Returns:
            torch.Tensor: 根据跳跃高度与目标跳跃高度差异计算出的奖励值。

        Examples:
            >>> env = YourEnvironmentClass()  # 环境类的实例化
            >>> reward = env._reward_jump()  # 计算跳跃高度的奖励
            >>> print(reward)

        Note:
            - 跳跃高度是通过agent的基座位置的第三个分量（Z轴高度）来确定的。
            - 目标跳跃高度是通过环境指令中的一个分量和配置中设定的基础高度目标来计算的。
        """
        # 设置参考高度为0，作为计算跳跃高度的基准
        reference_heights = 0
        # 计算agent的跳跃高度，即基座位置的Z轴高度减去参考高度
        body_height = self.env.base_pos[:, 2] - reference_heights
        # 根据环境指令和配置文件中的基础高度目标计算目标跳跃高度
        jump_height_target = self.env.commands[:, 3] + self.env.cfg.rewards.base_height_target
        # 计算奖励值，为负的跳跃高度差的平方，跳得越接近目标高度，奖励越高
        reward = - torch.square(body_height - jump_height_target)
        # 返回计算出的奖励值
        return reward

    def _reward_tracking_contacts_shaped_force(self):
        """
        计算基于接触力的形状奖励。

        此方法旨在通过比较期望的接触状态与实际接触力之间的差异来计算奖励。对于每只脚，如果期望接触状态为非接触（即期望值为0），
        则实际接触力越小奖励越高；反之，如果期望接触状态为接触（即期望值为1），则忽略该脚的接触力对奖励的影响。

        Attributes:
            env (Environment): 仿真环境对象，包含接触力(contact_forces)、脚的索引(feet_indices)、
                               期望的接触状态(desired_contact_states)以及配置(cfg)信息。

        Returns:
            torch.Tensor: 根据接触力计算得到的形状奖励，形状为(batch_size,)。

        Examples:
            >>> env = YourEnvironmentClass()  # 环境类的实例化
            >>> reward = env._reward_tracking_contacts_shaped_force()  # 计算基于接触力的形状奖励
            >>> print(reward)

        Note:
            - 接触力是通过计算脚与地面接触点的力的范数得到的。
            - 期望的接触状态是一个四元素向量，每个元素对应一只脚的期望接触状态（0表示非接触，1表示接触）。
        """
        # 计算每只脚的接触力的范数
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        # 获取期望的接触状态
        desired_contact = self.env.desired_contact_states

        # 初始化奖励为0
        reward = 0
        # 遍历四只脚
        for i in range(4):
            # 对于期望非接触的脚，计算基于接触力的惩罚，并累加到奖励中
            reward += - (1 - desired_contact[:, i]) * (
                    1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
        # 将累加的奖励平均分配到四只脚
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        """
        计算基于接触状态和脚部速度形状的奖励。

        该函数通过比较期望的接触状态和实际的脚部速度来计算奖励。奖励计算基于脚部速度与期望速度之间的差异，
        使用指数函数形式来塑造。这种方法旨在鼓励agent在期望接触状态时减小脚部的移动速度。

        Attributes:
            foot_velocities (torch.Tensor): 脚部速度的张量，形状为(num_envs, num_feet, 3)。
            desired_contact (torch.Tensor): 期望的接触状态张量，形状为(num_envs, num_feet)。

        Returns:
            torch.Tensor: 根据接触状态和脚部速度形状计算出的奖励值。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> reward = agent._reward_tracking_contacts_shaped_vel()  # 计算奖励
            >>> print(reward)

        Note:
            - 该函数假设有4只脚。
            - 奖励是通过对所有脚部的奖励求平均得到的。
        """
        # 计算每只脚的速度的范数，并将其重塑为(num_envs, num_feet)的形状
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        # 获取期望的接触状态
        desired_contact = self.env.desired_contact_states
        # 初始化奖励为0
        reward = 0
        # 遍历每只脚
        for i in range(4):
            # 根据脚部速度和期望接触状态计算奖励，并累加到总奖励中
            reward += - (desired_contact[:, i] * (
                    1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        # 将总奖励除以脚的数量，得到平均奖励
        return reward / 4

    def _reward_dof_pos(self):
        """
        计算基于关节位置的惩罚奖励。

        该函数计算当前关节位置与默认关节位置之间的差异，以此来生成一个惩罚项。该惩罚项旨在鼓励agent保持在一个相对默认状态的姿势，
        减少因过度偏离默认关节位置而可能导致的不稳定或非自然运动。

        Attributes:
            dof_pos (torch.Tensor): 当前关节位置的张量，形状为(num_envs, num_dofs)。
            default_dof_pos (torch.Tensor): 默认关节位置的张量，形状为(num_dofs,)。

        Returns:
            torch.Tensor: 基于关节位置差异计算出的惩罚奖励，形状为(num_envs,)。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> reward = agent._reward_dof_pos()  # 计算关节位置的惩罚奖励
            >>> print(reward)

        Note:
            - 该惩罚项通过计算当前关节位置与默认关节位置的平方差来实现。
            - 返回的奖励值越小，表示当前关节位置与默认位置的差异越小，反之亦然。
        """
        # 计算当前关节位置与默认关节位置的平方差，并在关节维度上求和，得到每个环境的惩罚值
        return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)

    def _reward_dof_vel(self):
        """
        计算基于关节速度的惩罚奖励。

        该函数计算所有关节速度的平方和，以此来生成一个惩罚项。该惩罚项旨在鼓励agent减少关节的快速移动，
        以避免可能导致的不稳定或非自然运动。

        Attributes:
            dof_vel (torch.Tensor): 当前关节速度的张量，形状为(num_envs, num_dofs)。

        Returns:
            torch.Tensor: 基于关节速度计算出的惩罚奖励，形状为(num_envs,)。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> reward = agent._reward_dof_vel()  # 计算关节速度的惩罚奖励
            >>> print(reward)

        Note:
            - 该惩罚项通过计算关节速度的平方和来实现。
            - 返回的奖励值越小，表示关节速度越小，反之亦然。
        """
        # 计算关节速度的平方和，并在关节维度上求和，得到每个环境的惩罚值
        return torch.sum(torch.square(self.env.dof_vel), dim=1)

    def _reward_action_smoothness_1(self):
        """
        计算基于动作平滑性的惩罚奖励。

        该函数通过计算连续动作之间的差异来生成一个惩罚项。这个惩罚项旨在鼓励agent执行更加平滑的动作序列，减少动作的突变，
        以此来提高运动的自然性和效率。特别地，第一步动作的变化不会被惩罚，以避免对初始状态的不必要惩罚。

        Attributes:
            joint_pos_target (torch.Tensor): 当前步骤的关节位置目标，形状为(num_envs, num_dofs)。
            last_joint_pos_target (torch.Tensor): 上一步骤的关节位置目标，形状为(num_envs, num_dofs)。
            last_actions (torch.Tensor): 上一步骤的动作，形状为(num_envs, num_dofs)。
            num_actuated_dof (int): 受控关节的数量。

        Returns:
            torch.Tensor: 基于动作平滑性差异计算出的惩罚奖励，形状为(num_envs,)。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> reward = agent._reward_action_smoothness_1()  # 计算动作平滑性的惩罚奖励
            >>> print(reward)

        Note:
            - 该惩罚项通过计算当前动作目标与上一动作目标之间的平方差来实现。
            - 第一步动作的变化不会被计入惩罚，以避免对初始动作的不必要惩罚。
        """
        # 计算当前动作目标与上一动作目标之间的平方差
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - self.env.last_joint_pos_target[:,
                                                                                       :self.env.num_actuated_dof])
        # 忽略第一步动作的变化，不对其进行惩罚
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)
        # 在关节维度上求和，得到每个环境的惩罚值
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        """
        计算基于动作平滑性的奖励，考虑连续三个时间步的动作变化。

        该函数通过计算连续三个时间步的关节目标位置的二次差分来评估动作的平滑性。这种方法旨在鼓励agent采取更加平滑的动作，
        减少动作之间的剧烈变化，以此来提高运动的自然性和效率。特别地，该函数会忽略仿真的前两个时间步，以避免在仿真开始时由于缺乏足够历史信息而导致的不准确评估。

        Attributes:
            joint_pos_target (torch.Tensor): 当前时间步的关节目标位置，形状为(num_envs, num_dofs)。
            last_joint_pos_target (torch.Tensor): 上一个时间步的关节目标位置，形状为(num_envs, num_dofs)。
            last_last_joint_pos_target (torch.Tensor): 上上个时间步的关节目标位置，形状为(num_envs, num_dofs)。
            last_actions (torch.Tensor): 上一个时间步的动作，形状为(num_envs, num_dofs)。
            last_last_actions (torch.Tensor): 上上个时间步的动作，形状为(num_envs, num_dofs)。

        Returns:
            torch.Tensor: 基于动作平滑性计算出的奖励值，形状为(num_envs,)。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> reward = agent._reward_action_smoothness_2()  # 计算动作平滑性的奖励
            >>> print(reward)

        Note:
            - 该奖励项通过计算关节目标位置的二次差分来实现，更加关注动作变化的变化率。
            - 通过忽略仿真的前两个时间步，可以避免在仿真开始阶段由于缺乏历史信息而导致的评估不准确。
        """
        # 计算连续三个时间步的关节目标位置的二次差分
        diff = torch.square(
            self.env.joint_pos_target[:, :self.env.num_actuated_dof] -
            2 * self.env.last_joint_pos_target[:, :self.env.num_actuated_dof] +
            self.env.last_last_joint_pos_target[:, :self.env.num_actuated_dof])
        # 忽略仿真的前两个时间步
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # 忽略第一个时间步
        diff = diff * (self.env.last_last_actions[:, :self.env.num_dof] != 0)  # 忽略第二个时间步
        # 在关节维度上求和，得到每个环境的奖励值
        return torch.sum(diff, dim=1)

    def _reward_feet_slip(self):
        """
        计算基于脚部滑动的惩罚奖励。

        该函数通过检测脚部是否在接触地面时发生滑动来计算惩罚奖励。如果脚部在接触地面时速度较大，即发生滑动，
        则会产生惩罚。该方法旨在鼓励agent在行走或跑动时保持脚部稳定，减少滑动，以提高运动的稳定性和效率。

        Attributes:
            contact_forces (torch.Tensor): 接触力的张量，形状为(num_envs, num_feet, 3)。
            feet_indices (list[int]): 脚部索引的列表。
            foot_velocities (torch.Tensor): 脚部速度的张量，形状为(num_envs, num_feet, 3)。

        Returns:
            torch.Tensor: 基于脚部滑动计算出的惩罚奖励，形状为(num_envs,)。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> reward = agent._reward_feet_slip()  # 计算脚部滑动的惩罚奖励
            >>> print(reward)

        Note:
            - 该惩罚项通过检测脚部在接触地面时的滑动情况来实现。
            - 通过考虑脚部的接触状态和速度，可以更准确地评估脚部滑动的情况。
        """
        # 检测脚部是否与地面接触，接触力大于1视为接触
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        # 结合当前和上一时刻的接触状态，以考虑接触的连续性
        contact_filt = torch.logical_or(contact, self.env.last_contacts)
        # 更新上一时刻的接触状态
        self.env.last_contacts = contact
        # 计算脚部在水平面上的速度的平方
        foot_velocities = torch.square(
            torch.norm(self.env.foot_velocities[:, :, 0:2], dim=2).view(self.env.num_envs, -1))
        # 计算在接触状态下脚部滑动的惩罚奖励
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        # 返回惩罚奖励
        return rew_slip

    def _reward_feet_contact_vel(self):
        """
        计算基于脚部接触速度的惩罚奖励。

        该函数通过检测脚部在接近地面时的速度来计算惩罚奖励。如果脚部在接近地面时速度较大，则会产生惩罚。
        该方法旨在鼓励agent在接触地面时减少脚部的移动速度，以此来提高着陆的稳定性和减少冲击。

        Attributes:
            foot_positions (torch.Tensor): 脚部位置的张量，形状为(num_envs, num_feet, 3)。
            foot_velocities (torch.Tensor): 脚部速度的张量，形状为(num_envs, num_feet, 3)。

        Returns:
            torch.Tensor: 基于脚部接触速度计算出的惩罚奖励，形状为(num_envs,)。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> reward = agent._reward_feet_contact_vel()  # 计算脚部接触速度的惩罚奖励
            >>> print(reward)

        Note:
            - 该惩罚项通过检测脚部在接近地面时的速度来实现。
            - 通过考虑脚部的接近地面状态和速度，可以更准确地评估着陆时的稳定性。
        """
        # 设置参考高度为0，用于判断脚部是否接近地面
        reference_heights = 0
        # 判断脚部是否接近地面，高度差小于0.03视为接近
        near_ground = self.env.foot_positions[:, :, 2] - reference_heights < 0.03
        # 计算脚部速度的平方
        foot_velocities = torch.square(
            torch.norm(self.env.foot_velocities[:, :, 0:3], dim=2).view(self.env.num_envs, -1))
        # 计算在接近地面时脚部速度的惩罚奖励
        rew_contact_vel = torch.sum(near_ground * foot_velocities, dim=1)
        # 返回惩罚奖励
        return rew_contact_vel

    def _reward_feet_contact_forces(self):
        """
        计算基于脚部接触力的惩罚奖励。

        该函数通过计算脚部接触力与设定的最大接触力之间的差异来生成一个惩罚项。如果脚部接触力超过了最大接触力的阈值，
        则会产生惩罚。该方法旨在鼓励agent在行走或跑动时减少对地面的冲击力，以提高运动的稳定性和减少对机器人结构的损害。

        Attributes:
            contact_forces (torch.Tensor): 接触力的张量，形状为(num_envs, num_feet, 3)。
            feet_indices (list[int]): 脚部索引的列表。
            max_contact_force (float): 设定的最大接触力阈值。

        Returns:
            torch.Tensor: 基于脚部接触力计算出的惩罚奖励，形状为(num_envs,)。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> reward = agent._reward_feet_contact_forces()  # 计算脚部接触力的惩罚奖励
            >>> print(reward)

        Note:
            - 该惩罚项通过计算脚部接触力与最大接触力阈值之间的差异来实现。
            - 只有当脚部接触力超过最大阈值时，才会产生惩罚。
        """
        # 计算脚部接触力的范数，并与最大接触力阈值之间的差异
        # 如果差异小于0，则使用clip函数将其设置为0，表示没有超过阈值，不产生惩罚
        return torch.sum((torch.norm(self.env.contact_forces[:, self.env.feet_indices, :],
                                     dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_clearance_cmd_linear(self):
        """
        计算足部清晰度与命令线性关系的奖励。

        此方法根据足部的高度与期望高度之间的差异来计算奖励。奖励值越小表示足部高度与目标高度的匹配程度越高。

        Methods:
            torch.abs: 计算绝对值。
            torch.clip: 将输入张量的每个元素限制在[min, max]范围内。
            torch.square: 计算张量的平方。
            torch.sum: 计算张量的和。

        Attributes:
            phases (torch.Tensor): 表示步行相位的张量。
            foot_height (torch.Tensor): 足部的当前高度。
            target_height (torch.Tensor): 目标足部高度。
            rew_foot_clearance (torch.Tensor): 足部清晰度奖励。

        Returns:
            torch.Tensor: 每个环境的足部清晰度奖励总和。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> reward = agent._reward_feet_clearance_cmd_linear()  # 计算足部清晰度奖励
            >>> print(reward)

        Note:
            - 足部高度与目标高度的差异越小，奖励越高。
            - 该方法假设env属性已经包含了必要的环境信息，如足部位置、命令等。
        """
        # 计算步行相位，用于调整目标高度
        phases = 1 - torch.abs(1.0 - torch.clip((self.env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        # 获取足部的当前高度
        foot_height = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1)
        # 计算目标足部高度，包括一个固定偏移量
        target_height = self.env.commands[:, 9].unsqueeze(1) * phases + 0.02  # offset for foot radius 2cm
        # 计算足部清晰度奖励，考虑期望的接触状态
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.env.desired_contact_states)
        # 返回每个环境的足部清晰度奖励总和
        return torch.sum(rew_foot_clearance, dim=1)

    def _reward_feet_impact_vel(self):
        """
        计算基于脚部撞击速度的惩罚奖励。

        该函数通过检测脚部在接触地面时的垂直速度来计算惩罚奖励。如果脚部在接触地面时的垂直速度过大，即撞击速度大，
        则会产生惩罚。该方法旨在鼓励agent在接触地面时减少脚部的撞击速度，以此来提高着陆的平稳性和减少对机器人结构的损害。

        Attributes:
            prev_foot_velocities (torch.Tensor): 上一时间步脚部的速度张量，形状为(num_envs, num_feet)。
            contact_forces (torch.Tensor): 接触力的张量，形状为(num_envs, num_feet, 3)。
            feet_indices (list[int]): 脚部索引的列表。

        Returns:
            torch.Tensor: 基于脚部撞击速度计算出的惩罚奖励，形状为(num_envs,)。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> reward = agent._reward_feet_impact_vel()  # 计算脚部撞击速度的惩罚奖励
            >>> print(reward)

        Note:
            - 该惩罚项通过检测脚部在接触地面时的垂直速度来实现。
            - 只有当脚部的垂直速度为负（向下移动）且接触力大于1时，才会产生惩罚。
        """
        # 获取上一时间步脚部的垂直速度
        prev_foot_velocities = self.env.prev_foot_velocities[:, :, 2].view(self.env.num_envs, -1)
        # 判断脚部是否与地面接触，接触力大于1视为接触
        contact_states = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1) > 1.0
        # 计算撞击速度的惩罚奖励，只考虑向下的速度（负值），速度限制在-100到0之间
        rew_foot_impact_vel = contact_states * torch.square(torch.clip(prev_foot_velocities, -100, 0))
        # 在脚部维度上求和，得到每个环境的惩罚值
        return torch.sum(rew_foot_impact_vel, dim=1)

    def _reward_collision(self):
        """
        计算基于选定身体部位碰撞的惩罚奖励。

        该函数通过检测选定身体部位是否发生碰撞来生成惩罚项。如果检测到碰撞（即接触力超过0.1的阈值），则会产生惩罚。
        该方法旨在鼓励agent避免与环境中的物体发生碰撞，以此来提高运动的安全性和避免潜在的损害。

        Attributes:
            contact_forces (torch.Tensor): 接触力的张量，形状为(num_envs, num_bodies, 3)。
            penalised_contact_indices (list[int]): 需要被惩罚碰撞的身体部位索引列表。

        Returns:
            torch.Tensor: 基于碰撞检测计算出的惩罚奖励，形状为(num_envs,)。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> reward = agent._reward_collision()  # 计算碰撞的惩罚奖励
            >>> print(reward)

        Note:
            - 该惩罚项通过检测选定身体部位的接触力是否超过0.1的阈值来判断是否发生碰撞。
            - 只有当检测到碰撞时，才会产生惩罚。
        """
        # 检测选定身体部位的接触力是否超过0.1的阈值，超过则视为发生碰撞
        collision_detection = 1. * (torch.norm(self.env.contact_forces[:,
                                               self.env.penalised_contact_indices, :], dim=-1) > 0.1)
        # 在身体部位维度上求和，得到每个环境的碰撞惩罚值
        return torch.sum(collision_detection, dim=1)

    def _reward_orientation_control(self):
        """
        计算基于基座朝向控制的惩罚奖励。

        该函数通过比较当前基座的朝向与期望朝向之间的差异来生成惩罚项。期望的朝向是根据给定的滚动和俯仰命令计算得出的。
        该方法旨在鼓励agent维持一个特定的朝向，以此来提高运动的稳定性和效率。

        Attributes:
            commands (torch.Tensor): 控制命令的张量，形状为(num_envs, num_commands)。
            gravity_vec (torch.Tensor): 重力向量，形状为(3,)。
            projected_gravity (torch.Tensor): 当前基座朝向下的重力投影，形状为(num_envs, 3)。

        Returns:
            torch.Tensor: 基于基座朝向控制差异计算出的惩罚奖励，形状为(num_envs,)。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> reward = agent._reward_orientation_control()  # 计算基座朝向控制的惩罚奖励
            >>> print(reward)

        Note:
            - 该惩罚项通过计算当前基座朝向与期望朝向之间的差异来实现。
            - 期望朝向是根据滚动和俯仰命令以及重力向量计算得出的。
        """
        # 获取滚动和俯仰命令
        roll_pitch_commands = self.env.commands[:, 10:12]
        # 根据俯仰命令计算四元数
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                         torch.tensor([1, 0, 0], device=self.env.device, dtype=torch.float))
        # 根据滚动命令计算四元数
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                          torch.tensor([0, 1, 0], device=self.env.device, dtype=torch.float))
        # 计算期望的基座四元数
        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        # 计算期望朝向下的重力投影
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.env.gravity_vec)
        # 计算当前基座朝向与期望朝向之间的差异
        return torch.sum(torch.square(self.env.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)

    def _reward_raibert_heuristic(self):
        """
        计算基于Raibert启发式方法的奖励。

        该函数通过比较当前脚步位置与基于Raibert启发式方法计算出的期望脚步位置之间的差异来生成奖励。
        Raibert启发式方法考虑了机器人的速度、期望的步态宽度和长度，以及脚步的相位，来计算每只脚的期望位置。
        该方法旨在鼓励agent采取更加稳定和高效的步态。

        Attributes:
            foot_positions (torch.Tensor): 脚部位置的张量，形状为(num_envs, num_feet, 3)。
            base_pos (torch.Tensor): 基座位置的张量，形状为(num_envs, 3)。
            base_quat (torch.Tensor): 基座四元数的张量，形状为(num_envs, 4)。
            commands (torch.Tensor): 控制命令的张量，形状为(num_envs, num_commands)。

        Returns:
            torch.Tensor: 基于Raibert启发式方法计算出的奖励，形状为(num_envs,)。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> reward = agent._reward_raibert_heuristic()  # 计算Raibert启发式方法的奖励
            >>> print(reward)

        Note:
            - 该奖励项通过比较当前脚步位置与期望脚步位置之间的差异来实现。
            - 期望脚步位置是根据机器人的速度、期望的步态宽度和长度，以及脚步的相位计算得出的。
        """
        # 将脚步位置转换到基座坐标系中
        cur_footsteps_translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # 计算期望的步态宽度和长度
        if self.env.cfg.commands.num_commands >= 13:
            desired_stance_width = self.env.commands[:, 12:13]
            desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2,
                                        -desired_stance_width / 2], dim=1)
        else:
            desired_stance_width = 0.3
            desired_ys_nom = torch.tensor(
                [desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2,
                 -desired_stance_width / 2], device=self.env.device).unsqueeze(0)

        if self.env.cfg.commands.num_commands >= 14:
            desired_stance_length = self.env.commands[:, 13:14]
            desired_xs_nom = torch.cat(
                [desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2,
                 -desired_stance_length / 2], dim=1)
        else:
            desired_stance_length = 0.45
            desired_xs_nom = torch.tensor(
                [desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2,
                 -desired_stance_length / 2], device=self.env.device).unsqueeze(0)

        # 计算Raibert启发式偏移
        phases = torch.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.env.commands[:, 4]
        x_vel_des = self.env.commands[:, 0:1]
        yaw_vel_des = self.env.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        # 更新期望的脚步位置
        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        # 合并期望的x和y位置
        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        # 计算当前脚步位置与期望脚步位置之间的差异
        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        # 计算奖励
        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
