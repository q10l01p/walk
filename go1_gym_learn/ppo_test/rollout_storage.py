import torch

from go1_gym_learn.utils import split_and_pad_trajectories


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.observation_histories = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.env_bins = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, obs_history_shape,
                 actions_shape, device='cpu'):
        """
        初始化用于存储环境转换的缓冲区。

        此构造函数初始化一个用于存储和管理环境转换数据的缓冲区，包括观察值、动作、奖励等。
        它为每个转换数据分配了适当的张量空间，并设置了用于训练的设备。

        Attributes:
            num_envs (int): 并行环境的数量。
            num_transitions_per_env (int): 每个环境存储的转换数量。
            obs_shape (tuple): 观察值的形状。
            privileged_obs_shape (tuple): 特权观察值的形状。
            obs_history_shape (tuple): 观察历史的形状。
            actions_shape (tuple): 动作的形状。
            device (str): 计算和存储张量的设备，默认为'cpu'。

        Examples:
            >>> buffer = RolloutBuffer(num_envs=5, num_transitions_per_env=100, obs_shape=(4,), privileged_obs_shape=(4,),
            >>>                        obs_history_shape=(10, 4), actions_shape=(1,), device='cuda')

        Note:
            - 特权观察值、观察历史和其他数据的形状必须根据具体任务进行设置。
        """
        # 设置设备
        self.device = device

        # 设置形状属性
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.obs_history_shape = obs_history_shape
        self.actions_shape = actions_shape

        # 初始化核心数据结构
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape,
                                                   device=self.device)
        self.observation_histories = torch.zeros(num_transitions_per_env, num_envs, *obs_history_shape,
                                                 device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # 初始化用于PPO算法的数据结构
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.env_bins = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # 设置环境和转换的数量
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # 初始化步数计数器
        self.step = 0

    def add_transitions(self, transition: Transition):
        """
        向缓冲区添加一步的转换数据。

        此方法负责将单步转换数据（包括观察值、动作、奖励等）添加到缓冲区中。
        如果缓冲区已满（即达到每个环境的转换数量上限），则会抛出异常。

        Args:
            transition (Transition): 包含一步转换数据的Transition对象。

        Raises:
            AssertionError: 如果尝试添加的转换超过了缓冲区的容量。

        Examples:
            >>> transition = Transition(observations, privileged_observations, observation_histories, actions, rewards, dones, values, actions_log_prob, action_mean, action_sigma, env_bins)
            >>> buffer.add_transitions(transition)

        Note:
            - Transition是一个具有多个字段的数据类，这些字段包含了环境转换所需的所有信息。
        """
        # 检查缓冲区是否已满
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        # 将转换数据复制到缓冲区中
        self.observations[self.step].copy_(transition.observations)
        self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.observation_histories[self.step].copy_(transition.observation_histories)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.env_bins[self.step].copy_(transition.env_bins.view(-1, 1))
        # 更新步数
        self.step += 1

    def clear(self):
        """
        重置模型的步数计数器。

        此方法用于将模型的内部步数计数器重置为0。通常在开始新的训练周期或评估周期时调用。

        Examples:
            >>> model = YourModelClass()
            >>> model.clear()  # 在开始新的训练或评估周期前重置步数计数器

        Note:
            - 假设模型类中有一个名为`step`的属性，用于追踪当前的步数。
        """
        # 将步数计数器重置为0
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        """
        计算回报值和优势值。

        该函数首先通过反向遍历每个环境的转换步骤来计算每一步的回报值，然后计算并标准化优势值。

        Attributes:
            last_values (torch.Tensor): 最后一步的价值估计。
            gamma (float): 折扣因子。
            lam (float): GAE（广义优势估计）的λ参数。

        Returns:
            None

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> agent.compute_returns(last_values=last_values, gamma=0.99, lam=0.95)  # 计算回报值和优势值
        """
        # 初始化优势值为0
        advantage = 0
        # 反向遍历每个环境的转换步骤
        for step in reversed(range(self.num_transitions_per_env)):
            # 如果是最后一步，下一步的价值为输入的最后一步价值
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                # 否则，下一步的价值为当前步骤的下一步价值
                next_values = self.values[step + 1]
            # 计算下一步是否不是终止状态，如果是终止状态则为0，否则为1
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # 计算TD误差
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # 计算优势值
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # 计算并存储回报值
            self.returns[step] = advantage + self.values[step]

        # 计算优势值
        self.advantages = self.returns - self.values
        # 标准化优势值
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        """
        计算并返回轨迹长度的平均值以及奖励的平均值。

        此函数通过分析完成标志（dones）和奖励（rewards）来计算整个批次中所有轨迹的平均长度和平均奖励。
        它首先将完成标志转换为一维数组，然后计算出每条轨迹的长度，最后计算这些长度的平均值以及奖励的平均值。

        Returns:
            tuple: 包含轨迹长度的平均值和奖励的平均值的元组。

        Examples:
            >>> model = YourModelClass()
            >>> avg_length, avg_reward = model.get_statistics()
            >>> print(f"Average Trajectory Length: {avg_length}, Average Reward: {avg_reward}")

        Note:
            - 假设`self.dones`和`self.rewards`已经按照[时间步, 环境]的格式排列。
        """
        # 获取完成标志
        done = self.dones
        # 将最后一个完成标志设置为1，确保每个轨迹都能被正确结束
        done[-1] = 1
        # 将完成标志转换为一维数组
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        # 计算完成标志的索引位置，并在开头添加一个-1作为初始索引
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        # 计算每条轨迹的长度
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        # 返回轨迹长度的平均值和奖励的平均值
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def get_statistics(self):
        """
        计算并返回轨迹长度的平均值以及奖励的平均值。

        此函数通过分析完成标志（dones）和奖励（rewards）来计算整个批次中所有轨迹的平均长度和平均奖励。
        它首先将完成标志转换为一维数组，然后计算出每条轨迹的长度，最后计算这些长度的平均值以及奖励的平均值。

        Returns:
            tuple: 包含轨迹长度的平均值和奖励的平均值的元组。

        Examples:
            >>> model = YourModelClass()
            >>> avg_length, avg_reward = model.get_statistics()
            >>> print(f"Average Trajectory Length: {avg_length}, Average Reward: {avg_reward}")

        Note:
            - 假设`self.dones`和`self.rewards`已经按照[时间步, 环境]的格式排列。
        """
        # 获取完成标志
        done = self.dones
        # 将最后一个完成标志设置为1，确保每个轨迹都能被正确结束
        done[-1] = 1
        # 将完成标志转换为一维数组
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        # 计算完成标志的索引位置，并在开头添加一个-1作为初始索引
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        # 计算每条轨迹的长度
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        # 返回轨迹长度的平均值和奖励的平均值
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """
        生成小批量数据供模型训练使用。

        此函数主要用于处理和准备数据，将环境观察、动作、回报等信息划分为多个小批量，以便于在训练过程中使用。
        它支持多个训练周期，每个周期都会重新洗牌索引以生成新的小批量数据。

        Attributes:
            num_mini_batches (int): 小批量的数量。
            num_epochs (int): 训练周期的数量，默认为8。

        Returns:
            Generator: 生成器，每次迭代返回一组小批量数据。

        Examples:
            >>> generator = model.mini_batch_generator(num_mini_batches=10, num_epochs=5)
            >>> for mini_batch in generator:
            >>>     # 使用mini_batch进行训练

        Note:
            - 该函数假设所有的数据已经通过属性如self.observations等存储在类实例中。
            - 使用randperm来随机生成索引，确保每个小批量数据的随机性。
        """
        # 计算每个小批量的大小
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        # 随机生成索引，不需要梯度
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # 将数据从二维展平为一维
        observations = self.observations.flatten(0, 1)
        privileged_obs = self.privileged_observations.flatten(0, 1)
        obs_history = self.observation_histories.flatten(0, 1)
        critic_observations = observations  # 使用普通观察值作为批评者观察值

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        old_env_bins = self.env_bins.flatten(0, 1)

        # 遍历每个训练周期
        for epoch in range(num_epochs):
            # 遍历每个小批量
            for i in range(num_mini_batches):
                # 计算当前小批量的起始和结束索引
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # 根据索引获取当前小批量的数据
                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                privileged_obs_batch = privileged_obs[batch_idx]
                obs_history_batch = obs_history[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                env_bins_batch = old_env_bins[batch_idx]

                # 生成器返回当前小批量的所有数据
                yield obs_batch, critic_observations_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, None, env_bins_batch

    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """
        生成循环神经网络训练所需的小批量数据。

        此函数用于将观察值、动作等数据按照小批量进行切分，以便于循环神经网络的训练过程中使用。
        它首先将轨迹数据进行分割和填充，然后在多个周期内生成小批量数据。

        Attributes:
            num_mini_batches (int): 每个周期内生成的小批量数量。
            num_epochs (int): 迭代周期数，默认为8。

        Yields:
            tuple: 包含观察值、批评者观察值、特权观察值、观察历史、动作、价值、优势、回报、
                   旧动作的对数概率、旧mu、旧sigma和掩码的批量数据。

        Examples:
            >>> generator = model.reccurent_mini_batch_generator(num_mini_batches=5)
            >>> for batch in generator:
            >>>     # 使用batch进行训练

        Note:
            - 该函数假设所有需要的数据属性（如观察值、动作等）已经作为类的属性存在。
            - 生成的小批量数据用于循环神经网络的训练，特别是在处理具有时间依赖性的任务时。
        """

        # 将观察值、特权观察值和观察历史分割并填充，以匹配轨迹长度
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        padded_privileged_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.privileged_observations,
                                                                                          self.dones)
        padded_obs_history_trajectories, trajectory_masks = split_and_pad_trajectories(self.observation_histories,
                                                                                       self.dones)
        # 将批评者的观察值设置为观察值的填充版本
        padded_critic_obs_trajectories = padded_obs_trajectories

        # 计算每个小批量的大小
        mini_batch_size = self.num_envs // num_mini_batches
        # 对于每个周期
        for ep in range(num_epochs):
            first_traj = 0
            # 对于每个小批量
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                # 处理完成标志，以便于计算轨迹的开始和结束
                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                # 计算当前小批量中的轨迹数量
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                # 从填充的轨迹中提取当前小批量的数据
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]
                privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]
                obs_history_batch = padded_obs_history_trajectories[:, first_traj:last_traj]

                # 提取动作、旧mu、旧sigma、回报、优势和价值的批量数据
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # 生成当前小批量的数据
                yield obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, masks_batch

                first_traj = last_traj
