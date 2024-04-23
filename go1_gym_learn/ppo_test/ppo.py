import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from go1_gym_learn.ppo_cse import ActorCritic
from go1_gym_learn.ppo_cse import RolloutStorage
from go1_gym_learn.ppo_cse import caches


class PPO_Args(PrefixProto):
    """
    定义PPO算法的参数配置类。

    该类继承自PrefixProto，用于存储和管理PPO算法的各种参数配置。

    Attributes:
        value_loss_coef (float): 值函数损失的系数。
        use_clipped_value_loss (bool): 是否使用裁剪的值函数损失。
        clip_param (float): PPO裁剪参数。
        entropy_coef (float): 熵系数，用于鼓励探索。
        num_learning_epochs (int): 学习的轮数。
        num_mini_batches (int): 每轮学习中的小批量数量。
        learning_rate (float): 学习率。
        adaptation_module_learning_rate (float): 适应模块的学习率。
        num_adaptation_module_substeps (int): 适应模块子步骤的数量。
        schedule (str): 学习率调整策略，可以是'adaptive'或'fixed'。
        gamma (float): 折扣因子。
        lam (float): GAE(Generalized Advantage Estimation)的λ参数。
        desired_kl (float): 期望的KL散度。
        max_grad_norm (float): 梯度裁剪的最大范数。
        selective_adaptation_module_loss (bool): 是否选择性地应用适应模块损失。
    """
    # algorithm
    value_loss_coef = 1.0  # 值函数损失的系数
    use_clipped_value_loss = True  # 是否使用裁剪的值函数损失
    clip_param = 0.2  # PPO裁剪参数
    entropy_coef = 0.01  # 熵系数，用于鼓励探索
    num_learning_epochs = 5  # 学习的总轮数
    num_mini_batches = 4  # 小批量的数量，小批量大小 = 环境数 * 步数 / 小批量数
    learning_rate = 1.e-3  # 学习率，原注释：5.e-4
    adaptation_module_learning_rate = 1.e-3  # 适应模块的学习率
    num_adaptation_module_substeps = 1  # 适应模块子步骤的数量
    schedule = 'adaptive'  # 学习率调整策略，可以是自适应或固定
    gamma = 0.99  # 折扣因子，用于计算未来奖励的当前价值
    lam = 0.95  # GAE(Generalized Advantage Estimation)中的λ参数
    desired_kl = 0.01  # 期望的KL散度，用于调整学习率
    max_grad_norm = 1.  # 梯度裁剪的最大范数

    selective_adaptation_module_loss = False  # 是否选择性地应用适应模块损失


class PPO:
    """
    实现PPO(Proximal Policy Optimization)算法的类。

    PPO是一种用于连续或离散动作空间的强化学习算法，它通过优化一个特定的目标函数来更新策略。

    Attributes:
        actor_critic (ActorCritic): 一个ActorCritic模型，用于生成动作和评估状态。
        device (str): 指定模型和数据应该在哪个设备上运行，'cpu'或'cuda'。

    Note:
        - 该类依赖于外部定义的`ActorCritic`类和`RolloutStorage`类。
        - `PPO_Args`类用于提供PPO算法的参数配置。
    """

    def __init__(self, actor_critic, device='cpu'):
        """
        初始化PPO对象。

        Args:
            actor_critic (ActorCritic): 用于策略和值函数的ActorCritic模型。
            device (str, optional): 计算将在其上执行的设备。默认为'cpu'。
        """
        self.device = device  # 设置运行设备

        # PPO组件
        self.actor_critic = actor_critic  # 设置ActorCritic模型
        self.actor_critic.to(device)  # 将模型移动到指定的设备
        self.storage = None  # 初始化存储，稍后设置
        # 初始化优化器
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=PPO_Args.learning_rate)
        # 初始化适应模块的优化器
        self.adaptation_module_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                      lr=PPO_Args.adaptation_module_learning_rate)
        # 如果存在解码器，则为其初始化优化器
        if self.actor_critic.decoder:
            self.decoder_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                lr=PPO_Args.adaptation_module_learning_rate)
        self.transition = RolloutStorage.Transition()  # 初始化转移存储

        self.learning_rate = PPO_Args.learning_rate  # 存储学习率

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape,
                     action_shape):
        """
        初始化存储器以保存环境的转换数据。

        该方法创建一个RolloutStorage实例，用于在训练过程中存储环境的观察值、动作等信息。

        Attributes:
            num_envs (int): 并行环境的数量。
            num_transitions_per_env (int): 每个环境存储的转换次数。
            actor_obs_shape (tuple): 演员观察值的形状。
            privileged_obs_shape (tuple): 特权观察值的形状。
            obs_history_shape (tuple): 观察历史的形状。
            action_shape (tuple): 动作的形状。

        Returns:
            None

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> agent.init_storage(num_envs=10, num_transitions_per_env=5, actor_obs_shape=(84, 84, 4),
            ...                    privileged_obs_shape=(84, 84, 4), obs_history_shape=(2, 84, 84, 4),
            ...                    action_shape=(1,))

        Note:
            - 'RolloutStorage'是用于存储和管理环境转换数据的类。
            - 该方法应在训练循环开始前调用，以确保所有必要的存储空间已经初始化。
        """
        # 创建RolloutStorage实例，用于存储环境转换数据
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                                      obs_history_shape, action_shape, self.device)

    def test_mode(self):
        """
        将actor_critic模型设置为测试模式。

        在测试模式下，模型的行为会有所不同，例如不会进行dropout等操作，以确保模型的行为是确定的。

        Attributes:
            actor_critic: 一个模型实例，具有测试和训练模式的切换能力。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> agent.test_mode()  # 将模型设置为测试模式

        Note:
            - 测试模式通常用于评估模型的性能，而不是在训练过程中。
        """
        self.actor_critic.test()  # 将actor_critic模型设置为测试模式

    def train_mode(self):
        """
        将模型设置为训练模式。
        此方法用于确保模型在训练时启用dropout和batch normalization等特性。

        Methods:
            train: 将模型设置为训练模式。

        Note:
            - 在进行模型训练之前调用此方法。
            - 与test_mode方法相对应，用于在模型评估和测试时切换模式。
        """
        self.actor_critic.train()  # 将actor_critic模型设置为训练模式

    def act(self, obs, privileged_obs, obs_history):
        """
        根据当前观察值和历史信息计算动作、值函数等。

        此方法使用actor_critic模型根据观察值和历史信息来计算动作和相关的值函数，并记录这些信息以供后续使用。

        Attributes:
            obs (torch.Tensor): 当前观察值。
            privileged_obs (torch.Tensor): 特权观察值，可能包含额外信息。
            obs_history (torch.Tensor): 观察值的历史序列。

        Returns:
            torch.Tensor: 计算得到的动作。

        Note:
            - 动作、值函数和动作的对数概率都会被记录下来。
            - 在环境执行step之前，需要记录当前的观察值和特权观察值。
        """
        # 计算动作和值函数
        self.transition.actions = self.actor_critic.act(obs_history).detach()  # 计算动作并从计算图中分离
        self.transition.values = self.actor_critic.evaluate(obs_history, privileged_obs).detach()  # 计算值函数并分离
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions).detach()  # 计算动作的对数概率并分离
        self.transition.action_mean = self.actor_critic.action_mean.detach()  # 记录动作均值
        self.transition.action_sigma = self.actor_critic.action_std.detach()  # 记录动作的标准差
        # 在环境执行step之前记录观察值和特权观察值
        self.transition.observations = obs  # 记录当前观察值
        self.transition.critic_observations = obs  # 记录用于评估的观察值
        self.transition.privileged_observations = privileged_obs  # 记录特权观察值
        self.transition.observation_histories = obs_history  # 记录观察历史
        return self.transition.actions  # 返回计算得到的动作

    def process_env_step(self, rewards, dones, infos):
        """
        处理环境步骤的结果，包括奖励、完成状态和额外信息。

        此方法用于处理环境返回的每一步结果，更新转换数据，并将其存储到RolloutStorage中。同时，对于完成的环境，重置actor_critic的状态。

        Attributes:
            rewards (torch.Tensor): 从环境中获得的奖励。
            dones (torch.Tensor): 表示环境是否完成的布尔值。
            infos (dict): 包含额外环境信息的字典，如'time_outs'和'env_bins'。

        Methods:
            add_transitions: 将当前转换数据添加到存储器中。
            clear: 清除当前转换数据。
            reset: 重置actor_critic的状态。

        Note:
            - 'time_outs'用于处理因超时而结束的情况，以便进行引导。
        """
        # 复制奖励值以避免修改原始数据
        self.transition.rewards = rewards.clone()
        # 记录完成状态
        self.transition.dones = dones
        # 从额外信息中记录环境分类信息
        self.transition.env_bins = infos["env_bins"]
        # 如果存在超时信息，则进行引导处理
        if 'time_outs' in infos:
            self.transition.rewards += PPO_Args.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        # 将当前转换数据添加到存储器中
        self.storage.add_transitions(self.transition)
        # 清除当前转换数据以准备下一步
        self.transition.clear()
        # 对于完成的环境，重置actor_critic的状态
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
        """
        根据最后的观察值计算回报。

        此方法使用最后的观察值来计算值函数，并基于此以及折扣因子和GAE(lambda)参数来计算回报。

        Attributes:
            last_critic_obs (torch.Tensor): 最后一步的观察值。
            last_critic_privileged_obs (torch.Tensor): 最后一步的特权观察值。

        Methods:
            evaluate: 使用actor_critic模型评估最后一步的观察值。
            compute_returns: 使用RolloutStorage中的方法计算回报。

        Note:
            - 计算回报是强化学习中的一个重要步骤，用于训练策略。
            - PPO_Args.gamma和PPO_Args.lam是预先定义的折扣因子和GAE(lambda)参数。
        """
        # 使用最后一步的观察值计算值函数
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs).detach()
        # 基于最后的值函数、折扣因子和GAE(lambda)参数计算回报
        self.storage.compute_returns(last_values, PPO_Args.gamma, PPO_Args.lam)

    @property
    def update(self):
        """
        执行模型的更新过程。

        此方法通过从存储器中生成的小批量数据进行多次迭代，计算损失并更新模型的参数。

        Attributes:
            None

        Returns:
            tuple: 包含各种损失值的元组，用于监控训练过程。

        Note:
            - 此方法涉及到多种损失的计算，包括代理损失、值函数损失和自适应模块损失等。
            - 通过梯度下降方法对模型参数进行更新。
        """
        # 初始化各种损失的累计值
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        mean_decoder_loss = 0
        mean_decoder_loss_student = 0
        mean_adaptation_module_test_loss = 0
        mean_decoder_test_loss = 0
        mean_decoder_test_loss_student = 0
        # 生成小批量数据的生成器
        generator = self.storage.mini_batch_generator(PPO_Args.num_mini_batches, PPO_Args.num_learning_epochs)
        # 遍历生成的小批量数据
        for obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, masks_batch, env_bins_batch in generator:

            # 计算动作的对数概率、值函数、均值和标准差
            self.actor_critic.act(obs_history_batch, masks=masks_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(obs_history_batch, privileged_obs_batch, masks=masks_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # 如果设置了期望的KL散度并且采用自适应调度，则根据KL散度调整学习率
            if PPO_Args.desired_kl != None and PPO_Args.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > PPO_Args.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < PPO_Args.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # 计算代理损失
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - PPO_Args.clip_param,
                                                                               1.0 + PPO_Args.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # 计算值函数损失
            if PPO_Args.use_clipped_value_loss:
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-PPO_Args.clip_param,
                                                                          PPO_Args.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # 计算总损失并进行梯度下降步骤
            loss = surrogate_loss + PPO_Args.value_loss_coef * value_loss - PPO_Args.entropy_coef * entropy_batch.mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), PPO_Args.max_grad_norm)
            self.optimizer.step()

            # 累计各种损失
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            # 自适应模块的梯度步骤
            # 此部分代码示例省略了具体的自适应模块更新逻辑

        # 计算损失的平均值
        num_updates = PPO_Args.num_learning_epochs * PPO_Args.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_adaptation_module_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_loss_student /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_adaptation_module_test_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_test_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_test_loss_student /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        # 清除存储器中的数据
        self.storage.clear()

        # 返回计算得到的平均损失值
        return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student
