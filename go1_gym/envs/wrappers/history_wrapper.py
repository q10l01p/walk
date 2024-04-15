import isaacgym
assert isaacgym
import torch
import gym


class HistoryWrapper(gym.Wrapper):
    """
    为环境添加观察历史记录的包装器。
    """

    def __init__(self, env):
        """
        为环境添加观察历史记录的包装器。

        这个类的目的是为环境的观测值添加一个历史记录功能，使得每个时间步的观测不仅包括当前的观测值，
        还包括之前一系列时间步的观测值。这对于需要考虑时间序列信息的任务特别有用。

        Attributes:
            env (gym.Env): 被包装的环境。
            obs_history_length (int): 需要保留的观测历史长度。
            num_obs_history (int): 历史观测值的总数量。
            obs_history (torch.Tensor): 保存历史观测值的张量。
            num_privileged_obs (int): 特权观测值的数量。

        Methods:
            __init__(self, env): 初始化包装器。

        Examples:
            >>> env = YourGymEnv()  # 创建你的gym环境
            >>> wrapped_env = HistoryWrapper(env)  # 使用HistoryWrapper包装环境
            >>> observation = wrapped_env.reset()  # 重置环境，获取初始观测值

        Note:
            - 这个包装器假设环境已经有一个名为`cfg`的配置属性，其中包含观测历史长度的配置。
            - 需要环境支持`num_envs`和`num_obs`属性，分别表示环境数量和每个环境的观测维度。
            - `num_privileged_obs`属性需要在外部设置。
        """
        super().__init__(env)  # 调用父类的初始化方法
        self.env = env  # 保存被包装的环境

        # 从环境配置中获取需要保留的观测历史长度
        self.obs_history_length = self.env.cfg.env.num_observation_history

        # 计算历史观测值的总数量
        self.num_obs_history = self.obs_history_length * self.env.num_obs

        # 初始化保存历史观测值的张量
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        # 特权观测值的数量，需要在外部设置
        self.num_privileged_obs = self.env.num_privileged_obs

    def step(self, action):
        """
        执行环境的一步操作，并返回相关信息。

        该方法接收一个动作作为输入，通过环境模型执行这个动作，然后返回观察结果、奖励、是否完成标志和额外信息。
        其中，额外信息中包含了特权观察信息和观察历史。

        Methods:
            env.step(action): 执行一个动作，并返回观察结果、奖励、是否完成标志和额外信息。

        Attributes:
            env (Environment): 代表当前与agent交互的环境。
            obs_history (torch.Tensor): 存储观察历史的张量。

        Returns:
            dict: 包含普通观察结果、特权观察结果和观察历史的字典。
            float: 此步骤获得的奖励。
            bool: 表示当前步骤后环境是否已完成。
            dict: 包含额外信息的字典。

        Examples:
            >>> action = [0.1, 0.2, 0.3]  # 假设的动作
            >>> step_info = agent.step(action)
            >>> print(step_info)

        Note:
            - 特权观察信息是指那些普通观察所不能提供，但对于学习算法有帮助的信息。
            - 观察历史是指过去若干步的观察结果的集合，有助于理解环境状态的变化。
        """
        # 执行动作并获取观察结果、奖励、完成标志和额外信息
        obs, rew, done, info = self.env.step(action)
        # 从额外信息中提取特权观察信息
        privileged_obs = info["privileged_obs"]
        # 更新观察历史，将新的观察结果拼接到历史记录中
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        # 返回包含观察结果、特权观察结果和观察历史的字典，以及奖励和完成标志
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}, rew, done, info

    def get_observations(self):
        """
        获取当前环境的观察结果，并更新观察历史。

        该方法用于从环境中获取当前的观察结果和特权观察结果，并将这些观察结果更新到观察历史中。

        Methods:
            env.get_observations(): 获取当前环境的普通观察结果。
            env.get_privileged_observations(): 获取当前环境的特权观察结果。

        Attributes:
            env (Environment): 代表当前与agent交互的环境。
            obs_history (torch.Tensor): 存储观察历史的张量。

        Returns:
            dict: 包含普通观察结果、特权观察结果和观察历史的字典。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> observations = agent.get_observations()  # 获取当前环境的观察结果
            >>> print(observations)

        Note:
            - 特权观察结果是指那些普通观察结果之外的、可能对学习过程有帮助的额外信息。
            - 观察历史用于在决策时考虑过去的状态，有助于理解环境的动态变化。
        """
        # 从环境中获取当前的普通观察结果
        obs = self.env.get_observations()
        # 从环境中获取当前的特权观察结果
        privileged_obs = self.env.get_privileged_observations()
        # 更新观察历史，将新的观察结果添加到历史记录中
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        # 返回包含普通观察结果、特权观察结果和观察历史的字典
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}

    def reset_idx(self, env_ids):
        """
        重置指定环境索引的观察历史。

        该方法用于重置给定环境ID列表中每个环境的观察历史。它首先调用父类的reset_idx方法来重置环境，
        然后将对应环境的观察历史清零。

        Methods:
            super().reset_idx(env_ids): 调用父类的reset_idx方法来重置指定的环境。

        Attributes:
            obs_history (torch.Tensor): 存储观察历史的张量。

        Parameters:
            env_ids (list[int]): 需要重置的环境ID列表。

        Returns:
            Any: 父类reset_idx方法的返回值。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> env_ids = [0, 2]  # 假设需要重置的环境ID
            >>> agent.reset_idx(env_ids)  # 重置指定环境的观察历史

        Note:
            - 该方法的调用确保了在环境重置时，相关的观察历史也会被清除，避免了历史信息的干扰。
            - 如果该方法没有被适时调用，可能会导致观察历史信息不准确，影响学习效果。
        """
        # 调用父类的reset_idx方法来重置指定的环境，并获取返回值
        ret = super().reset_idx(env_ids)
        # 将指定环境的观察历史清零
        self.obs_history[env_ids, :] = 0
        # 返回父类reset_idx方法的返回值
        return ret

    def reset(self):
        """
        重置环境并获取初始观察结果。

        该方法首先调用父类的reset方法来重置环境，然后获取特权观察结果，并将观察历史重置为零。

        Methods:
            super().reset(): 调用父类的reset方法来重置环境。
            env.get_privileged_observations(): 获取当前环境的特权观察结果。

        Attributes:
            env (Environment): 代表当前与agent交互的环境。
            obs_history (torch.Tensor): 存储观察历史的张量。

        Returns:
            dict: 包含普通观察结果、特权观察结果和观察历史的字典。

        Examples:
            >>> agent = YourAgentClass()  # 代理类的实例化
            >>> reset_info = agent.reset()  # 重置环境并获取初始观察结果
            >>> print(reset_info)

        Note:
            - 特权观察结果是指那些普通观察结果之外的、可能对学习过程有帮助的额外信息。
            - 在每次环境重置时，观察历史也会被重置，以确保从每个新的开始状态开始学习。
        """
        # 调用父类的reset方法来重置环境
        ret = super().reset()
        # 获取当前环境的特权观察结果
        privileged_obs = self.env.get_privileged_observations()
        # 将观察历史重置为零
        self.obs_history[:, :] = 0
        # 返回包含普通观察结果、特权观察结果和观察历史的字典
        return {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history}


if __name__ == "__main__":
    from tqdm import trange
    import matplotlib.pyplot as plt

    import ml_logger as logger

    from go1_gym_learn.ppo import Runner
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from go1_gym_learn.ppo.actor_critic import AC_Args

    from go1_gym.envs.base.legged_robot_config import Cfg
    from go1_gym.envs.mini_cheetah.mini_cheetah_config import config_mini_cheetah
    config_mini_cheetah(Cfg)

    test_env = gym.make("VelocityTrackingEasyEnv-v0", cfg=Cfg)
    env = HistoryWrapper(test_env)

    env.reset()
    action = torch.zeros(test_env.num_envs, 12)
    for i in trange(3):
        obs, rew, done, info = env.step(action)
        print(obs.keys())
        print(f"obs: {obs['obs']}")
        print(f"privileged obs: {obs['privileged_obs']}")
        print(f"obs_history: {obs['obs_history']}")

        img = env.render('rgb_array')
        plt.imshow(img)
        plt.show()
