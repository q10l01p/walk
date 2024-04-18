import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal


class AC_Args(PrefixProto, cli=False):
    """
    AC_Args类定义了Actor-Critic算法的参数配置。

    该类继承自PrefixProto，用于配置Actor-Critic算法的各种参数，包括策略网络和价值网络的结构、激活函数以及是否使用解码器等。

    Attributes:
        init_noise_std (float): 策略网络输出动作的初始噪声标准差。
        actor_hidden_dims (list): Actor网络的隐藏层维度。
        critic_hidden_dims (list): Critic网络的隐藏层维度。
        activation (str): 网络层使用的激活函数类型。
        adaptation_module_branch_hidden_dims (list): 适应模块分支的隐藏层维度。
        use_decoder (bool): 是否在网络中使用解码器。

    Examples:
        >>> ac_args = AC_Args()
        >>> print(ac_args.init_noise_std)
        1.0

    Note:
        - 激活函数支持多种类型，包括elu, relu, selu, crelu, lrelu, tanh, sigmoid等。
        - 该配置类不直接参与算法的运行，而是用于初始化算法时提供参数配置。
    """

    # 策略网络输出动作的初始噪声标准差
    init_noise_std = 1.0
    # Actor网络的隐藏层维度
    actor_hidden_dims = [512, 256, 128]
    # Critic网络的隐藏层维度
    critic_hidden_dims = [512, 256, 128]
    # 网络层使用的激活函数类型
    activation = 'elu'  # 可以是elu, relu, selu, crelu, lrelu, tanh, sigmoid中的任意一种

    # 适应模块分支的隐藏层维度
    adaptation_module_branch_hidden_dims = [256, 128]

    # 是否在网络中使用解码器
    use_decoder = False


class ActorCritic(nn.Module):
    """
    ActorCritic类实现了一个Actor-Critic架构的神经网络模型。

    该类继承自PyTorch的nn.Module，用于构建和训练Actor-Critic算法的网络结构，包括策略网络（Actor）和价值网络（Critic）。

    Attributes:
        is_recurrent (bool): 指示模型是否具有循环结构。
        decoder (bool): 根据AC_Args类中的配置决定是否使用解码器。
        num_obs_history (int): 观察历史的数量。
        num_privileged_obs (int): 特权观察的数量。
        adaptation_module (nn.Sequential): 适应模块，用于处理观察历史。
        actor_body (nn.Sequential): Actor网络的主体部分。
        critic_body (nn.Sequential): Critic网络的主体部分。
        std (nn.Parameter): 动作噪声的标准差。
        distribution (None): 动作分布，初始化为None。

    Methods:
        __init__: 构造函数，初始化网络结构。

    Examples:
        >>> actor_critic = ActorCritic(num_obs=24, num_privileged_obs=8, num_obs_history=4, num_actions=2)
        >>> print(actor_critic)

    Note:
        - 该模型支持配置是否使用解码器和选择不同的激活函数。
        - 通过AC_Args类配置模型参数。
    """

    is_recurrent = False  # 指示模型是否具有循环结构

    def __init__(self, num_obs, num_privileged_obs, num_obs_history, num_actions, **kwargs):
        """
        初始化ActorCritic模型。

        参数:
        - num_obs: 观察的数量。
        - num_privileged_obs: 特权观察的数量。
        - num_obs_history: 观察历史的数量。
        - num_actions: 可执行动作的数量。
        - **kwargs: 接收额外的未预期参数。
        """
        if kwargs:  # 如果存在未预期的参数
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))  # 打印忽略的参数
        self.decoder = AC_Args.use_decoder  # 根据AC_Args类中的配置决定是否使用解码器
        super().__init__()  # 调用父类的构造函数

        self.num_obs_history = num_obs_history  # 观察历史的数量
        self.num_privileged_obs = num_privileged_obs  # 特权观察的数量

        activation = get_activation(AC_Args.activation)  # 获取激活函数

        # 适应模块
        adaptation_module_layers = []  # 适应模块层列表
        adaptation_module_layers.append(nn.Linear(self.num_obs_history, AC_Args.adaptation_module_branch_hidden_dims[0]))  # 添加第一层
        adaptation_module_layers.append(activation)  # 添加激活函数
        for l in range(len(AC_Args.adaptation_module_branch_hidden_dims)):  # 遍历隐藏层维度
            if l == len(AC_Args.adaptation_module_branch_hidden_dims) - 1:  # 如果是最后一层
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))  # 连接到特权观察
            else:  # 如果不是最后一层
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l],
                              AC_Args.adaptation_module_branch_hidden_dims[l + 1]))  # 连接到下一层
                adaptation_module_layers.append(activation)  # 添加激活函数
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)  # 创建适应模块

        # 策略网络
        actor_layers = []  # Actor网络层列表
        actor_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs_history, AC_Args.actor_hidden_dims[0]))  # 添加第一层
        actor_layers.append(activation)  # 添加激活函数
        for l in range(len(AC_Args.actor_hidden_dims)):  # 遍历隐藏层维度
            if l == len(AC_Args.actor_hidden_dims) - 1:  # 如果是最后一层
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions))  # 连接到动作输出
            else:  # 如果不是最后一层
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1]))  # 连接到下一层
                actor_layers.append(activation)  # 添加激活函数
        self.actor_body = nn.Sequential(*actor_layers)  # 创建Actor网络

        # 价值网络
        critic_layers = []  # Critic网络层列表
        critic_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs_history, AC_Args.critic_hidden_dims[0]))  # 添加第一层
        critic_layers.append(activation)  # 添加激活函数
        for l in range(len(AC_Args.critic_hidden_dims)):  # 遍历隐藏层维度
            if l == len(AC_Args.critic_hidden_dims) - 1:  # 如果是最后一层
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1))  # 连接到价值输出
            else:  # 如果不是最后一层
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]))  # 连接到下一层
                critic_layers.append(activation)  # 添加激活函数
        self.critic_body = nn.Sequential(*critic_layers)  # 创建Critic网络

        print(f"Adaptation Module: {self.adaptation_module}")  # 打印适应模块信息
        print(f"Actor MLP: {self.actor_body}")  # 打印Actor网络信息
        print(f"Critic MLP: {self.critic_body}")  # 打印Critic网络信息

        # 动作噪声
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))  # 初始化动作噪声标准差
        self.distribution = None  # 初始化动作分布为None
        # 禁用参数验证以加速
        Normal.set_default_validate_args = False  # 禁用正态分布的参数验证

    @staticmethod
    def init_weights(sequential, scales):
        """
        初始化权重为正交初始化。

        该静态方法用于对给定的序列模型中的线性层进行正交初始化。

        Parameters:
            sequential (nn.Sequential): 需要初始化权重的序列模型。
            scales (list): 每一层的初始化增益。

        Note:
            - 当前此方法未被使用。
            - 只对序列模型中的线性层进行初始化。

        Examples:
            >>> model = ActorCritic(...)
            >>> ActorCritic.init_weights(model.actor_body, [1.0, 0.5, 0.1])
        """
        # not used at the moment  # 当前未使用
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]  # 对序列模型中的每个线性层进行正交初始化

    def reset(self, dones=None):
        """
        重置模型的状态。

        该方法提供了一种机制，用于在环境或某些条件下重置模型的内部状态。在这个示例中，`reset`方法被故意留空，
        作为一个可选实现的占位符，允许在需要时重置模型状态。

        Attributes:
            dones (torch.Tensor, optional): 一个布尔张量，指示哪些环境实例需要被重置。默认为None，表示不指定。

        Examples:
            >>> model = YourModelClass()  # 假设有一个模型类
            >>> model.reset()
            >>> # 或者，如果支持多环境实例：
            >>> dones = torch.tensor([True, False, True])  # 假设有三个环境实例，其中两个需要重置
            >>> model.reset(dones=dones)

        Note:
            - 在某些类型的模型中，如强化学习代理，可能需要在每个episode结束时重置模型的内部状态。
            - `dones`参数的具体使用取决于模型的设计和任务需求。在一些场景中，可能需要根据`dones`来决定是否重置特定的环境或代理的状态。
        """
        pass  # 在这个示例中，方法体被留空

    def forward(self):
        """
        前向传播方法的占位符。

        该方法是模型的核心，用于定义模型如何处理输入数据并产生输出。在这个示例中，`forward`方法被故意留空，
        并抛出`NotImplementedError`异常，表示该方法需要在子类中被具体实现。

        Raises:
            NotImplementedError: 总是抛出，提示该方法需要在子类中具体实现。

        Examples:
            # 假设有一个继承此类的子类
            class YourModelClass(YourBaseClass):
                def forward(self, input):
                    # 实现具体的前向传播逻辑
                    pass

        Note:
            - 在PyTorch中，自定义模型通常通过继承`nn.Module`类并重写`forward`方法来实现。
            - `forward`方法的具体参数和返回值取决于模型的设计和任务需求。
        """
        raise NotImplementedError  # 抛出未实现异常，提示用户需要在子类中实现该方法

    @property
    def action_mean(self):
        """
        获取当前概率分布的均值。

        该属性提供了一种访问模型当前动作概率分布均值的简便方法。均值反映了模型预期动作的中心位置，
        是理解模型行为倾向的关键指标。

        Returns:
            torch.Tensor: 当前动作概率分布的均值。

        Examples:
            >>> model = YourModelClass()  # 假设有一个模型类
            >>> mean = model.action_mean
            >>> print(mean)

        Note:
            - 该属性假设`self.distribution`已经被正确初始化，并且是一个正态分布（Normal distribution）。
            - 返回的均值张量的形状取决于分布的初始化参数。
        """
        return self.distribution.mean  # 返回当前概率分布的均值

    @property
    def action_std(self):
        """
        获取当前概率分布的标准差。

        该属性提供了一种访问模型当前动作概率分布标准差的简便方法。标准差是衡量动作输出变异性的重要指标，
        可以帮助理解模型的不确定性水平。

        Returns:
            torch.Tensor: 当前动作概率分布的标准差。

        Examples:
            >>> model = YourModelClass()  # 假设有一个模型类
            >>> std = model.action_std
            >>> print(std)

        Note:
            - 该属性假设`self.distribution`已经被正确初始化，并且是一个正态分布（Normal distribution）。
            - 返回的标准差张量的形状取决于分布的初始化参数。
        """
        return self.distribution.stddev  # 返回当前概率分布的标准差

    @property
    def entropy(self):
        """
        计算当前概率分布的熵。

        熵是衡量概率分布随机性的指标，高熵意味着分布具有较高的不确定性。
        本属性通过调用概率分布的`entropy`方法计算熵，然后在最后一个维度上求和，
        以得到一个标量值，表示整个概率分布的熵。

        Returns:
            torch.Tensor: 当前概率分布的熵，是一个标量。

        Examples:
            >>> model = YourModelClass()  # 假设有一个模型类
            >>> print(model.entropy)

        Note:
            - 该属性假设`self.distribution`已经被正确初始化，并且可以计算熵。
            - 熵的计算是在概率分布的支持维度上进行的，最后通过对最后一个维度求和来获得整个分布的熵。
        """
        return self.distribution.entropy().sum(dim=-1)  # 计算概率分布的熵，并在最后一个维度上求和

    def update_distribution(self, observation_history):
        """
        根据观测历史更新动作概率分布。

        该方法首先通过适应模块（adaptation_module）处理观测历史，以生成潜在表示（latent）。
        然后，将观测历史和潜在表示拼接后输入到行动者网络（actor_body），以计算动作的均值（mean）。
        最后，使用计算得到的均值和标准差（self.std）初始化一个正态分布（Normal），作为模型的动作概率分布。

        Attributes:
            observation_history (torch.Tensor): 观测历史，是一个张量。

        Note:
            - 该方法假设适应模块（adaptation_module）和行动者网络（actor_body）已经在模型中定义。
            - 标准差（self.std）是一个预先定义的模型参数，用于控制动作分布的离散程度。
            - 更新的动作概率分布存储在`self.distribution`中，供后续采样动作使用。
        """
        latent = self.adaptation_module(observation_history)  # 通过适应模块处理观测历史，生成潜在表示
        mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))  # 将观测历史和潜在表示拼接后，输入到行动者网络，计算动作的均值
        self.distribution = Normal(mean, mean * 0. + self.std)  # 使用均值和标准差初始化正态分布，作为动作概率分布

    def act(self, observation_history, **kwargs):
        """
        根据观测历史更新概率分布，并从中采样动作。

        该方法首先调用`update_distribution`方法，根据提供的观测历史更新模型的动作概率分布。
        然后，从更新后的概率分布中采样一个动作作为输出。

        Attributes:
            observation_history (torch.Tensor): 观测历史，是一个张量。
            **kwargs: 可变长的关键字参数，用于提供额外的信息，但在此方法中未直接使用。

        Returns:
            torch.Tensor: 从概率分布中采样得到的动作。

        Examples:
            >>> model = YourModelClass()  # 假设有一个模型类
            >>> observation_history = torch.randn(1, 10)  # 创建一个假设的观测历史
            >>> action = model.act(observation_history)
            >>> print(action)

        Note:
            - 该方法假设`update_distribution`方法已经在模型中定义，并能根据观测历史更新动作概率分布。
            - 动作的具体形状和内容取决于概率分布的类型和参数。
        """
        self.update_distribution(observation_history)  # 根据观测历史更新概率分布
        return self.distribution.sample()  # 从更新后的概率分布中采样动作

    def get_actions_log_prob(self, actions):
        """
        计算给定动作在当前策略下的对数概率。

        该方法利用模型中定义的概率分布（self.distribution），计算一组动作的对数概率。
        对数概率的计算是在分布的支持维度上进行的，最后通过对最后一个维度求和来获得每个动作的对数概率。

        Attributes:
            actions (torch.Tensor): 执行的动作，是一个张量。

        Returns:
            torch.Tensor: 给定动作的对数概率，是一个一维张量。

        Examples:
            >>> model = YourModelClass()  # 假设有一个模型类
            >>> actions = torch.randn(1, 4)  # 创建一个假设的动作张量
            >>> log_prob = model.get_actions_log_prob(actions)
            >>> print(log_prob)

        Note:
            - 该方法假设`self.distribution`已经被正确初始化，并且可以计算给定动作的对数概率。
            - 对数概率的求和是在动作张量的最后一个维度上进行的，这通常对应于动作空间的维度。
        """
        return self.distribution.log_prob(actions).sum(dim=-1)  # 计算动作的对数概率，并在最后一个维度上求和

    def act_expert(self, ob, policy_info={}):
        """
        专家模式下的行为决策函数。

        该方法基于观测历史和特权观测来决定行为，通过调用`act_teacher`方法来实现。它将观测历史和特权观测作为输入，
        并利用教师模型的行为生成网络来计算行为的均值。

        Attributes:
            ob (dict): 包含观测历史和特权观测的字典，键分别为"obs_history"和"privileged_obs"。
            policy_info (dict, optional): 策略相关信息的字典，用于存储额外的信息。默认为空字典。

        Returns:
            torch.Tensor: 行为的均值，由教师模型的行为生成网络计算得出。

        Examples:
            >>> model = YourModelClass()  # 假设有一个模型类
            >>> ob = {
            ...     "obs_history": torch.randn(1, 10),  # 创建一个包含观测历史的字典
            ...     "privileged_obs": torch.randn(1, 5)  # 创建一个包含特权观测的字典
            ... }
            >>> actions_mean = model.act_expert(ob)
            >>> print(actions_mean)

        Note:
            - 该方法假设`act_teacher`方法已经在模型中定义，并且可以处理观测历史和特权观测。
            - 与`act_student`方法不同，此方法使用的是教师模型，它可以访问额外的特权信息来做出决策。
        """
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"])  # 调用act_teacher方法，传入观测历史和特权观测，返回行为的均值

    def act_inference(self, ob, policy_info={}):
        """
        推理阶段的行为决策函数，用于学生模型。

        该方法基于观测历史来决定行为，通过调用`act_student`方法来实现。它将观测历史作为输入，
        并利用学生模型的适应模块和行为生成网络来计算行为的均值。

        Attributes:
            ob (dict): 包含观测历史的字典，键为"obs_history"。
            policy_info (dict, optional): 策略相关信息的字典，用于存储额外的信息，如潜在表示。默认为空字典。

        Returns:
            torch.Tensor: 行为的均值，由学生模型的行为生成网络计算得出。

        Examples:
            >>> model = YourModelClass()  # 假设有一个模型类
            >>> ob = {"obs_history": torch.randn(1, 10)}  # 创建一个包含观测历史的字典
            >>> policy_info = {}
            >>> actions_mean = model.act_inference(ob, policy_info=policy_info)
            >>> print(actions_mean)

        Note:
            - 该方法假设`act_student`方法已经在模型中定义，并且可以处理观测历史。
            - `policy_info`字典可以用于在方法调用间传递额外的信息，如潜在表示。
        """
        return self.act_student(ob["obs_history"], policy_info=policy_info)  # 调用act_student方法，传入观测历史和策略信息字典，返回行为的均值

    def act_student(self, observation_history, policy_info={}):
        """
        根据学生模型的观测历史来生成动作的均值。

        该方法首先通过适应模块（adaptation_module）处理观测历史，以生成潜在表示（latent）。
        然后，将观测历史和潜在表示拼接后输入到行动者网络（actor network），以生成动作的均值。
        同时，将潜在表示存储在策略信息（policy_info）字典中，以供后续使用。

        Attributes:
            observation_history (torch.Tensor): 学生模型的观测历史，是一个张量。
            policy_info (dict, optional): 一个字典，用于存储策略相关的额外信息。默认为空字典。

        Returns:
            torch.Tensor: 通过行动者网络计算得到的动作均值。

        Examples:
            >>> student_model = YourStudentModelClass()
            >>> observation_history = torch.randn(1, 10)  # 创建一个假设的观测历史
            >>> policy_info = {}
            >>> actions_mean = student_model.act_student(observation_history, policy_info)
            >>> print(actions_mean)
            >>> print(policy_info["latents"])

        Note:
            - 该方法假设适应模块（adaptation_module）和行动者网络（actor network）已经在模型中定义。
            - 潜在表示（latent）被转换为numpy数组，并存储在policy_info字典中，以便于后续处理。
        """
        latent = self.adaptation_module(observation_history)  # 通过适应模块处理观测历史，生成潜在表示
        actions_mean = self.actor_body(
            torch.cat((observation_history, latent), dim=-1))  # 将观测历史和潜在表示拼接后，输入到行动者网络，生成动作的均值
        policy_info["latents"] = latent.detach().cpu().numpy()  # 将潜在表示转换为numpy数组，并存储在policy_info字典中
        return actions_mean  # 返回动作的均值

    def act_teacher(self, observation_history, privileged_info, policy_info={}):
        """
        教师策略的行动函数，根据观测历史和特权信息生成行动的均值。

        该方法首先将观测历史和特权信息进行拼接，然后通过行动者网络(actor network)来生成行动的均值。
        同时，将特权信息作为潜在状态(latents)添加到策略信息(policy_info)字典中。

        Attributes:
            observation_history (torch.Tensor): 观测历史，是一个张量。
            privileged_info (torch.Tensor): 特权信息，也是一个张量。
            policy_info (dict, optional): 策略相关信息的字典，默认为空字典。

        Returns:
            torch.Tensor: 通过行动者网络计算得到的行动均值。

        Examples:
            >>> model = YourModelClass()  # 假设有一个模型类
            >>> observation_history = torch.randn(1, 10)  # 创建一个假设的观测历史
            >>> privileged_info = torch.randn(1, 5)  # 创建一个假设的特权信息
            >>> actions_mean = model.act_teacher(observation_history, privileged_info)
            >>> print(actions_mean)

        Note:
            - 观测历史和特权信息需要有相同的第一维度大小，即批次大小(batch size)。
            - 此方法假设行动者网络(actor network)已经在模型中定义。
            - 特权信息被认为是潜在状态(latents)，并添加到策略信息(policy_info)中，可能用于后续的决策过程或分析。
        """
        actions_mean = self.actor_body(
            torch.cat((observation_history, privileged_info), dim=-1))  # 将观测历史和特权信息在最后一个维度上拼接，并通过行动者网络计算行动的均值
        policy_info["latents"] = privileged_info  # 将特权信息作为潜在状态添加到策略信息中
        return actions_mean  # 返回计算得到的行动均值

    def evaluate(self, observation_history, privileged_observations, **kwargs):
        """
        根据观测历史和特权观测来评估价值。

        该方法通过将观测历史和特权观测拼接后，输入到批评者网络(critic network)中，以计算当前状态的价值。

        Attributes:
            observation_history (torch.Tensor): 观测历史，是一个张量。
            privileged_observations (torch.Tensor): 特权观测，也是一个张量。
            **kwargs: 可变长的关键字参数，用于提供额外的信息，但在此方法中未直接使用。

        Returns:
            torch.Tensor: 通过批评者网络计算得到的当前状态的价值。

        Examples:
            >>> model = YourModelClass()  # 假设有一个模型类
            >>> observation_history = torch.randn(1, 10)  # 创建一个假设的观测历史
            >>> privileged_observations = torch.randn(1, 5)  # 创建一个假设的特权观测
            >>> value = model.evaluate(observation_history, privileged_observations)
            >>> print(value)

        Note:
            - 观测历史和特权观测需要有相同的第一维度大小，即批次大小(batch size)。
            - 此方法假设批评者网络(critic network)已经在模型中定义。
        """
        value = self.critic_body(
            torch.cat((observation_history, privileged_observations), dim=-1))  # 将观测历史和特权观测在最后一个维度上拼接，并通过批评者网络计算价值
        return value  # 返回计算得到的价值

    def get_student_latent(self, observation_history):
        """
        根据观察历史通过适应模块计算学生模型的潜在表示。

        该方法接收一系列观察值，并通过适应模块（adaptation_module）处理这些观察值，
        以生成学生模型的潜在表示。这个潜在表示可以用于后续的决策过程或行为预测。

        Attributes:
            observation_history (torch.Tensor): 观察历史的张量，其形状和内容取决于具体任务。

        Returns:
            torch.Tensor: 学生模型的潜在表示，具体形状和内容取决于adaptation_module的实现。

        Examples:
            >>> student_model = YourStudentModelClass()
            >>> observation_history = torch.randn(1, 10)  # 假设有10个观察值
            >>> latent_representation = student_model.get_student_latent(observation_history)
            >>> print(latent_representation.shape)
            torch.Size([...])

        Note:
            - 这个方法的具体实现和返回值依赖于adaptation_module的结构和功能。
            - observation_history的具体形状取决于任务需求和数据结构。
        """
        return self.adaptation_module(observation_history)  # 通过适应模块处理观察历史，返回学生模型的潜在表示


def get_activation(act_name):
    """
    根据激活函数名称返回相应的PyTorch激活函数对象。

    该函数支持多种常见的激活函数，包括ELU、SELU、ReLU、LeakyReLU、Tanh和Sigmoid。
    如果传入的激活函数名称不被支持，将打印错误信息并返回None。

    Attributes:
        act_name (str): 激活函数的名称，期望为小写字符串。

    Returns:
        nn.Module: 对应于输入名称的PyTorch激活函数对象，如果名称无效则返回None。

    Examples:
        >>> activation = get_activation("relu")
        >>> print(activation)
        ReLU()

    Note:
        - "crelu"也会返回ReLU激活函数，尽管在PyTorch中没有直接对应的CReLU激活函数。
        - 如果传入未知的激活函数名称，函数将打印错误信息并返回None，而不是抛出异常。
    """
    if act_name == "elu":  # 如果激活函数名称为"elu"
        return nn.ELU()  # 返回ELU激活函数对象
    elif act_name == "selu":  # 如果激活函数名称为"selu"
        return nn.SELU()  # 返回SELU激活函数对象
    elif act_name == "relu":  # 如果激活函数名称为"relu"
        return nn.ReLU()  # 返回ReLU激活函数对象
    elif act_name == "crelu":  # 如果激活函数名称为"crelu"
        return nn.ReLU()  # 返回ReLU激活函数对象，尽管名称为crelu
    elif act_name == "lrelu":  # 如果激活函数名称为"lrelu"
        return nn.LeakyReLU()  # 返回LeakyReLU激活函数对象
    elif act_name == "tanh":  # 如果激活函数名称为"tanh"
        return nn.Tanh()  # 返回Tanh激活函数对象
    elif act_name == "sigmoid":  # 如果激活函数名称为"sigmoid"
        return nn.Sigmoid()  # 返回Sigmoid激活函数对象
    else:  # 如果激活函数名称不在支持的列表中
        print("invalid activation function!")  # 打印错误信息
        return None  # 返回None表示没有找到对应的激活函数
