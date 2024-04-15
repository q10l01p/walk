import numpy as np
import torch
from matplotlib import pyplot as plt


def is_met(scale, l2_err, threshold):
    """
    判断给定的L2误差是否满足特定阈值条件。

    根据输入的尺度(scale)、L2误差(l2_err)和阈值(threshold)，计算L2误差相对于尺度的比例，
    并判断该比例是否小于给定的阈值。如果是，则认为条件满足。

    Attributes:
        scale (float): 用于归一化L2误差的尺度。
        l2_err (float): 计算得到的L2误差。
        threshold (float): 判断条件是否满足的阈值。

    Returns:
        bool: 如果L2误差相对于尺度的比例小于阈值，则返回True，否则返回False。

    Examples:
        >>> is_met(10.0, 0.5, 0.1)
        True
        >>> is_met(10.0, 2.0, 0.1)
        False

    Note:
        - 该函数用于判断误差是否在可接受的范围内。
        - 尺度(scale)应为正数，L2误差(l2_err)应为非负数。
    """
    # 计算L2误差相对于尺度的比例，并判断是否小于给定阈值
    return (l2_err / scale) < threshold


def key_is_met(metric_cache, config, ep_len, target_key, env_id, threshold):
    """
    根据特定指标判断是否满足给定阈值条件。

    该函数通过查找指定环境ID和指标键值对应的指标值，并将其与环境的episode长度进行比较，
    以判断该指标值是否满足给定的阈值条件。

    Attributes:
        metric_cache (dict): 存储各个环境指标值的缓存。
        config (dict): 配置信息，本函数中未直接使用，但保留以便扩展。
        ep_len (int): 当前环境的episode长度。
        target_key (str): 需要检查的指标键名。
        env_id (int): 环境的唯一标识ID。
        threshold (float): 判断条件是否满足的阈值。

    Returns:
        bool: 如果指标值满足阈值条件，则返回True，否则返回False。

    Examples:
        >>> metric_cache = {'accuracy': {0: 0.8, 1: 0.9}}
        >>> config = {}
        >>> key_is_met(metric_cache, config, 100, 'accuracy', 0, 0.01)
        True

    Note:
        - 该函数用于根据环境的特定指标值判断是否达到了设定的目标。
        - 本例中，scale和l2_err被设定为示例值，实际应用中应根据metric_cache中的值进行计算。
    """
    # 初始化scale和l2_err为示例值，实际应用中需要根据metric_cache的值进行调整
    scale = 1
    l2_err = 0
    # 调用is_met函数，判断是否满足给定的阈值条件
    return is_met(scale, l2_err, threshold)


class Curriculum:
    def set_to(self, low, high, value=1.0):
        """
        将给定范围内的网格点权重设置为特定值。

        此方法用于初始化或调整分布，通过指定一个范围和值，将该范围内的所有网格点的权重设置为给定值。

        Attributes:
            low (np.ndarray): 指定范围的下界，形状为 (n,)，其中 n 为维度数。
            high (np.ndarray): 指定范围的上界，形状为 (n,)，其中 n 为维度数。
            value (float, optional): 要设置的权重值，默认为 1.0。

        Methods:
            np.logical_and: 用于计算两个数组中对应元素的逻辑与。
            all: 沿指定轴检查数组中的所有元素是否满足条件。

        Returns:
            None: 此方法不返回任何值，但会修改对象的 weights 属性。

        Raises:
            AssertionError: 如果指定的范围内没有网格点，则抛出断言错误。

        Examples:
            >>> grid = YourGridClass(...)  # 初始化你的网格类实例
            >>> grid.set_to(low=np.array([0, 0]), high=np.array([5, 5]), value=0.5)
            # 将所有在 [0, 0] 到 [5, 5] 范围内的网格点权重设置为 0.5

        Note:
            - 确保 low 和 high 的维度与网格维度相匹配。
            - 如果指定范围外的网格点，此方法不会对它们的权重产生影响。
        """
        # 计算网格点是否在指定的范围内
        inds = np.logical_and(
            self.grid >= low[:, None],  # 检查每个维度的网格点是否大于等于下界
            self.grid <= high[:, None]  # 检查每个维度的网格点是否小于等于上界
        ).all(axis=0)  # 确保所有维度的条件都满足

        # 断言：确保至少有一个网格点在指定的范围内
        assert len(inds) != 0, "You are initializing your distribution with an empty domain!"

        # 将范围内的网格点权重设置为指定值
        self.weights[inds] = value

    def __init__(self, seed, **key_ranges):
        """
        初始化参数空间的实例。

        此构造函数创建一个参数空间，其中包含了由用户指定的每个参数的离散化值。它使用随机种子来初始化随机数生成器，
        并根据用户提供的参数范围和离散化步长来计算每个参数的离散化值。

        Attributes:
            seed (int): 用于初始化随机数生成器的种子。
            key_ranges (dict): 参数的键和其范围及离散化步长的字典。

        Methods:
            np.random.RandomState: 初始化随机数生成器。
            np.linspace: 在指定的间隔内返回均匀间隔的数字。
            np.stack: 沿新轴连接数组序列。
            np.meshgrid: 生成坐标矩阵。

        Returns:
            None: 此方法不返回任何值，但会初始化多个实例属性。

        Examples:
            >>> param_space = ParamSpace(seed=42, param1=(0, 10, 5), param2=(-1, 1, 10))
            # 创建一个参数空间实例，其中param1在0到10之间离散化为5个值，param2在-1到1之间离散化为10个值

        Note:
            - key_ranges中的每个条目格式为 {参数名: (最小值, 最大值, 离散化步数)}。
            - 确保每个参数的最小值小于最大值。
        """
        # 初始化随机数生成器
        self.rng = np.random.RandomState(seed)

        # 初始化配置和索引字典
        self.cfg = cfg = {}
        self.indices = indices = {}
        # 遍历每个参数及其范围和步长
        for key, v_range in key_ranges.items():
            # 计算每个参数的离散化步长
            bin_size = (v_range[1] - v_range[0]) / v_range[2]
            # 使用np.linspace生成每个参数的离散化值
            cfg[key] = np.linspace(v_range[0] + bin_size / 2, v_range[1] - bin_size / 2, v_range[2])
            # 生成每个参数的索引值
            indices[key] = np.linspace(0, v_range[2] - 1, v_range[2])

        # 计算所有参数的最小值和最大值
        self.lows = np.array([range[0] for range in key_ranges.values()])
        self.highs = np.array([range[1] for range in key_ranges.values()])

        # 计算每个参数的离散化步长
        self.bin_sizes = {key: (v_range[1] - v_range[0]) / v_range[2] for key, v_range in key_ranges.items()}

        # 生成原始网格和索引网格
        self._raw_grid = np.stack(np.meshgrid(*cfg.values(), indexing='ij'))
        self._idx_grid = np.stack(np.meshgrid(*indices.values(), indexing='ij'))
        # 提取参数键
        self.keys = [*key_ranges.keys()]
        # 将原始网格和索引网格重塑为二维数组
        self.grid = self._raw_grid.reshape([len(self.keys), -1])
        self.idx_grid = self._idx_grid.reshape([len(self.keys), -1])

        # 计算网格中的点数
        self._l = l = len(self.grid[0])
        # 计算每个参数的离散化值数量
        self.ls = {key: len(self.cfg[key]) for key in self.cfg.keys()}

        # 初始化权重和索引
        self.weights = np.zeros(l)
        self.indices = np.arange(l)

    def __len__(self):
        """
        返回参数空间中的点的总数。

        此方法允许直接使用len()函数来获取参数空间实例中的点的总数，提高了代码的可读性和易用性。

        Returns:
            int: 参数空间中的点的总数。

        Examples:
            >>> param_space = ParamSpace(seed=42, param1=(0, 10, 5), param2=(-1, 1, 10))
            >>> print(len(param_space))
            # 输出参数空间中的点的总数

        Note:
            - 此方法依赖于在__init__方法中计算并存储的_l属性。
        """
        # 返回参数空间中的点的总数
        return self._l

    def __getitem__(self, *keys):
        """
        根据给定的键获取参数空间中的项。

        此方法允许用户通过参数名来访问参数空间中的特定参数值或子空间。如果提供多个键，
        则返回由这些键指定的参数形成的子空间。

        Attributes:
            keys (tuple): 一个包含一个或多个参数名的元组。

        Returns:
            np.ndarray: 如果提供单个键，返回与该键对应的参数值数组。
                        如果提供多个键，返回一个数组，其中包含由这些键指定的参数形成的子空间。

        Examples:
            >>> param_space = ParamSpace(seed=42, param1=(0, 10, 5), param2=(-1, 1, 10))
            >>> print(param_space['param1'])
            # 输出param1参数的所有离散化值
            >>> print(param_space['param1', 'param2'])
            # 输出由param1和param2参数形成的子空间

        Note:
            - 如果提供的键不在参数空间中，将抛出KeyError。
        """
        # 此处应实现根据keys访问参数空间的逻辑，示例代码未实现具体功能
        pass

    def update(self, **kwargs):
        """
        更新参数空间的配置。

        此方法允许动态更新参数空间的配置，例如调整参数的范围或离散化步长。传入的关键字参数应与初始化时相同的格式。

        Attributes:
            kwargs (dict): 包含参数名称及其新的范围和离散化步长的字典。

        Methods:
            无

        Returns:
            None: 此方法不返回任何值，但会根据提供的参数更新实例的配置。

        Examples:
            >>> param_space = ParamSpace(seed=42, param1=(0, 10, 5))
            >>> param_space.update(param1=(0, 20, 10))
            # 更新param1的范围为0到20，并将其离散化为10个值

        Note:
            - 更新操作可能会影响参数空间的大小和结构，需谨慎使用。
            - 确保传入的参数名称已存在于参数空间的配置中。
        """
        # 该方法的具体实现将依赖于kwargs中的参数，此处省略具体实现代码。
        pass

    def sample_bins(self, batch_size, low=None, high=None):
        """
        从参数空间中采样指定数量的点。

        此方法根据权重分布从参数空间中随机采样。可以指定采样的范围限制，如果未指定，则默认为均匀采样。

        Attributes:
            batch_size (int): 需要采样的点的数量。
            low (np.ndarray, optional): 采样范围的下界。默认为None。
            high (np.ndarray, optional): 采样范围的上界。默认为None。

        Methods:
            np.logical_and: 用于计算两个数组中对应元素的逻辑与。
            all: 沿指定轴检查数组中的所有元素是否满足条件。
            np.zeros_like: 返回与给定数组形状和类型相同的零数组。
            np.random.choice: 根据给定的概率分布随机选择元素。

        Returns:
            tuple: 包含两个元素的元组。
                   第一个元素是采样得到的点的坐标数组，形状为(batch_size, 参数数量)。
                   第二个元素是采样点在参数空间中的索引数组，形状为(batch_size,)。

        Examples:
            >>> param_space = ParamSpace(seed=42, param1=(0, 10, 5), param2=(-1, 1, 10))
            >>> samples, indices = param_space.sample_bins(10)
            # 从参数空间中随机采样10个点

        Note:
            - 如果指定了low和high，则只从指定范围内采样。
            - 如果未指定low和high，则默认对整个参数空间进行均匀采样。
        """
        if low is not None and high is not None:  # 如果给定了采样范围
            # 计算有效的索引，即在给定范围内的点
            valid_inds = np.logical_and(
                self.grid >= low[:, None],
                self.grid <= high[:, None]
            ).all(axis=0)
            # 创建一个临时权重数组，仅在有效索引处非零
            temp_weights = np.zeros_like(self.weights)
            temp_weights[valid_inds] = self.weights[valid_inds]
            # 根据权重分布选择索引
            inds = self.rng.choice(self.indices, batch_size, p=temp_weights / temp_weights.sum())
        else:  # 如果没有给定采样范围
            # 根据权重分布选择索引
            inds = self.rng.choice(self.indices, batch_size, p=self.weights / self.weights.sum())

        # 返回采样点的坐标和索引
        return self.grid.T[inds], inds

    def sample_uniform_from_cell(self, centroids):
        """
        从每个给定的单元格中均匀采样一个点。

        此方法根据提供的质心和参数空间的离散化步长，在每个参数的单元格内均匀采样一个点。

        Attributes:
            centroids (np.ndarray): 每个单元格的质心，形状为 (n_cells, n_params)，其中n_cells是单元格数量，n_params是参数数量。

        Methods:
            np.array: 创建数组。
            self.rng.uniform: 在给定的范围内均匀采样。

        Returns:
            np.ndarray: 采样得到的点，形状与centroids相同。

        Examples:
            >>> param_space = ParamSpace(seed=42, param1=(0, 10, 5), param2=(-1, 1, 10))
            >>> centroids = np.array([[2.5, 0], [7.5, 0]])
            >>> samples = param_space.sample_uniform_from_cell(centroids)
            # 从param1和param2构成的两个单元格中均匀采样点

        Note:
            - 确保centroids的形状正确，且每个质心都位于参数空间内。
        """
        # 将bin_sizes从字典转换为数组
        bin_sizes = np.array([*self.bin_sizes.values()])
        # 计算采样范围的上下界
        low, high = centroids - bin_sizes / 2, centroids + bin_sizes / 2
        # 在每个单元格内均匀采样一个点
        return self.rng.uniform(low, high)
        # .clip(self.lows, self.highs) 已注释的部分可以用于限制采样点不超出参数空间的边界，但在此示例中未启用

    def sample(self, batch_size, low=None, high=None):
        """
        从参数空间中采样指定数量的点，并在每个采样点的相应单元格内进行均匀采样。

        此方法首先根据权重分布和可选的范围限制从参数空间中随机选择单元格，然后在每个选定的单元格内进行均匀采样以获得最终的采样点。

        Attributes:
            batch_size (int): 需要采样的点的数量。
            low (np.ndarray, optional): 采样范围的下界。默认为None。
            high (np.ndarray, optional): 采样范围的上界。默认为None。

        Methods:
            sample_bins: 根据权重分布和可选的范围限制从参数空间中随机选择单元格。
            sample_uniform_from_cell: 在给定单元格内进行均匀采样。

        Returns:
            tuple: 包含两个元素的元组。
                   第一个元素是一个数组，其中包含采样得到的点。
                   第二个元素是一个数组，包含这些点对应的索引。

        Examples:
            >>> param_space = ParamSpace(seed=42, param1=(0, 10, 5), param2=(-1, 1, 10))
            >>> samples, inds = param_space.sample(batch_size=100)
            # 从参数空间中随机采样100个点

        Note:
            - 如果指定了low和high，则只在这个范围内采样。
            - 采样是在每个选定单元格的范围内均匀进行的。
        """
        # 首先从参数空间中根据权重分布和可选的范围限制选择单元格
        cgf_centroid, inds = self.sample_bins(batch_size, low=low, high=high)
        # 然后在每个选定的单元格内进行均匀采样以获得最终的采样点
        return np.stack([self.sample_uniform_from_cell(v_range) for v_range in cgf_centroid]), inds


class SumCurriculum(Curriculum):
    def __init__(self, seed, **kwargs):
        super().__init__(seed, **kwargs)

        self.success = np.zeros(len(self))
        self.trials = np.zeros(len(self))

    def update(self, bin_inds, l1_error, threshold):
        is_success = l1_error < threshold
        self.success[bin_inds[is_success]] += 1
        self.trials[bin_inds] += 1

    def success_rates(self, *keys):
        s_rate = self.success / (self.trials + 1e-6)
        s_rate = s_rate.reshape(list(self.ls.values()))
        marginals = tuple(i for i, key in enumerate(self.keys) if key not in keys)
        if marginals:
            return s_rate.mean(axis=marginals)
        return s_rate


class RewardThresholdCurriculum(Curriculum):
    def __init__(self, seed, **kwargs):
        """
        初始化参数空间的扩展实例，增加了与奖励和速度相关的属性。

        此构造函数在基类的基础上添加了几个与仿真环境中的代理性能相关的属性，如线性奖励、角速度奖励、线性速度、角速度和仿真时长。

        Attributes:
            seed (int): 用于初始化随机数生成器的种子。
            kwargs (dict): 包含参数名称及其范围和离散化步长的字典。

        Methods:
            super().__init__: 调用基类的构造函数来初始化参数空间。

        Returns:
            None: 此方法不返回任何值，但会初始化多个实例属性。

        Examples:
            >>> extended_param_space = ExtendedParamSpace(seed=42, param1=(0, 10, 5), param2=(-1, 1, 10))
            # 创建一个扩展的参数空间实例，其中包含了额外的与仿真环境性能相关的属性

        Note:
            - 此类假定基类已正确实现了参数空间的基本功能。
            - 新增的属性用于记录仿真过程中的各项指标。
        """
        # 调用基类的构造函数来初始化参数空间
        super().__init__(seed, **kwargs)

        # 初始化与仿真环境中的代理性能相关的属性
        self.episode_reward_lin = np.zeros(len(self))  # 记录每个参数组合下的线性奖励
        self.episode_reward_ang = np.zeros(len(self))  # 记录每个参数组合下的角速度奖励
        self.episode_lin_vel_raw = np.zeros(len(self))  # 记录每个参数组合下的线性速度
        self.episode_ang_vel_raw = np.zeros(len(self))  # 记录每个参数组合下的角速度
        self.episode_duration = np.zeros(len(self))  # 记录每个参数组合下的仿真时长

    def get_local_bins(self, bin_inds, ranges=0.1):
        """
        获取指定索引周围的局部单元格索引。

        此方法根据给定的索引和范围，返回参数空间中相邻的单元格索引。这可以用于获取某个点周围的局部区域内的所有点的索引。

        Attributes:
            bin_inds (np.ndarray): 指定的单元格索引，形状为 (n_inds,)，其中n_inds是索引的数量。
            ranges (float or np.ndarray, optional): 局部区域的范围。如果为浮点数，则对所有维度使用相同的范围。
                                                    如果为数组，则指定每个维度的范围。默认为0.1。

        Methods:
            np.ones: 创建一个元素全为1的数组。
            np.logical_and: 用于计算两个数组中对应元素的逻辑与。
            all: 沿指定轴检查数组中的所有元素是否满足条件。

        Returns:
            np.ndarray: 相邻单元格的布尔索引数组，形状为 (n_grid, n_inds)，其中n_grid是参数空间中的点的总数。

        Examples:
            >>> param_space = ParamSpace(seed=42, param1=(0, 10, 5), param2=(-1, 1, 10))
            >>> local_bins = param_space.get_local_bins(bin_inds=np.array([0, 10]), ranges=0.2)
            # 获取索引为0和10的单元格周围范围为0.2的局部单元格索引

        Note:
            - ranges参数允许灵活定义局部区域的大小，可以是统一的浮点数或为每个维度指定不同的值。
        """
        # 如果ranges是浮点数，则为所有维度创建相同的范围数组
        if isinstance(ranges, float):
            ranges = np.ones(self.grid.shape[0]) * ranges
        # 确保bin_inds是一维数组
        bin_inds = bin_inds.reshape(-1)

        # 计算相邻单元格的索引
        adjacent_inds = np.logical_and(
            self.grid[:, None, :].repeat(bin_inds.shape[0], axis=1) >= self.grid[:, bin_inds, None] - ranges.reshape(-1,
                                                                                                                     1,
                                                                                                                     1),
            self.grid[:, None, :].repeat(bin_inds.shape[0], axis=1) <= self.grid[:, bin_inds, None] + ranges.reshape(-1,
                                                                                                                     1,
                                                                                                                     1)
        ).all(axis=0)

        # 返回相邻单元格的布尔索引数组
        return adjacent_inds

    def update(self, bin_inds, task_rewards, success_thresholds, local_range=0.5):
        """
        根据任务奖励和成功阈值更新参数空间的权重。

        此方法根据每个任务的奖励和预设的成功阈值，更新指定索引处的权重，并将成功的任务周围的局部区域权重也进行更新。

        Attributes:
            bin_inds (np.ndarray): 需要更新权重的单元格索引。
            task_rewards (np.ndarray): 对应于bin_inds的任务奖励。
            success_thresholds (np.ndarray): 成功的阈值，与任务奖励一一对应。
            local_range (float, optional): 局部区域的范围，用于更新周围单元格的权重。默认为0.5。

        Methods:
            np.clip: 将数组中的元素限制在给定的范围内。
            get_local_bins: 获取指定索引周围的局部单元格索引。

        Returns:
            None: 此方法不返回任何值，但会更新实例的weights属性。

        Examples:
            >>> param_space = ParamSpace(seed=42, param1=(0, 10, 5), param2=(-1, 1, 10))
            >>> param_space.update(bin_inds=np.array([0, 10]), task_rewards=np.array([0.5, 0.8]), success_thresholds=np.array([0.6, 0.7]))
            # 更新索引为0和10的单元格权重，以及它们周围的局部区域权重

        Note:
            - 如果任务奖励大于对应的成功阈值，则认为任务成功。
            - 成功的任务会使其对应的单元格权重增加，同时也会更新其周围局部区域的权重。
        """
        # 初始化成功标志
        is_success = 1.
        # 遍历每个任务奖励和成功阈值，计算成功标志
        for task_reward, success_threshold in zip(task_rewards, success_thresholds):
            is_success = is_success * (task_reward > success_threshold).cpu()
        # 如果没有设置成功阈值，则所有任务都视为不成功
        if len(success_thresholds) == 0:
            is_success = np.array([False] * len(bin_inds))
        else:
            is_success = np.array(is_success.bool())

        # 更新成功任务的权重
        self.weights[bin_inds[is_success]] = np.clip(self.weights[bin_inds[is_success]] + 0.2, 0, 1)
        # 获取成功任务周围的局部单元格索引
        adjacents = self.get_local_bins(bin_inds[is_success], ranges=local_range)
        # 更新局部区域的权重
        for adjacent in adjacents:
            adjacent_inds = np.array(adjacent.nonzero()[0])
            self.weights[adjacent_inds] = np.clip(self.weights[adjacent_inds] + 0.2, 0, 1)

    def log(self, bin_inds, lin_vel_raw=None, ang_vel_raw=None, episode_duration=None):
        """
        记录指定索引处的线性速度、角速度和仿真时长。

        此方法用于记录仿真过程中的代理性能指标，包括线性速度、角速度和仿真时长。

        Attributes:
            bin_inds (np.ndarray): 需要记录数据的单元格索引。
            lin_vel_raw (torch.Tensor, optional): 线性速度的原始数据。
            ang_vel_raw (torch.Tensor, optional): 角速度的原始数据。
            episode_duration (torch.Tensor, optional): 仿真时长的原始数据。

        Methods:
            cpu: 将数据从GPU转移到CPU。
            numpy: 将torch.Tensor转换为numpy数组。

        Returns:
            None: 此方法不返回任何值，但会更新实例的相关属性。

        Examples:
            >>> param_space = ParamSpace(seed=42, param1=(0, 10, 5), param2=(-1, 1, 10))
            >>> param_space.log(bin_inds=np.array([0, 10]), lin_vel_raw=torch.tensor([0.5, 0.8]), ang_vel_raw=torch.tensor([0.1, 0.2]), episode_duration=torch.tensor([100, 200]))
            # 记录索引为0和10的单元格的线性速度、角速度和仿真时长

        Note:
            - 确保传入的数据类型正确，如果数据在GPU上，需要先转移到CPU。
        """
        # 记录线性速度、角速度和仿真时长
        self.episode_lin_vel_raw[bin_inds] = lin_vel_raw.cpu().numpy()
        self.episode_ang_vel_raw[bin_inds] = ang_vel_raw.cpu().numpy()
        self.episode_duration[bin_inds] = episode_duration.cpu().numpy()


if __name__ == '__main__':
    r = RewardThresholdCurriculum(100, x=(-1, 1, 5), y=(-1, 1, 2), z=(-1, 1, 11))

    assert r._raw_grid.shape == (3, 5, 2, 11), "grid shape is wrong: {}".format(r._raw_grid.shape)

    low, high = np.array([-1.0, -0.6, -1.0]), np.array([1.0, 0.6, 1.0])

    # 假设 set_to 方法已经实现
    # r.set_to(low, high, value=1.0)

    adjacents = r.get_local_bins(np.array([10, ]), ranges=0.5)
    for adjacent in adjacents:
        adjacent_inds = np.array(adjacent.nonzero()[0])
        print(adjacent_inds)
        # 假设 update 方法接受 lin_vel_rewards, ang_vel_rewards, lin_vel_threshold, ang_vel_threshold 参数
        r.update(bin_inds=adjacent_inds, task_rewards=np.ones_like(adjacent_inds),
                 success_thresholds=np.zeros_like(adjacent_inds), local_range=0.5)

    samples, bins = r.sample(10_000)

    plt.scatter(*samples.T[:2])
    plt.show()
