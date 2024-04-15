# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import math

import numpy as np
from isaacgym import terrain_utils
from numpy.random import choice

from go1_gym.envs.base.legged_robot_config import Cfg


class Terrain:
    """
    初始化并配置地形对象，用于仿真环境中。

    此类负责根据配置初始化地形，包括地形的类型、尺寸和特定参数。它支持不同类型的地形，如平面或自定义三角网格地形（trimesh）。
    根据地形类型，它可能会生成高度场或三角网格数据。对于训练和评估环境，可以有不同的配置。

    Attributes:
        cfg (Cfg.terrain): 地形的主要配置参数。
        eval_cfg (Optional[Cfg.terrain]): 评估环境的地形配置参数，如果不同于训练环境。
        num_robots (int): 仿真环境中机器人的数量。
        num_eval_robots (int): 评估环境中机器人的数量。

    Methods:
        load_cfgs: 根据配置加载训练和评估环境的地形尺寸。
        initialize_terrains: 初始化地形数据，根据地形类型生成高度场或三角网格。

    Examples:
        >>> cfg = Cfg.terrain()
        >>> terrain = Terrain(cfg, num_robots=1)
        >>> print(terrain.type)
        trimesh
        >>> print(terrain.vertices.shape)
        (100, 100, 3)
        >>> print(terrain.triangles.shape)
        (198, 3)
        >>> print(terrain.heightsamples.shape)
        (200, 200)
        >>> print(terrain.height_field_raw.shape)
        (200, 200)
        >>> print(terrain.tot_rows)
        200
        >>> print(terrain.tot_cols)
        200
        >>> print(terrain.train_rows)
        [0, 1, 2, ..., 197, 198, 199]
        >>> print(terrain.train_cols)
        [0, 1, 2, ..., 197, 198, 199]
        >>> print(terrain.eval_rows)
        []
        >>> print(terrain.eval_cols)
        []

    Note:
        - 如果地形类型为"none"或"plane"，则不进行进一步的初始化。
        - 对于"trimesh"类型的地形，会使用工具函数将高度场转换为三角网格数据。
    """
    def __init__(self, cfg, num_robots, eval_cfg=None, num_eval_robots=0) -> None:
        self.cfg = cfg  # 地形配置
        self.eval_cfg = eval_cfg  # 评估环境的地形配置
        self.num_robots = num_robots  # 仿真环境中的机器人数量
        self.type = cfg.mesh_type  # 地形类型
        # 如果地形类型为"none"或"plane"，则不进行初始化
        if self.type in ["none", 'plane']:
            return
        # 加载训练和评估环境的地形尺寸配置
        self.train_rows, self.train_cols, self.eval_rows, self.eval_cols = self.load_cfgs()
        # 计算总行数和列数
        self.tot_rows = len(self.train_rows) + len(self.eval_rows)
        self.tot_cols = max(len(self.train_cols), len(self.eval_cols))
        # 设置环境的长度和宽度
        self.cfg.env_length = cfg.terrain_length
        self.cfg.env_width = cfg.terrain_width
        # 初始化高度场数组
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        # 初始化地形数据
        self.initialize_terrains()
        # 保存高度样本
        self.heightsamples = self.height_field_raw
        # 如果地形类型为"trimesh"，将高度场转换为三角网格
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw, self.cfg.horizontal_scale, self.cfg.vertical_scale, self.cfg.slope_treshold)

    def load_cfgs(self):
        """
        加载配置并更新行列索引及偏移量。

        此方法主要用于加载和更新训练和评估配置文件中的参数。它首先加载训练配置，然后根据是否存在评估配置，决定是否加载评估配置。
        加载配置后，此方法会更新配置中的行列索引和偏移量，这对于后续数据处理非常关键。

        Attributes:
            cfg (Config): 训练配置对象，包含训练过程中的各种参数。
            eval_cfg (Config or None): 评估配置对象，如果不进行评估，则为None。

        Returns:
            tuple: 包含四个元素的元组，分别是训练配置的行索引、列索引和评估配置的行索引、列索引。
                   如果没有评估配置，评估配置的行索引和列索引将返回空列表。

        Examples:
            >>> loader = ConfigLoader()
            >>> train_row_indices, train_col_indices, eval_row_indices, eval_col_indices = loader.load_cfgs()
            >>> print(train_row_indices, train_col_indices)  # 打印训练配置的行列索引
            >>> if eval_row_indices:  # 如果存在评估配置的行索引
            ...     print(eval_row_indices, eval_col_indices)  # 打印评估配置的行列索引

        Note:
            - 此方法假设cfg和eval_cfg已经是加载状态，即它们是Config类的实例。
            - 更新的行列索引和偏移量对于数据处理和模型训练至关重要。
        """
        self._load_cfg(self.cfg)  # 加载训练配置
        self.cfg.row_indices = np.arange(0, self.cfg.tot_rows)  # 更新训练配置的行索引
        self.cfg.col_indices = np.arange(0, self.cfg.tot_cols)  # 更新训练配置的列索引
        self.cfg.x_offset = 0  # 设置训练配置的x偏移量为0
        self.cfg.rows_offset = 0  # 设置训练配置的行偏移量为0
        if self.eval_cfg is None:  # 如果没有评估配置
            return self.cfg.row_indices, self.cfg.col_indices, [], []  # 返回训练配置的行列索引，评估配置的行列索引为空列表
        else:
            self._load_cfg(self.eval_cfg)  # 加载评估配置
            self.eval_cfg.row_indices = np.arange(self.cfg.tot_rows,
                                                  self.cfg.tot_rows + self.eval_cfg.tot_rows)  # 更新评估配置的行索引
            self.eval_cfg.col_indices = np.arange(0, self.eval_cfg.tot_cols)  # 更新评估配置的列索引
            self.eval_cfg.x_offset = self.cfg.tot_rows  # 设置评估配置的x偏移量
            self.eval_cfg.rows_offset = self.cfg.num_rows  # 设置评估配置的行偏移量
            return self.cfg.row_indices, self.cfg.col_indices, self.eval_cfg.row_indices, self.eval_cfg.col_indices  # 返回训练和评估配置的行列索引

    def _load_cfg(self, cfg):
        """
        加载并更新配置对象中的地形、环境尺寸和边界参数。

        此方法负责计算地形比例累积、子地形数量、每个环境的原点位置、每个环境的像素尺寸以及总行列数和边界大小。
        这些计算对于后续环境的初始化和渲染至关重要。

        Attributes:
            cfg (Config): 配置对象，包含地形比例、环境尺寸、边界大小等参数。

        Note:
            - `terrain_proportions` 应为地形类型的比例列表，此方法将其转换为累积比例，以便于后续操作。
            - `num_sub_terrains` 为根据行列数计算得到的子地形总数。
            - `env_origins` 存储每个子环境的原点位置，初始化为全零矩阵。
            - `width_per_env_pixels` 和 `length_per_env_pixels` 分别为每个环境的宽度和长度，以像素为单位。
            - `border` 为根据配置中的边界大小和水平缩放比例计算得到的边界宽度，以像素为单位。
            - `tot_cols` 和 `tot_rows` 分别为总列数和总行数，包括边界。
        """
        # 计算地形比例的累积值，用于后续地形选择
        cfg.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]
        # 计算子地形的总数，即行数乘以列数
        cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        # 初始化每个环境的原点位置矩阵为零
        cfg.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        # 根据水平缩放比例计算每个环境的宽度，单位为像素
        cfg.width_per_env_pixels = int(cfg.terrain_length / cfg.horizontal_scale)
        # 根据水平缩放比例计算每个环境的长度，单位为像素
        cfg.length_per_env_pixels = int(cfg.terrain_width / cfg.horizontal_scale)
        # 根据水平缩放比例计算边界宽度，单位为像素
        cfg.border = int(cfg.border_size / cfg.horizontal_scale)
        # 计算总列数，包括边界
        cfg.tot_cols = int(cfg.num_cols * cfg.width_per_env_pixels) + 2 * cfg.border
        # 计算总行数，包括边界
        cfg.tot_rows = int(cfg.num_rows * cfg.length_per_env_pixels) + 2 * cfg.border

    def initialize_terrains(self):
        """
        初始化训练和评估配置中的地形。

        此方法负责初始化训练配置（cfg）和评估配置（eval_cfg）中的地形。它首先初始化训练配置中的地形，
        如果存在评估配置，随后也会初始化评估配置中的地形。

        Attributes:
            cfg (Config): 训练配置对象，包含训练过程中的各种参数。
            eval_cfg (Config or None): 评估配置对象，如果不进行评估，则为None。

        Examples:
            >>> terrain_initializer = TerrainInitializer()
            >>> terrain_initializer.initialize_terrains()
            # 此时，cfg 和 eval_cfg（如果不为None）中的地形已经被初始化。

        Note:
            - 此方法假设cfg和eval_cfg已经是加载状态，即它们是Config类的实例。
            - 初始化地形是准备仿真环境的必要步骤之一。
        """
        self._initialize_terrain(self.cfg)  # 初始化训练配置中的地形
        if self.eval_cfg is not None:  # 如果存在评估配置
            self._initialize_terrain(self.eval_cfg)  # 初始化评估配置中的地形

    def _initialize_terrain(self, cfg):
        """
        根据配置初始化特定的地形。

        此方法根据配置对象中的指示选择合适的地形初始化方法。它支持三种地形初始化模式：
        - 课程学习模式：根据学习进度逐步引入复杂的地形。
        - 选定地形模式：始终使用配置中指定的地形。
        - 随机地形模式：每次初始化时随机选择地形。

        Attributes:
            cfg (Config): 配置对象，包含地形初始化相关的参数。

        Note:
            - `curriculum` 指示是否使用课程学习模式。
            - `selected` 指示是否使用选定地形模式。
            - 如果 `curriculum` 和 `selected` 均为False，则默认使用随机地形模式。
        """
        if cfg.curriculum:  # 如果配置中启用了课程学习模式
            self.curriculum(cfg)  # 调用课程学习模式的地形初始化方法
        elif cfg.selected:  # 如果配置中指定了选定地形模式
            self.selected_terrain(cfg)  # 调用选定地形模式的地形初始化方法
        else:  # 如果上述两种模式均未启用
            self.randomized_terrain(cfg)  # 调用随机地形模式的地形初始化方法

    def randomized_terrain(self, cfg):
        """
        为配置中指定的每个子地形随机生成地形。

        此方法遍历配置中指定的所有子地形，为每个子地形随机选择地形类型和难度等级。然后，根据这些选择生成地形，并将其添加到地图中。
        这种随机化方法允许每次实验或训练时环境具有多样性，从而增加了任务的挑战性和泛化能力。

        Attributes:
            cfg (Config): 配置对象，包含地形比例、子地形数量等参数。

        Methods:
            make_terrain(cfg, choice, difficulty, proportions): 根据随机选择和难度生成地形。
            add_terrain_to_map(cfg, terrain, i, j): 将生成的地形添加到地图的指定位置。

        Examples:
            >>> terrain_manager = TerrainManager()
            >>> cfg = Config()  # 假设Config是一个配置类
            >>> terrain_manager.randomized_terrain(cfg)
            # 此时，cfg指定的所有子地形将被随机生成的地形填充。

        Note:
            - 此方法假设`cfg`已经是加载状态，即它是Config类的实例。
            - `num_sub_terrains`、`num_rows`、`num_cols`和`proportions`是此方法中使用的重要配置参数。
        """
        for k in range(cfg.num_sub_terrains):  # 遍历所有子地形
            # 计算子地形在世界坐标中的位置
            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))
            # 随机选择地形类型
            choice = np.random.uniform(0, 1)
            # 随机选择地形难度
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            # 根据选择和难度生成地形
            terrain = self.make_terrain(cfg, choice, difficulty, cfg.proportions)
            # 将生成的地形添加到地图的指定位置
            self.add_terrain_to_map(cfg, terrain, i, j)

    def curriculum(self, cfg):
        """
        根据课程学习策略为每个子地形生成地形。

        此方法按照配置中的课程学习策略，逐行逐列地为每个子地形生成地形。难度和地形类型的选择基于子地形在地图中的位置，
        以实现难度逐渐增加和地形多样化。这种方法旨在通过逐步增加任务难度来促进学习过程。

        Attributes:
            cfg (Config): 配置对象，包含地形比例、子地形数量、难度缩放等参数。

        Methods:
            make_terrain(cfg, choice, difficulty, proportions): 根据位置计算的选择和难度生成地形。
            add_terrain_to_map(cfg, terrain, i, j): 将生成的地形添加到地图中的指定位置。

        Note:
            - 难度是根据子地形的行位置计算的，以实现从上到下难度逐渐增加。
            - 地形类型的选择是根据子地形的列位置计算的，以实现从左到右地形的多样化。
        """
        for j in range(cfg.num_cols):  # 遍历所有列
            for i in range(cfg.num_rows):  # 遍历所有行
                # 计算难度，基于子地形的行位置
                difficulty = i / cfg.num_rows * cfg.difficulty_scale
                # 计算地形类型选择，基于子地形的列位置
                choice = j / cfg.num_cols + 0.001

                # 根据计算的选择和难度生成地形
                terrain = self.make_terrain(cfg, choice, difficulty, cfg.proportions)
                # 将生成的地形添加到地图中的指定位置
                self.add_terrain_to_map(cfg, terrain, i, j)

    def selected_terrain(self, cfg):
        """
        为所有子地形生成指定类型的地形。

        此方法根据配置中指定的地形类型为每个子地形生成相同类型的地形。它首先从配置中提取地形类型和其他相关参数，
        然后为配置中指定的每个子地形创建并初始化指定类型的地形对象。这种方法适用于需要在整个环境中统一使用特定地形类型的场景。

        Attributes:
            cfg (Config): 配置对象，包含地形类型、子地形数量、环境尺寸等参数。

        Methods:
            add_terrain_to_map(cfg, terrain, i, j): 将生成的地形添加到地图中的指定位置。

        Note:
            - `terrain_kwargs` 字典中应包含地形类型以及初始化地形所需的其他参数。
            - 此方法假设`terrain_utils.SubTerrain`类和相应的地形类型构造函数已经定义并可用。
        """
        # 从配置中提取地形类型
        terrain_type = cfg.terrain_kwargs.pop('type')
        for k in range(cfg.num_sub_terrains):  # 遍历所有子地形
            # 计算子地形在世界坐标中的位置
            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))

            # 创建指定类型的地形对象
            terrain = terrain_utils.SubTerrain("terrain",
                                               width=cfg.width_per_env_pixels,
                                               length=cfg.width_per_env_pixels,
                                               vertical_scale=cfg.vertical_scale,
                                               horizontal_scale=cfg.horizontal_scale)

            # 使用提取的地形类型和其他参数初始化地形
            eval(terrain_type)(terrain, **cfg.terrain_kwargs)
            # 将生成的地形添加到地图中的指定位置
            self.add_terrain_to_map(cfg, terrain, i, j)

    def make_terrain(self, cfg, choice, difficulty, proportions):
        """
        根据配置和难度生成地形。

        根据给定的难度和选择参数，生成不同类型的地形。地形的生成依赖于难度系数、选择的地形类型以及地形的比例配置。

        Attributes:
            cfg: 地形配置，包含地形的宽度、长度、垂直和水平比例等参数。
            choice: 用于选择地形类型的参数。
            difficulty: 地形难度，影响地形的斜率、障碍物高度等特征。
            proportions: 地形类型的比例数组，用于决定生成哪种类型的地形。

        Returns:
            terrain: 生成的地形对象。

        Examples:
            >>> terrain = make_terrain(cfg, 2, 0.5, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            >>> print(terrain)

        Note:
            - 该函数支持生成多种类型的地形，包括平坦地形、随机地形、台阶地形等。
            - 地形的具体类型和特征由choice参数和difficulty参数共同决定。
        """
        # 初始化子地形对象，设置宽度、长度、垂直和水平比例
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=cfg.width_per_env_pixels,
                                           length=cfg.width_per_env_pixels,
                                           vertical_scale=cfg.vertical_scale,
                                           horizontal_scale=cfg.horizontal_scale)
        # 根据难度计算斜率
        slope = difficulty * 0.4
        # 计算台阶高度
        step_height = 0.05 + 0.18 * difficulty
        # 计算离散障碍物的高度
        discrete_obstacles_height = 0.05 + difficulty * (cfg.max_platform_height - 0.05)
        # 计算踏石的大小
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        # 根据难度计算踏石间距
        stone_distance = 0.05 if difficulty == 0 else 0.1

        # 根据choice值和proportions数组生成不同类型的地形
        if choice < proportions[0]:
            if choice < proportions[0] / 2:
                # 如果choice小于一半，斜率取负
                slope *= -1
            # 生成金字塔斜坡地形
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < proportions[1]:
            # 生成金字塔斜坡地形
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            # 生成随机均匀地形
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05,
                                                 step=self.cfg.terrain_smoothness, downsampled_scale=0.2)
        elif choice < proportions[3]:
            if choice < proportions[2]:
                # 如果choice小于proportions[2]，台阶高度取负
                step_height *= -1
            # 生成金字塔台阶地形
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < proportions[4]:
            # 设置矩形障碍物的数量和大小范围
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            # 生成离散障碍物地形
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                     rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < proportions[5]:
            # 生成踏石地形
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size,
                                                  stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < proportions[6]:
            # 此区间不生成特定地形
            pass
        elif choice < proportions[7]:
            # 此区间不生成特定地形
            pass
        elif choice < proportions[8]:
            # 生成随机均匀地形，噪声幅度由配置决定
            terrain_utils.random_uniform_terrain(terrain, min_height=-cfg.terrain_noise_magnitude,
                                                 max_height=cfg.terrain_noise_magnitude, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < proportions[9]:
            # 生成随机均匀地形，高度范围固定
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05,
                                                 step=self.cfg.terrain_smoothness, downsampled_scale=0.2)
            # 将地形的一半高度设置为0
            terrain.height_field_raw[0:terrain.length // 2, :] = 0

        return terrain

    def add_terrain_to_map(self, cfg, terrain, row, col):
        """
        将地形添加到仿真地图中。

        此方法根据配置和指定的行列位置，将给定的地形添加到仿真地图的高度场中。它首先计算地形在地图中的具体位置坐标，
        然后更新地图的高度数据。此外，该方法还计算并更新了环境原点的位置和高度。

        Attributes:
            cfg (Config): 仿真环境的配置参数，包含地图边界、每个环境的尺寸、偏移量等。
            terrain (Terrain): 要添加到地图中的地形对象，包含地形的高度数据。
            row (int): 地形所在的行位置。
            col (int): 地形所在的列位置。

        Methods:
            add_terrain_to_map: 根据给定的行列位置，将地形添加到仿真地图中。

        Returns:
            None

        Examples:
            >>> cfg = Config()  # 初始化配置
            >>> terrain = Terrain()  # 创建地形对象
            >>> simulator = Simulator()  # 创建仿真器实例
            >>> simulator.add_terrain_to_map(cfg, terrain, 0, 0)  # 将地形添加到地图的(0,0)位置

        Note:
            - 确保传入的行列位置不超出地图的范围。
            - 地形的高度数据将直接影响仿真环境的地形高度。
        """
        i = row  # 地形所在的行位置
        j = col  # 地形所在的列位置
        # 计算地形在地图中的具体位置坐标
        start_x = cfg.border + i * cfg.length_per_env_pixels + cfg.x_offset
        end_x = cfg.border + (i + 1) * cfg.length_per_env_pixels + cfg.x_offset
        start_y = cfg.border + j * cfg.width_per_env_pixels
        end_y = cfg.border + (j + 1) * cfg.width_per_env_pixels
        # 更新地图的高度数据
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # 计算环境原点的位置
        env_origin_x = (i + 0.5) * cfg.terrain_length + cfg.x_offset * terrain.horizontal_scale
        env_origin_y = (j + 0.5) * cfg.terrain_width
        # 计算环境原点的高度
        env_origin_z = np.max(self.height_field_raw[start_x: end_x, start_y:end_y]) * terrain.vertical_scale
        # 更新环境原点的位置
        cfg.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
