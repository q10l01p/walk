# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

from params_proto import PrefixProto, ParamsProto


class Cfg(PrefixProto, cli=False):

    class env(PrefixProto, cli=False):
        """
        Cfg类，用于定义和存储环境配置参数。

        该配置类包含了环境的基本设置，如环境数量、观察值和动作的维度、是否记录视频等。此外，还包括了一些高级特性的配置，
        如是否观察特权信息（用于不对称训练）、是否观察接触状态等。

        Attributes:
            env: 环境相关的配置参数。

        Examples:
            >>> cfg = Cfg()
            >>> print(cfg.env.num_envs)
            4096
            创建一个配置实例，并打印环境数量。

        Note:
            - 该配置类是仿真环境初始化时必须提供的参数之一。
            - 配置参数可以根据需要进行调整。
        """
        num_envs = 4096  # 环境数量
        num_observations = 235  # 观察值的总维度
        num_scalar_observations = 42  # 标量观察值的维度
        num_privileged_obs = 18  # 特权观察值的维度，用于不对称训练
        privileged_future_horizon = 1  # 特权观察的未来视野
        num_actions = 12  # 动作的维度
        num_observation_history = 15  # 观察历史的长度
        env_spacing = 3.  # 环境间的间隔，不适用于高度场/三角网格
        send_timeouts = True  # 是否向算法发送超时信息
        episode_length_s = 20  # 单个episode的长度，以秒为单位
        observe_vel = True  # 是否观察速度
        observe_only_ang_vel = False  # 是否仅观察角速度
        observe_only_lin_vel = False  # 是否仅观察线速度
        observe_yaw = False  # 是否观察偏航角
        observe_contact_states = False  # 是否观察接触状态
        observe_command = True  # 是否观察命令
        observe_height_command = False  # 是否观察高度命令
        observe_gait_commands = False  # 是否观察步态命令
        observe_timing_parameter = False  # 是否观察时间参数
        observe_clock_inputs = False  # 是否观察时钟输入
        observe_two_prev_actions = False  # 是否观察前两个动作
        observe_imu = False  # 是否观察惯性测量单元(IMU)
        record_video = True  # 是否记录视频
        recording_width_px = 360  # 视频宽度，以像素为单位
        recording_height_px = 240  # 视频高度，以像素为单位
        recording_mode = "COLOR"  # 视频模式
        num_recording_envs = 1  # 记录视频的环境数量
        debug_viz = False  # 是否开启调试可视化
        all_agents_share = False  # 是否所有智能体共享相同的环境

        # 特权观察相关的配置参数
        priv_observe_friction = True  # 是否观察摩擦力
        priv_observe_friction_indep = True  # 是否独立观察摩擦力
        priv_observe_ground_friction = False  # 是否观察地面摩擦力
        priv_observe_ground_friction_per_foot = False  # 是否按脚观察地面摩擦力
        priv_observe_restitution = True  # 是否观察恢复系数
        priv_observe_base_mass = True  # 是否观察基座质量
        priv_observe_com_displacement = True  # 是否观察质心位移
        priv_observe_motor_strength = False  # 是否观察电机强度
        priv_observe_motor_offset = False  # 是否观察电机偏移
        priv_observe_joint_friction = True  # 是否观察关节摩擦力
        priv_observe_Kp_factor = True  # 是否观察比例增益因子
        priv_observe_Kd_factor = True  # 是否观察微分增益因子
        priv_observe_contact_forces = False  # 是否观察接触力
        priv_observe_contact_states = False  # 是否观察接触状态
        priv_observe_body_velocity = False  # 是否观察身体速度
        priv_observe_foot_height = False  # 是否观察脚高
        priv_observe_body_height = False  # 是否观察身体高度
        priv_observe_gravity = False  # 是否观察重力
        priv_observe_terrain_type = False  # 是否观察地形类型
        priv_observe_clock_inputs = False  # 是否观察时钟输入
        priv_observe_doubletime_clock_inputs = False  # 是否观察双倍时间的时钟输入
        priv_observe_halftime_clock_inputs = False  # 是否观察半倍时间的时钟输入
        priv_observe_desired_contact_states = False  # 是否观察期望的接触状态
        priv_observe_dummy_variable = False  # 是否观察虚拟变量

    class terrain(PrefixProto, cli=False):
        """
        terrain子类，用于定义地形相关的配置参数。

        该配置类包含了地形的类型、尺度、摩擦系数、恢复系数等参数，以及地形噪声、地形平滑度等高级特性的配置。
        它还支持地形的课程学习设置，允许动态调整地形难度。

        Attributes:

        Examples:
            >>> cfg = Cfg()
            >>> print(cfg.terrain.mesh_type)
            'trimesh'
            创建一个配置实例，并打印地形类型。

        Note:
            - 地形类型的选择对仿真环境的物理特性和视觉效果有重要影响。
            - 课程学习设置允许在训练过程中逐步增加地形的难度，以提高学习效率和适应性。
        """
        mesh_type = 'trimesh'  # 地形类型：三角网格
        horizontal_scale = 0.1  # 水平尺度[m]
        vertical_scale = 0.005  # 垂直尺度[m]
        border_size = 0  # 边界大小[m]
        curriculum = True  # 启用课程学习
        static_friction = 1.0  # 静摩擦系数
        dynamic_friction = 1.0  # 动摩擦系数
        restitution = 0.0  # 恢复系数
        terrain_noise_magnitude = 0.1  # 地形噪声幅度
        terrain_smoothness = 0.005  # 地形平滑度，适用于粗糙地形
        measure_heights = True  # 测量地形高度
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8]  # 测量点x坐标
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]  # 测量点y坐标
        selected = False  # 选择唯一的地形类型
        terrain_kwargs = None  # 选定地形的参数字典
        min_init_terrain_level = 0  # 课程学习的最小地形等级
        max_init_terrain_level = 5  # 课程学习的最大地形等级
        terrain_length = 8.  # 地形长度
        terrain_width = 8.  # 地形宽度
        num_rows = 10  # 地形行数（等级数）
        num_cols = 20  # 地形列数（类型数）
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]  # 地形类型比例
        slope_treshold = 0.75  # 斜坡阈值
        difficulty_scale = 1.  # 难度尺度
        x_init_range = 1.  # x方向初始化范围
        y_init_range = 1.  # y方向初始化范围
        yaw_init_range = 0.  # 偏航角初始化范围
        x_init_offset = 0.  # x方向初始化偏移
        y_init_offset = 0.  # y方向初始化偏移
        teleport_robots = True  # 传送机器人
        teleport_thresh = 2.0  # 传送阈值
        max_platform_height = 0.2  # 最大平台高度
        center_robots = False  # 将机器人置于中心
        center_span = 5  # 中心跨度

    class commands(PrefixProto, cli=False):
        """
        commands子类，用于定义命令和动作相关的配置参数。

        该配置类包含了命令的课程学习设置、命令的类型和数量、命令变化的时间间隔等参数。此外，还包括了跳跃动作的配置、
        身体高度命令的范围、速度限制等高级特性的配置。它还支持命令的分布式采样和课程学习的高级设置。

        Attributes:

        Note:
            - 命令和动作的配置对于控制策略的性能有重要影响。
            - 课程学习和分布式命令采样可以提高学习效率和策略的适应性。
        """
        command_curriculum = False  # 是否启用命令的课程学习
        max_reverse_curriculum = 1.  # 最大反向课程学习值
        max_forward_curriculum = 1.  # 最大前向课程学习值
        yaw_command_curriculum = False  # 是否启用偏航命令的课程学习
        max_yaw_curriculum = 1.  # 最大偏航课程学习值
        exclusive_command_sampling = False  # 是否启用独占命令采样
        num_commands = 3  # 命令的数量
        resampling_time = 10.  # 命令变化的时间间隔[s]
        subsample_gait = False  # 是否对步态进行子采样
        gait_interval_s = 10.  # 步态参数重采样的时间间隔
        vel_interval_s = 10.  # 速度命令重采样的时间间隔
        jump_interval_s = 20.  # 跳跃命令重采样的时间间隔
        jump_duration_s = 0.1  # 跳跃动作的持续时间
        jump_height = 0.3  # 跳跃高度
        heading_command = True  # 是否根据偏航误差计算角速度命令
        global_reference = False  # 是否使用全局参考
        observe_accel = False  # 是否观察加速度
        distributional_commands = False  # 是否使用分布式命令
        curriculum_type = "RewardThresholdCurriculum"  # 课程学习的类型
        lipschitz_threshold = 0.9  # Lipschitz阈值
        num_lin_vel_bins = 20  # 线速度的分箱数量
        lin_vel_step = 0.3  # 线速度的步长
        num_ang_vel_bins = 20  # 角速度的分箱数量
        ang_vel_step = 0.3  # 角速度的步长
        distribution_update_extension_distance = 1  # 分布更新的扩展距离
        curriculum_seed = 100  # 课程学习的种子
        lin_vel_x = [-1.0, 1.0]  # 线速度x的范围[min, max] [m/s]
        lin_vel_y = [-1.0, 1.0]  # 线速度y的范围[min, max] [m/s]
        ang_vel_yaw = [-1, 1]  # 角速度偏航的范围[min, max] [rad/s]
        body_height_cmd = [-0.05, 0.05]  # 身体高度命令的范围[min, max]
        impulse_height_commands = False  # 是否使用冲击高度命令
        limit_vel_x = [-10.0, 10.0]  # x方向速度的限制[min, max]
        limit_vel_y = [-0.6, 0.6]  # y方向速度的限制[min, max]
        limit_vel_yaw = [-10.0, 10.0]  # 偏航速度的限制[min, max]
        limit_body_height = [-0.05, 0.05]  # 身体高度的限制[min, max]
        limit_gait_phase = [0, 0.01]  # 步态相位的限制[min, max]
        limit_gait_offset = [0, 0.01]  # 步态偏移的限制[min, max]
        limit_gait_bound = [0, 0.01]  # 步态边界的限制[min, max]
        limit_gait_frequency = [2.0, 2.01]  # 步态频率的限制[min, max]
        limit_gait_duration = [0.49, 0.5]  # 步态持续时间的限制[min, max]
        limit_footswing_height = [0.06, 0.061]  # 脚摆高度的限制[min, max]
        limit_body_pitch = [0.0, 0.01]  # 身体俯仰的限制[min, max]
        limit_body_roll = [0.0, 0.01]  # 身体翻滚的限制[min, max]
        limit_aux_reward_coef = [0.0, 0.01]  # 辅助奖励系数的限制[min, max]
        limit_compliance = [0.0, 0.01]  # 合规性的限制[min, max]
        limit_stance_width = [0.0, 0.01]  # 站姿宽度的限制[min, max]
        limit_stance_length = [0.0, 0.01]  # 站姿长度的限制[min, max]
        num_bins_vel_x = 25  # x方向速度的分箱数量
        num_bins_vel_y = 3  # y方向速度的分箱数量
        num_bins_vel_yaw = 25  # 偏航速度的分箱数量
        num_bins_body_height = 1  # 身体高度的分箱数量
        num_bins_gait_frequency = 11  # 步态频率的分箱数量
        num_bins_gait_phase = 11  # 步态相位的分箱数量
        num_bins_gait_offset = 2  # 步态偏移的分箱数量
        num_bins_gait_bound = 2  # 步态边界的分箱数量
        num_bins_gait_duration = 3  # 步态持续时间的分箱数量
        num_bins_footswing_height = 1  # 脚摆高度的分箱数量
        num_bins_body_pitch = 1  # 身体俯仰的分箱数量
        num_bins_body_roll = 1  # 身体翻滚的分箱数量
        num_bins_aux_reward_coef = 1  # 辅助奖励系数的分箱数量
        num_bins_compliance = 1  # 合规性的分箱数量
        num_bins_stance_width = 1  # 站姿宽度的分箱数量
        num_bins_stance_length = 1  # 站姿长度的分箱数量
        heading = [-3.14, 3.14]  # 偏航的范围[min, max]
        gait_phase_cmd_range = [0.0, 0.01]  # 步态相位命令的范围[min, max]
        gait_offset_cmd_range = [0.0, 0.01]  # 步态偏移命令的范围[min, max]
        gait_bound_cmd_range = [0.0, 0.01]  # 步态边界命令的范围[min, max]
        gait_frequency_cmd_range = [2.0, 2.01]  # 步态频率命令的范围[min, max]
        gait_duration_cmd_range = [0.49, 0.5]  # 步态持续时间命令的范围[min, max]
        footswing_height_range = [0.06, 0.061]  # 脚摆高度命令的范围[min, max]
        body_pitch_range = [0.0, 0.01]  # 身体俯仰命令的范围[min, max]
        body_roll_range = [0.0, 0.01]  # 身体翻滚命令的范围[min, max]
        aux_reward_coef_range = [0.0, 0.01]  # 辅助奖励系数命令的范围[min, max]
        compliance_range = [0.0, 0.01]  # 合规性命令的范围[min, max]
        stance_width_range = [0.0, 0.01]  # 站姿宽度命令的范围[min, max]
        stance_length_range = [0.0, 0.01]  # 站姿长度命令的范围[min, max]
        exclusive_phase_offset = True  # 是否使用独占相位偏移
        binary_phases = False  # 是否使用二进制相位
        pacing_offset = False  # 是否使用步态偏移
        balance_gait_distribution = True  # 是否平衡步态分布
        gaitwise_curricula = True  # 是否使用步态课程学习

    class curriculum_thresholds(PrefixProto, cli=False):
        """
        定义仿真环境中课程学习阶段的性能阈值参数。

        该类继承自PrefixProto，可用于命令行接口（CLI）设置中，以调整仿真环境中课程学习的性能阈值。

        Attributes:
            tracking_lin_vel (float): 线速度跟踪的性能阈值，接近1表示更严格的要求。
            tracking_ang_vel (float): 角速度跟踪的性能阈值，接近1表示更严格的要求。
            tracking_contacts_shaped_force (float): 形状化接触力跟踪的性能阈值，接近1表示更严格的要求。
            tracking_contacts_shaped_vel (float): 形状化接触速度跟踪的性能阈值，接近1表示更严格的要求。
        """
        # 线速度跟踪的性能阈值，接近1表示更严格的要求
        tracking_lin_vel = 0.8
        # 角速度跟踪的性能阈值，接近1表示更严格的要求
        tracking_ang_vel = 0.5
        # 形状化接触力跟踪的性能阈值，接近1表示更严格的要求
        tracking_contacts_shaped_force = 0.8
        # 形状化接触速度跟踪的性能阈值，接近1表示更严格的要求
        tracking_contacts_shaped_vel = 0.8

    class init_state(PrefixProto, cli=False):
        """
        定义仿真环境中代理的初始状态参数。

        该类继承自PrefixProto，可用于命令行接口（CLI）设置中，以调整仿真环境中代理的初始状态。

        Attributes:
            pos (list): 代理的初始位置，格式为[x, y, z]，单位为米。
            rot (list): 代理的初始旋转，格式为[x, y, z, w]，表示四元数。
            lin_vel (list): 代理的初始线速度，格式为[x, y, z]，单位为米/秒。
            ang_vel (list): 代理的初始角速度，格式为[x, y, z]，单位为弧度/秒。
            default_joint_angles (dict): 当动作为0.0时的目标关节角度，键为关节名称，值为角度。
        """
        # 代理的初始位置，单位为米
        pos = [0.0, 0.0, 1.]
        # 代理的初始旋转，表示为四元数
        rot = [0.0, 0.0, 0.0, 1.0]
        # 代理的初始线速度，单位为米/秒
        lin_vel = [0.0, 0.0, 0.0]
        # 代理的初始角速度，单位为弧度/秒
        ang_vel = [0.0, 0.0, 0.0]
        # 当动作为0.0时的目标关节角度
        default_joint_angles = {"joint_a": 0., "joint_b": 0.}

    class control(PrefixProto, cli=False):
        """
        定义和管理仿真环境中控制策略的参数。

        该类继承自PrefixProto，可用于命令行接口（CLI）设置中，以调整仿真环境中控制策略的各个参数。

        Attributes:
            control_type (str): 控制类型，'actuator_net'表示通过执行器网络控制，'P'表示位置控制，'V'表示速度控制，'T'表示扭矩控制。
            stiffness (dict): PD控制器的刚度参数，以[N*m/rad]为单位。
            damping (dict): PD控制器的阻尼参数，以[N*m*s/rad]为单位。
            action_scale (float): 动作缩放比例，目标角度 = action_scale * 动作 + 默认角度。
            hip_scale_reduction (float): 髋关节缩放比例减少量。
            decimation (int): 控制动作更新的减采样率，即仿真时间步长（sim DT）上每个策略时间步长（policy DT）的控制动作更新次数。
        """
        # 控制类型
        control_type = 'actuator_net'  # 'P'表示位置控制，'V'表示速度控制，'T'表示扭矩控制
        # PD控制器的刚度参数
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        # PD控制器的阻尼参数
        damping = {'joint_a': 1.0, 'joint_b': 1.5}  # [N*m*s/rad]
        # 动作缩放比例
        action_scale = 0.5  # 目标角度 = action_scale * 动作 + 默认角度
        # 髋关节缩放比例减少量
        hip_scale_reduction = 1.0
        # 控制动作更新的减采样率
        decimation = 4  # 即仿真时间步长（sim DT）上每个策略时间步长（policy DT）的控制动作更新次数

    class asset(PrefixProto, cli=False):
        """
        定义和管理仿真环境中使用的资产（如机器人模型）的参数。

        该类继承自PrefixProto，可用于命令行接口（CLI）设置中，以调整仿真环境中资产的各个参数。

        Attributes:

        """
        file = ""  # 资产文件的路径
        foot_name = "None"  # 脚部体的名称
        penalize_contacts_on = []  # 需要对接触进行惩罚的部位列表
        terminate_after_contacts_on = []  # 接触这些部位后终止仿真的部位列表
        disable_gravity = False  # 是否禁用重力
        collapse_fixed_joints = True  # 是否合并由固定关节连接的体
        fix_base_link = False  # 是否固定机器人的基座
        default_dof_drive_mode = 3  # 默认的关节驱动模式
        self_collisions = 0  # 自碰撞设置
        replace_cylinder_with_capsule = True  # 是否用胶囊体替换碰撞用的圆柱体
        flip_visual_attachments = True  # 是否翻转视觉附件

        density = 0.001  # 密度
        angular_damping = 0.  # 角阻尼
        linear_damping = 0.  # 线阻尼
        max_angular_velocity = 1000.  # 最大角速度
        max_linear_velocity = 1000.  # 最大线速度
        armature = 0.  # 骨架
        thickness = 0.01  # 厚度

    class domain_rand(PrefixProto, cli=False):
        """
        定义仿真环境中的领域随机化参数。

        该类继承自PrefixProto，可用于命令行接口（CLI）设置中，以调整仿真环境中的领域随机化参数。

        Attributes:

        """
        # 随机化间隔时间，单位为秒
        rand_interval_s = 10
        # 是否在仿真开始后随机化刚体属性
        randomize_rigids_after_start = True
        # 是否随机化摩擦系数
        randomize_friction = True
        # 摩擦系数的随机化范围
        friction_range = [0.5, 1.25]
        # 是否随机化弹性系数
        randomize_restitution = False
        # 弹性系数的随机化范围
        restitution_range = [0, 1.0]
        # 是否随机化基座质量
        randomize_base_mass = False
        # 附加质量的随机化范围
        added_mass_range = [-1., 1.]
        # 是否随机化质心位移
        randomize_com_displacement = False
        # 质心位移的随机化范围
        com_displacement_range = [-0.15, 0.15]
        # 是否随机化电机强度
        randomize_motor_strength = False
        # 电机强度的随机化范围
        motor_strength_range = [0.9, 1.1]
        # 是否随机化比例增益系数（Kp）
        randomize_Kp_factor = False
        # 比例增益系数的随机化范围
        Kp_factor_range = [0.8, 1.3]
        # 是否随机化微分增益系数（Kd）
        randomize_Kd_factor = False
        # 微分增益系数的随机化范围
        Kd_factor_range = [0.5, 1.5]
        # 重力随机化间隔时间，单位为秒
        gravity_rand_interval_s = 7
        # 重力脉冲持续时间
        gravity_impulse_duration = 1.0
        # 是否随机化重力
        randomize_gravity = False
        # 重力的随机化范围
        gravity_range = [-1.0, 1.0]
        # 是否推动机器人
        push_robots = True
        # 推动间隔时间，单位为秒
        push_interval_s = 15
        # 最大推动速度（XY平面）
        max_push_vel_xy = 1.
        # 是否随机化延迟时间步
        randomize_lag_timesteps = True
        # 延迟时间步的数量
        lag_timesteps = 6

    class rewards(PrefixProto, cli=False):
        """
        定义和管理仿真环境中奖励函数的参数。

        该类继承自PrefixProto，可用于命令行接口（CLI）设置中，以调整仿真环境中奖励函数的各个参数。

        Attributes:
        """
        # 如果为True，则将负的总奖励值裁剪为零
        only_positive_rewards = True
        # 采用JI22风格的正奖励裁剪方式
        only_positive_rewards_ji22_style = False
        # 负奖励的标准差
        sigma_rew_neg = 5
        # 奖励容器的名称
        reward_container_name = "CoRLRewards"
        # 跟踪奖励的标准差
        tracking_sigma = 0.25
        # 横向跟踪奖励的标准差
        tracking_sigma_lat = 0.25
        # 纵向跟踪奖励的标准差
        tracking_sigma_long = 0.25
        # 偏航角跟踪奖励的标准差
        tracking_sigma_yaw = 0.25
        # URDF限制的百分比，超过此限制的值将被惩罚
        soft_dof_pos_limit = 1.
        # 软关节速度限制
        soft_dof_vel_limit = 1.
        # 软扭矩限制
        soft_torque_limit = 1.
        # 基座目标高度
        base_height_target = 1.
        # 超过此值的接触力将被惩罚
        max_contact_force = 100.
        # 是否使用终止时的身体高度条件
        use_terminal_body_height = False
        # 终止时的身体高度阈值
        terminal_body_height = 0.20
        # 是否使用终止时的脚高度条件
        use_terminal_foot_height = False
        # 终止时的脚高度阈值
        terminal_foot_height = -0.005
        # 是否使用终止时的滚转和俯仰条件
        use_terminal_roll_pitch = False
        # 终止时的身体方向阈值
        terminal_body_ori = 0.5
        # 步态概率的κ值
        kappa_gait_probs = 0.07
        # 步态力的标准差
        gait_force_sigma = 50.
        # 步态速度的标准差
        gait_vel_sigma = 0.5
        # 脚摆高度
        footswing_height = 0.09

    class reward_scales(ParamsProto, cli=False):
        """
        定义和管理奖励函数中各个组成部分的权重。

        该类继承自ParamsProto，可用于命令行接口（CLI）设置中，以调整仿真环境中奖励函数的各个参数。

        Attributes:
        """
        termination = -0.0  # 终止条件的奖励权重
        tracking_lin_vel = 1.0  # 线速度跟踪的奖励权重
        tracking_ang_vel = 0.5  # 角速度跟踪的奖励权重
        lin_vel_z = -2.0  # Z轴线速度的奖励权重
        ang_vel_xy = -0.05  # XY平面内角速度的奖励权重
        orientation = -0.  # 朝向的奖励权重
        torques = -0.00001  # 扭矩的奖励权重
        dof_vel = -0.  # 关节速度的奖励权重
        dof_acc = -2.5e-7  # 关节加速度的奖励权重
        base_height = -0.  # 基座高度的奖励权重
        feet_air_time = 1.0  # 脚离地时间的奖励权重
        collision = -1.  # 碰撞的奖励权重
        feet_stumble = -0.0  # 脚部绊倒的奖励权重
        action_rate = -0.01  # 动作变化率的奖励权重
        stand_still = -0.  # 静止站立的奖励权重
        tracking_lin_vel_lat = 0.  # 横向线速度跟踪的奖励权重
        tracking_lin_vel_long = 0.  # 纵向线速度跟踪的奖励权重
        tracking_contacts = 0.  # 接触跟踪的奖励权重
        tracking_contacts_shaped = 0.  # 形状化接触跟踪的奖励权重
        tracking_contacts_shaped_force = 0.  # 形状化接触力跟踪的奖励权重
        tracking_contacts_shaped_vel = 0.  # 形状化接触速度跟踪的奖励权重
        jump = 0.0  # 跳跃的奖励权重
        energy = 0.0  # 能量的奖励权重
        energy_expenditure = 0.0  # 能量消耗的奖励权重
        survival = 0.0  # 生存的奖励权重
        dof_pos_limits = 0.0  # 关节位置限制的奖励权重
        feet_contact_forces = 0.  # 脚部接触力的奖励权重
        feet_slip = 0.  # 脚部滑动的奖励权重
        feet_clearance_cmd_linear = 0.  # 脚部离地命令的线性部分的奖励权重
        dof_pos = 0.  # 关节位置的奖励权重
        action_smoothness_1 = 0.  # 动作平滑性（一阶）的奖励权重
        action_smoothness_2 = 0.  # 动作平滑性（二阶）的奖励权重
        base_motion = 0.  # 基座运动的奖励权重
        feet_impact_vel = 0.0  # 脚部撞击速度的奖励权重
        raibert_heuristic = 0.0  # Raibert启发式的奖励权重

    class normalization(PrefixProto, cli=False):
        """
        提供一种标准化策略，用于将各种物理参数和状态标准化到合理的范围内。

        该类继承自PrefixProto，可用于命令行接口（CLI）设置中，以调整仿真环境中的物理参数和状态。

        Attributes:

        Examples:
            >>> norm = normalization()
            >>> print(norm.clip_observations)
            100.0

        Note:
            - 该类不直接参与物理模拟的计算，而是提供一组标准化参数，用于调整仿真环境的物理特性。
            - 通过调整这些参数，可以模拟不同的物理条件，如不同的地面摩擦、重力加速度等。
        """

        # 观测值的裁剪上限
        clip_observations = 100.
        # 动作值的裁剪上限
        clip_actions = 100.

        # 摩擦系数的范围
        friction_range = [0.05, 4.5]
        # 地面摩擦系数的范围
        ground_friction_range = [0.05, 4.5]
        # 弹性系数的范围
        restitution_range = [0, 1.0]
        # 附加质量的范围
        added_mass_range = [-1., 3.]
        # 质心位移的范围
        com_displacement_range = [-0.1, 0.1]
        # 电机强度的范围
        motor_strength_range = [0.9, 1.1]
        # 电机偏移的范围
        motor_offset_range = [-0.05, 0.05]
        # 比例增益系数的范围
        Kp_factor_range = [0.8, 1.3]
        # 微分增益系数的范围
        Kd_factor_range = [0.5, 1.5]
        # 关节摩擦系数的范围
        joint_friction_range = [0.0, 0.7]
        # 接触力的范围
        contact_force_range = [0.0, 50.0]
        # 接触状态的范围
        contact_state_range = [0.0, 1.0]
        # 身体速度的范围
        body_velocity_range = [-6.0, 6.0]
        # 足部高度的范围
        foot_height_range = [0.0, 0.15]
        # 身体高度的范围
        body_height_range = [0.0, 0.60]
        # 重力加速度的范围
        gravity_range = [-1.0, 1.0]
        # 运动状态的范围
        motion = [-0.01, 0.01]

    class obs_scales(PrefixProto, cli=False):
        """
        obs_scales子类，用于定义仿真环境中观察值的缩放比例。

        该配置类包含了线速度、角速度、关节位置、关节速度、IMU数据、高度测量、摩擦测量、身体高度命令等观察值的缩放比例。
        这些缩放比例用于调整观察值的范围，以适应不同的学习算法或提高学习效率。

        Attributes:

        Note:
            - 观察值的缩放比例对于控制策略的性能和学习效率有重要影响。
            - 缩放比例的设置应根据具体任务和学习算法的要求进行调整。
        """
        lin_vel = 2.0  # 线速度的缩放比例
        ang_vel = 0.25  # 角速度的缩放比例
        dof_pos = 1.0  # 关节位置的缩放比例
        dof_vel = 0.05  # 关节速度的缩放比例
        imu = 0.1  # IMU数据的缩放比例
        height_measurements = 5.0  # 高度测量的缩放比例
        friction_measurements = 1.0  # 摩擦测量的缩放比例
        body_height_cmd = 2.0  # 身体高度命令的缩放比例
        gait_phase_cmd = 1.0  # 步态相位命令的缩放比例
        gait_freq_cmd = 1.0  # 步态频率命令的缩放比例
        footswing_height_cmd = 0.15  # 脚摆高度命令的缩放比例
        body_pitch_cmd = 0.3  # 身体俯仰命令的缩放比例
        body_roll_cmd = 0.3  # 身体翻滚命令的缩放比例
        aux_reward_cmd = 1.0  # 辅助奖励命令的缩放比例
        compliance_cmd = 1.0  # 合规性命令的缩放比例
        stance_width_cmd = 1.0  # 站姿宽度命令的缩放比例
        stance_length_cmd = 1.0  # 站姿长度命令的缩放比例
        segmentation_image = 1.0  # 分割图像的缩放比例
        rgb_image = 1.0  # RGB图像的缩放比例
        depth_image = 1.0  # 深度图像的缩放比例

    class noise(PrefixProto, cli=False):
        """
        noise子类，用于定义仿真环境中噪声的配置参数。

        该配置类包含了是否添加噪声的开关、噪声水平等参数。噪声的引入可以使仿真环境更加接近真实世界的不确定性，
        对于测试和训练具有鲁棒性的控制策略非常有用。

        Attributes:
            add_noise (bool): 是否在仿真环境中添加噪声。
            noise_level (float): 噪声水平，用于缩放其他噪声参数的值。

        Note:
            - 添加噪声可以提高控制策略的鲁棒性，但也可能增加学习任务的难度。
            - 噪声水平的设置应根据具体任务和仿真环境的特性进行调整。
        """
        add_noise = True  # 是否在仿真环境中添加噪声
        noise_level = 1.0  # 噪声水平，用于缩放其他噪声参数的值

    class noise_scales(PrefixProto, cli=False):
        """
        noise_scales子类，用于定义仿真环境中各种测量和传感器数据的噪声比例。

        该配置类包含了关节位置、关节速度、线速度、角速度、IMU数据、重力、接触状态、高度测量、摩擦测量以及图像数据等的噪声比例。
        这些噪声比例参数用于模拟真实世界中的传感器误差和测量噪声，对于训练更加健壮的控制策略非常重要。

        Attributes:
            dof_pos (float): 关节位置的噪声比例。
            dof_vel (float): 关节速度的噪声比例。
            lin_vel (float): 线速度的噪声比例。
            ang_vel (float): 角速度的噪声比例。
            imu (float): IMU数据的噪声比例。
            gravity (float): 重力的噪声比例。
            contact_states (float): 接触状态的噪声比例。
            height_measurements (float): 高度测量的噪声比例。
            friction_measurements (float): 摩擦测量的噪声比例。
            segmentation_image (float): 分割图像的噪声比例。
            rgb_image (float): RGB图像的噪声比例。
            depth_image (float): 深度图像的噪声比例。

        Note:
            - 噪声比例的设置应根据实际应用场景和传感器特性进行调整。
            - 适当的噪声模拟可以帮助控制策略学习到更加鲁棒的行为。
        """
        dof_pos = 0.01  # 关节位置的噪声比例
        dof_vel = 1.5  # 关节速度的噪声比例
        lin_vel = 0.1  # 线速度的噪声比例
        ang_vel = 0.2  # 角速度的噪声比例
        imu = 0.1  # IMU数据的噪声比例
        gravity = 0.05  # 重力的噪声比例
        contact_states = 0.05  # 接触状态的噪声比例
        height_measurements = 0.1  # 高度测量的噪声比例
        friction_measurements = 0.0  # 摩擦测量的噪声比例，设置为0表示无噪声
        segmentation_image = 0.0  # 分割图像的噪声比例，设置为0表示无噪声
        rgb_image = 0.0  # RGB图像的噪声比例，设置为0表示无噪声
        depth_image = 0.0  # 深度图像的噪声比例，设置为0表示无噪声

    # viewer camera:
    class viewer(PrefixProto, cli=False):
        """
        viewer子类，用于定义仿真环境观察者（视图）的配置参数。

        该配置类包含了观察者的参考环境编号、位置和观察点。这些参数决定了观察者在仿真环境中的视角和观察的焦点，
        对于进行仿真时的视觉效果和分析非常重要。

        Attributes:
            ref_env (int): 观察者的参考环境编号，用于确定观察的是哪一个仿真环境。
            pos (list): 观察者的位置坐标[x, y, z]，单位为米[m]。
            lookat (list): 观察者的观察点坐标[x, y, z]，单位为米[m]。

        Note:
            - 观察者的位置和观察点设置对于调试和展示仿真效果非常关键。
            - 通过调整这些参数，可以从不同的角度和距离观察仿真环境，以获得最佳的视觉效果。
        """
        ref_env = 0  # 观察者的参考环境编号
        pos = [10, 0, 6]  # 观察者的位置坐标[x, y, z]，单位为米[m]
        lookat = [11., 5, 3.]  # 观察者的观察点坐标[x, y, z]，单位为米[m]

    class sim(PrefixProto, cli=False):
        """
        sim子类，用于定义仿真相关的配置参数。

        该配置类包含了仿真的基本设置，如时间步长、子步数、重力加速度、是否使用GPU加速等。此外，还包括了物理引擎特定的配置，
        如PhysX引擎的线程数、求解器类型、接触偏移等参数。

        Attributes:
            dt (float): 仿真的时间步长。
            substeps (int): 每个仿真步骤的子步数。
            gravity (list): 重力加速度向量[m/s^2]。
            up_axis (int): 上轴的方向，0表示y轴，1表示z轴。
            use_gpu_pipeline (bool): 是否使用GPU加速。
            physx: PhysX引擎相关的配置参数。

        Note:
            - 使用GPU加速可以显著提高仿真的效率，特别是在处理大量环境时。
            - 物理引擎的配置对仿真的准确性和性能有重要影响。
        """
        dt = 0.005  # 仿真的时间步长
        substeps = 1  # 每个仿真步骤的子步数
        gravity = [0., 0., -9.81]  # 重力加速度向量[m/s^2]
        up_axis = 1  # 上轴的方向，0表示y轴，1表示z轴
        use_gpu_pipeline = True  # 是否使用GPU加速

        class physx(PrefixProto, cli=False):
            """
            physx子类，用于定义PhysX物理引擎相关的配置参数。
            """
            num_threads = 10  # PhysX引擎使用的线程数
            solver_type = 1  # 求解器类型，0: pgs, 1: tgs
            num_position_iterations = 4  # 位置迭代次数
            num_velocity_iterations = 0  # 速度迭代次数
            contact_offset = 0.01  # 接触偏移[m]
            rest_offset = 0.0  # 静止偏移[m]
            bounce_threshold_velocity = 0.5  # 弹跳阈值速度[m/s]
            max_depenetration_velocity = 1.0  # 最大去穿透速度
            max_gpu_contact_pairs = 2 ** 23  # GPU接触对的最大数量，2**24 -> 对于8000个及以上的环境是必需的
            default_buffer_size_multiplier = 5  # 默认缓冲区大小乘数
            contact_collection = 2  # 接触收集，0: 从不，1: 最后一个子步骤，2: 所有子步骤（默认=2）
