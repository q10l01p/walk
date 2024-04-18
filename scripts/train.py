def train_go1(headless=True):
    """
    训练GO1机器人模型。

    此函数配置并启动GO1机器人的训练过程。包括环境配置、奖励设置、动作空间定义等。

    Attributes:
        headless (bool): 是否在无头模式下运行，即不显示图形界面。

    Methods:
        config_go1: 配置GO1机器人的环境参数。
        VelocityTrackingEasyEnv: 初始化GO1机器人的环境。
        logger.log_params: 记录实验参数。
        Runner.learn: 启动训练过程。

    Examples:
        >>> train_go1(headless=True)  # 在无头模式下训练GO1机器人

    Note:
        - 请确保机器配置能够满足训练需求。
        - 训练过程可能需要较长时间。
    """
    import isaacgym
    assert isaacgym
    import torch

    from go1_gym.envs.base.legged_robot_config import Cfg
    from go1_gym.envs.go1.go1_config import config_go1
    from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

    from ml_logger import logger

    from go1_gym_learn.ppo_cse import Runner
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from go1_gym_learn.ppo_cse.actor_critic import AC_Args
    from go1_gym_learn.ppo_cse.ppo import PPO_Args
    from go1_gym_learn.ppo_cse import RunnerArgs

    # 调用配置函数，配置GO1机器人的环境参数
    config_go1(Cfg)

    # 设置线性速度和角速度的分箱数量
    Cfg.commands.num_lin_vel_bins = 30
    Cfg.commands.num_ang_vel_bins = 30

    # 设置追踪角速度、线速度和接触力的曲线阈值
    Cfg.curriculum_thresholds.tracking_ang_vel = 0.7
    Cfg.curriculum_thresholds.tracking_lin_vel = 0.8
    Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90

    # 启用分布式命令
    Cfg.commands.distributional_commands = True

    # 设置域随机化中的延迟时间步和随机化延迟时间步
    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True

    # 设置控制类型为“actuator_net”
    Cfg.control.control_type = "actuator_net"

    # 设置在仿真开始后是否随机化刚体属性
    Cfg.domain_rand.randomize_rigids_after_start = False
    # 设置是否观察运动信息
    Cfg.env.priv_observe_motion = False
    # 设置是否观察经重力变换后的运动信息
    Cfg.env.priv_observe_gravity_transformed_motion = False
    # 设置是否随机化独立摩擦系数
    Cfg.domain_rand.randomize_friction_indep = False
    # 设置是否观察独立摩擦系数
    Cfg.env.priv_observe_friction_indep = False
    # 设置是否随机化摩擦系数
    Cfg.domain_rand.randomize_friction = True
    # 设置是否观察摩擦系数
    Cfg.env.priv_observe_friction = True
    # 设置摩擦系数的随机化范围
    Cfg.domain_rand.friction_range = [0.1, 3.0]
    # 设置是否随机化恢复系数
    Cfg.domain_rand.randomize_restitution = True
    # 设置是否观察恢复系数
    Cfg.env.priv_observe_restitution = True
    # 设置恢复系数的随机化范围
    Cfg.domain_rand.restitution_range = [0.0, 0.4]
    # 设置是否随机化基座质量
    Cfg.domain_rand.randomize_base_mass = True
    # 设置是否观察基座质量
    Cfg.env.priv_observe_base_mass = False
    # 设置额外质量的随机化范围
    Cfg.domain_rand.added_mass_range = [-1.0, 3.0]
    # 设置是否随机化重力
    Cfg.domain_rand.randomize_gravity = True
    # 设置重力的随机化范围
    Cfg.domain_rand.gravity_range = [-1.0, 1.0]
    # 设置重力随机化的间隔时间
    Cfg.domain_rand.gravity_rand_interval_s = 8.0
    # 设置重力脉冲的持续时间
    Cfg.domain_rand.gravity_impulse_duration = 0.99
    # 设置是否观察重力信息
    Cfg.env.priv_observe_gravity = False
    # 设置是否随机化质心位移
    Cfg.domain_rand.randomize_com_displacement = False
    # 设置质心位移的随机化范围
    Cfg.domain_rand.com_displacement_range = [-0.15, 0.15]
    # 设置是否观察质心位移
    Cfg.env.priv_observe_com_displacement = False
    # 设置是否随机化地面摩擦
    Cfg.domain_rand.randomize_ground_friction = True
    # 设置是否观察地面摩擦
    Cfg.env.priv_observe_ground_friction = False
    # 设置是否观察每只脚的地面摩擦
    Cfg.env.priv_observe_ground_friction_per_foot = False
    # 设置地面摩擦的随机化范围
    Cfg.domain_rand.ground_friction_range = [0.0, 0.0]
    # 设置是否随机化电机强度
    Cfg.domain_rand.randomize_motor_strength = True
    # 设置电机强度的随机化范围
    Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    # 设置是否观察电机强度
    Cfg.env.priv_observe_motor_strength = False
    # 设置是否随机化电机偏移
    Cfg.domain_rand.randomize_motor_offset = True
    # 设置电机偏移的随机化范围
    Cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    # 设置是否观察电机偏移
    Cfg.env.priv_observe_motor_offset = False
    # 设置是否推动机器人
    Cfg.domain_rand.push_robots = False
    # 设置是否随机化Kp因子
    Cfg.domain_rand.randomize_Kp_factor = False
    # 设置是否观察Kp因子
    Cfg.env.priv_observe_Kp_factor = False
    # 设置是否随机化Kd因子
    Cfg.domain_rand.randomize_Kd_factor = False
    # 设置是否观察Kd因子
    Cfg.env.priv_observe_Kd_factor = False
    # 设置是否观察机器人的身体速度
    Cfg.env.priv_observe_body_velocity = False
    # 设置是否观察机器人的身体高度
    Cfg.env.priv_observe_body_height = False
    # 设置是否观察期望的接触状态
    Cfg.env.priv_observe_desired_contact_states = False
    # 设置是否观察接触力
    Cfg.env.priv_observe_contact_forces = False
    # 设置是否观察脚的位移
    Cfg.env.priv_observe_foot_displacement = False
    # 设置是否观察经重力变换后的脚位移
    Cfg.env.priv_observe_gravity_transformed_foot_displacement = False

    # 设置特权观察的数量
    Cfg.env.num_privileged_obs = 2  # 设置特权观察的数量为2
    # 设置观察历史的长度
    Cfg.env.num_observation_history = 10  # 设置观察历史的长度为10
    # 设置脚接触力的奖励比例
    Cfg.reward_scales.feet_contact_forces = 0.0  # 设置脚接触力的奖励比例为0.0，即不奖励脚接触力

    # 设置域随机化的间隔时间
    Cfg.domain_rand.rand_interval_s = 4
    # 设置命令的数量
    Cfg.commands.num_commands = 15
    # 设置是否观察前两个动作
    Cfg.env.observe_two_prev_actions = True
    # 设置是否观察偏航角
    Cfg.env.observe_yaw = False
    # 设置观察的总数量
    Cfg.env.num_observations = 70
    # 设置标量观察的数量
    Cfg.env.num_scalar_observations = 70
    # 设置是否观察步态命令
    Cfg.env.observe_gait_commands = True
    # 设置是否观察时间参数
    Cfg.env.observe_timing_parameter = False
    # 设置是否观察时钟输入
    Cfg.env.observe_clock_inputs = True

    # 设置地形高度的随机化范围
    Cfg.domain_rand.tile_height_range = [-0.0, 0.0]
    # 设置是否使用地形高度的课程学习
    Cfg.domain_rand.tile_height_curriculum = False
    # 设置地形高度更新的间隔
    Cfg.domain_rand.tile_height_update_interval = 1000000
    # 设置地形高度课程学习的步长
    Cfg.domain_rand.tile_height_curriculum_step = 0.01
    # 设置地形边界的大小
    Cfg.terrain.border_size = 0.0
    # 设置地形的网格类型
    Cfg.terrain.mesh_type = "trimesh"
    # 设置地形的列数
    Cfg.terrain.num_cols = 10
    # 设置地形的行数
    Cfg.terrain.num_rows = 10
    # 设置地形的宽度
    Cfg.terrain.terrain_width = 5.0
    # 设置地形的长度
    Cfg.terrain.terrain_length = 5.0
    # 设置机器人初始位置的X轴范围
    Cfg.terrain.x_init_range = 0.2
    # 设置机器人初始位置的Y轴范围
    Cfg.terrain.y_init_range = 0.2
    # 设置机器人传送的阈值
    Cfg.terrain.teleport_thresh = 0.3
    # 设置是否传送机器人
    Cfg.terrain.teleport_robots = False
    # 设置是否将机器人置于地形中心
    Cfg.terrain.center_robots = True
    # 设置地形中心的跨度
    Cfg.terrain.center_span = 4
    # 设置地形的水平缩放比例
    Cfg.terrain.horizontal_scale = 0.10
    # 设置是否使用足部高度作为终止奖励条件
    Cfg.rewards.use_terminal_foot_height = False
    # 设置是否使用身体高度作为终止奖励条件
    Cfg.rewards.use_terminal_body_height = True
    # 设置终止奖励条件的身体高度阈值
    Cfg.rewards.terminal_body_height = 0.05
    # 设置是否使用身体倾斜角度作为终止奖励条件
    Cfg.rewards.use_terminal_roll_pitch = True
    # 设置终止奖励条件的身体倾斜角度阈值
    Cfg.rewards.terminal_body_ori = 1.6

    # 设置命令重采样时间
    Cfg.commands.resampling_time = 10

    # 设置各种行为的奖励比例
    Cfg.reward_scales.feet_slip = -0.04  # 脚滑动
    Cfg.reward_scales.action_smoothness_1 = -0.1  # 动作平滑性1
    Cfg.reward_scales.action_smoothness_2 = -0.1  # 动作平滑性2
    Cfg.reward_scales.dof_vel = -1e-4  # 关节速度
    Cfg.reward_scales.dof_pos = -0.0  # 关节位置
    Cfg.reward_scales.jump = 10.0  # 跳跃
    Cfg.reward_scales.base_height = 0.0  # 基座高度
    Cfg.rewards.base_height_target = 0.30  # 基座高度目标
    Cfg.reward_scales.estimation_bonus = 0.0  # 估计奖励
    Cfg.reward_scales.raibert_heuristic = -10.0  # Raibert启发式
    Cfg.reward_scales.feet_impact_vel = -0.0  # 脚部撞击速度
    Cfg.reward_scales.feet_clearance = -0.0  # 脚部间隙
    Cfg.reward_scales.feet_clearance_cmd = -0.0  # 脚部间隙命令
    Cfg.reward_scales.feet_clearance_cmd_linear = -30.0  # 脚部间隙命令线性
    Cfg.reward_scales.orientation = 0.0  # 方向
    Cfg.reward_scales.orientation_control = -5.0  # 方向控制
    Cfg.reward_scales.tracking_stance_width = -0.0  # 跟踪步幅宽度
    Cfg.reward_scales.tracking_stance_length = -0.0  # 跟踪步幅长度
    Cfg.reward_scales.lin_vel_z = -0.02  # 线速度Z
    Cfg.reward_scales.ang_vel_xy = -0.001  # 角速度XY
    Cfg.reward_scales.feet_air_time = 0.0  # 脚部空中时间
    Cfg.reward_scales.hop_symmetry = 0.0  # 跳跃对称性
    Cfg.rewards.kappa_gait_probs = 0.07  # 步态概率Kappa
    Cfg.rewards.gait_force_sigma = 100.  # 步态力Sigma
    Cfg.rewards.gait_vel_sigma = 10.  # 步态速度Sigma
    Cfg.reward_scales.tracking_contacts_shaped_force = 4.0  # 跟踪接触形状力
    Cfg.reward_scales.tracking_contacts_shaped_vel = 4.0  # 跟踪接触形状速度
    Cfg.reward_scales.collision = -5.0  # 碰撞

    # 设置奖励容器的名称
    Cfg.rewards.reward_container_name = "CoRLRewards"  # 使用CoRL会议的奖励计算方法
    # 设置是否仅使用正奖励
    Cfg.rewards.only_positive_rewards = False  # 允许使用负奖励
    # 设置是否采用JI22风格的仅正奖励方法
    Cfg.rewards.only_positive_rewards_ji22_style = True  # 采用JI22风格，即使是负奖励也以正奖励的形式给出
    # 设置负奖励的标准差
    Cfg.rewards.sigma_rew_neg = 0.02  # 负奖励的标准差，用于计算负奖励的变异性

    # 设置线速度x的命令范围
    Cfg.commands.lin_vel_x = [-1.0, 1.0]
    # 设置线速度y的命令范围
    Cfg.commands.lin_vel_y = [-0.6, 0.6]
    # 设置角速度yaw的命令范围
    Cfg.commands.ang_vel_yaw = [-1.0, 1.0]
    # 设置身体高度的命令范围
    Cfg.commands.body_height_cmd = [-0.25, 0.15]
    # 设置步态频率的命令范围
    Cfg.commands.gait_frequency_cmd_range = [2.0, 4.0]
    # 设置步态相位的命令范围
    Cfg.commands.gait_phase_cmd_range = [0.0, 1.0]
    # 设置步态偏移的命令范围
    Cfg.commands.gait_offset_cmd_range = [0.0, 1.0]
    # 设置步态边界的命令范围
    Cfg.commands.gait_bound_cmd_range = [0.0, 1.0]
    # 设置步态持续时间的命令范围
    Cfg.commands.gait_duration_cmd_range = [0.5, 0.5]
    # 设置脚摆高度的命令范围
    Cfg.commands.footswing_height_range = [0.03, 0.35]
    # 设置身体俯仰角的命令范围
    Cfg.commands.body_pitch_range = [-0.4, 0.4]
    # 设置身体侧倾角的命令范围
    Cfg.commands.body_roll_range = [-0.0, 0.0]
    # 设置站立宽度的命令范围
    Cfg.commands.stance_width_range = [0.10, 0.45]
    # 设置站立长度的命令范围
    Cfg.commands.stance_length_range = [0.35, 0.45]

    # 设置线速度x的限制范围
    Cfg.commands.limit_vel_x = [-5.0, 5.0]
    # 设置线速度y的限制范围
    Cfg.commands.limit_vel_y = [-0.6, 0.6]
    # 设置角速度yaw的限制范围
    Cfg.commands.limit_vel_yaw = [-5.0, 5.0]
    # 设置身体高度的限制范围
    Cfg.commands.limit_body_height = [-0.25, 0.15]
    # 设置步态频率的限制范围
    Cfg.commands.limit_gait_frequency = [2.0, 4.0]
    # 设置步态相位的限制范围
    Cfg.commands.limit_gait_phase = [0.0, 1.0]
    # 设置步态偏移的限制范围
    Cfg.commands.limit_gait_offset = [0.0, 1.0]
    # 设置步态边界的限制范围
    Cfg.commands.limit_gait_bound = [0.0, 1.0]
    # 设置步态持续时间的限制范围
    Cfg.commands.limit_gait_duration = [0.5, 0.5]
    # 设置脚摆高度的限制范围
    Cfg.commands.limit_footswing_height = [0.03, 0.35]
    # 设置身体俯仰角的限制范围
    Cfg.commands.limit_body_pitch = [-0.4, 0.4]
    # 设置身体侧倾角的限制范围
    Cfg.commands.limit_body_roll = [-0.0, 0.0]
    # 设置站立宽度的限制范围
    Cfg.commands.limit_stance_width = [0.10, 0.45]
    # 设置站立长度的限制范围
    Cfg.commands.limit_stance_length = [0.35, 0.45]

    # 设置速度x分量的离散化箱数
    Cfg.commands.num_bins_vel_x = 21
    # 设置速度y分量的离散化箱数
    Cfg.commands.num_bins_vel_y = 1
    # 设置速度yaw分量的离散化箱数
    Cfg.commands.num_bins_vel_yaw = 21
    # 设置身体高度的离散化箱数
    Cfg.commands.num_bins_body_height = 1
    # 设置步态频率的离散化箱数
    Cfg.commands.num_bins_gait_frequency = 1
    # 设置步态相位的离散化箱数
    Cfg.commands.num_bins_gait_phase = 1
    # 设置步态偏移的离散化箱数
    Cfg.commands.num_bins_gait_offset = 1
    # 设置步态边界的离散化箱数
    Cfg.commands.num_bins_gait_bound = 1
    # 设置步态持续时间的离散化箱数
    Cfg.commands.num_bins_gait_duration = 1
    # 设置足摆高度的离散化箱数
    Cfg.commands.num_bins_footswing_height = 1
    # 设置身体滚动的离散化箱数
    Cfg.commands.num_bins_body_roll = 1
    # 设置身体俯仰的离散化箱数
    Cfg.commands.num_bins_body_pitch = 1
    # 设置站姿宽度的离散化箱数
    Cfg.commands.num_bins_stance_width = 1

    # 设置摩擦力规范化范围
    Cfg.normalization.friction_range = [0, 1]
    # 设置地面摩擦力规范化范围
    Cfg.normalization.ground_friction_range = [0, 1]
    # 设置机器人初始朝向范围
    Cfg.terrain.yaw_init_range = 3.14
    # 设置动作剪辑值
    Cfg.normalization.clip_actions = 10.0

    # 设置步态相位是否互斥
    Cfg.commands.exclusive_phase_offset = False
    # 设置步态偏移
    Cfg.commands.pacing_offset = False
    # 设置二进制相位
    Cfg.commands.binary_phases = True
    # 设置步态课程学习
    Cfg.commands.gaitwise_curricula = True

    # 初始化环境，指定仿真设备为cuda:0，启用无头模式
    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=True, cfg=Cfg)

    # 记录实验参数
    logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                      Cfg=vars(Cfg))

    # 使用HistoryWrapper对环境进行包装，以便记录和使用过去的观察结果
    env = HistoryWrapper(env)
    # 指定使用的GPU ID
    gpu_id = 0
    # 初始化训练器，指定训练使用的设备
    runner = Runner(env, device=f"cuda:{gpu_id}")
    # 启动训练过程，指定训练迭代次数、是否在随机长度的情况下初始化、以及评估频率
    runner.learn(num_learning_iterations=100000, init_at_random_ep_len=True, eval_freq=100)
    

if __name__ == '__main__':
    """
    主程序入口。

    当此脚本被直接运行时，将配置日志记录器，记录训练过程中的关键指标，并启动GO1机器人的训练过程。

    Note:
        - 为了可视化训练过程，请将headless参数设置为False。
        - 日志记录器将保存训练过程中的关键指标和视频，便于后续分析和评估。
    """
    # 导入必要的库
    from pathlib import Path
    from ml_logger import logger
    from go1_gym import MINI_GYM_ROOT_DIR

    # 获取当前脚本的文件名（不含扩展名）
    stem = Path(__file__).stem
    # 配置日志记录器
    logger.configure(logger.utcnow(f'gait-conditioned-agility/%Y-%m-%d/{stem}/%H%M%S.%f'),
                     root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
    # 记录训练过程中的关键指标到.charts.yml文件
    logger.log_text("""
                charts: 
                - yKey: train/episode/rew_total/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_lin_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_force/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_1/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_smoothness_2/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_contacts_shaped_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_orientation_control/mean
                  xKey: iterations
                - yKey: train/episode/rew_dof_pos/mean
                  xKey: iterations
                - yKey: train/episode/command_area_trot/mean
                  xKey: iterations
                - yKey: train/episode/max_terrain_height/mean
                  xKey: iterations
                - type: video
                  glob: "videos/*.mp4"
                - yKey: adaptation_loss/mean
                  xKey: iterations
                """, filename=".charts.yml", dedent=True)

    # 启动训练过程，为了可视化训练过程，设置headless=False
    train_go1(headless=False)
