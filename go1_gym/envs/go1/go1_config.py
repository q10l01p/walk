from typing import Union

from params_proto import Meta

from go1_gym.envs.base.legged_robot_config import Cfg


def config_go1(Cnfg: Union[Cfg, Meta]):
    """
    配置GO1机器人的仿真环境和控制参数。

    该函数用于设置GO1机器人在仿真环境中的初始状态、控制策略、物理属性、奖励机制、地形特性以及域随机化参数等。
    通过这些配置，可以定制化仿真环境以满足不同的实验需求。

    Parameters:
        Cnfg (Union[Cfg, Meta]): 配置对象，用于设置和存储仿真环境的各种参数。

    Methods:
        - init_state: 机器人的初始状态
        - control: 控制参数
        - asset: 机器人资产
        - rewards: 奖励机制
        - reward_scales: 奖励缩放
        - terrain: 地形
        - env: 环境参数
        - commands: 控制命令
        - domain_rand: 域随机化

    Examples:
        >>> from go1_gym.envs.go1.go1_config import config_go1
        >>> from go1_gym.envs.base.legged_robot_config import Cfg
        >>> cfg = Cfg()
        >>> config_go1(cfg)

    Note:
        - 该函数不返回任何值，而是直接修改传入的配置对象。
        - 配置参数包括机器人的初始位置、关节角度、控制类型、动作缩放比例、URDF文件路径、奖励机制等。
    """
    # 初始化仿真环境状态
    _ = Cnfg.init_state
    # 设置机器人的初始位置
    _.pos = [0.0, 0.0, 0.34]  # x,y,z [m]
    # 设置关节的默认角度，这些角度是当动作为0时的目标角度
    _.default_joint_angles = {
        'FL_hip_joint': 0.1,  # [rad]
        'RL_hip_joint': 0.1,  # [rad]
        'FR_hip_joint': -0.1,  # [rad]
        'RR_hip_joint': -0.1,  # [rad]
        'FL_thigh_joint': 0.8,  # [rad]
        'RL_thigh_joint': 1.,  # [rad]
        'FR_thigh_joint': 0.8,  # [rad]
        'RR_thigh_joint': 1.,  # [rad]
        'FL_calf_joint': -1.5,  # [rad]
        'RL_calf_joint': -1.5,  # [rad]
        'FR_calf_joint': -1.5,  # [rad]
        'RR_calf_joint': -1.5  # [rad]
    }

    # 配置控制参数
    _ = Cnfg.control
    _.control_type = 'P'  # 控制类型
    _.stiffness = {'joint': 20.}  # 关节刚度 [N*m/rad]
    _.damping = {'joint': 0.5}  # 关节阻尼 [N*m*s/rad]
    _.action_scale = 0.25  # 动作缩放比例
    _.hip_scale_reduction = 0.5  # 髋关节缩放比例减少
    _.decimation = 4  # 控制动作更新的减采样率

    # 配置机器人资产
    _ = Cnfg.asset
    _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'  # URDF文件路径
    _.foot_name = "foot"  # 脚部名称
    _.penalize_contacts_on = ["thigh", "calf"]  # 对大腿和小腿的接触进行惩罚
    _.terminate_after_contacts_on = ["base"]  # 基座接触后终止
    _.self_collisions = 0  # 自碰撞设置（1禁用，0启用）
    _.flip_visual_attachments = False  # 翻转视觉附件
    _.fix_base_link = False  # 固定基座链接

    # 配置奖励机制
    _ = Cnfg.rewards
    _.soft_dof_pos_limit = 0.9  # 软关节位置限制
    _.base_height_target = 0.34  # 基座高度目标

    # 配置奖励缩放
    _ = Cnfg.reward_scales
    _.torques = -0.0001  # 扭矩奖励缩放
    _.action_rate = -0.01  # 动作速率奖励缩放
    _.dof_pos_limits = -10.0  # 关节位置限制奖励缩放
    _.orientation = -5.  # 朝向奖励缩放
    _.base_height = -30.  # 基座高度奖励缩放

    # 配置地形
    _ = Cnfg.terrain
    _.mesh_type = 'trimesh'  # 地形网格类型
    _.measure_heights = False  # 测量高度
    _.terrain_noise_magnitude = 0.0  # 地形噪声幅度
    _.teleport_robots = True  # 机器人传送
    _.border_size = 50  # 边界大小
    _.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]  # 地形比例
    _.curriculum = False  # 课程学习

    # 配置环境参数
    _ = Cnfg.env
    _.num_observations = 42  # 观测数量
    _.observe_vel = False  # 观测速度
    _.num_envs = 4000  # 环境数量

    # 配置控制命令
    _ = Cnfg.commands
    _.lin_vel_x = [-1.0, 1.0]  # 线速度x范围
    _.lin_vel_y = [-1.0, 1.0]  # 线速度y范围
    _.heading_command = False  # 航向命令
    _.resampling_time = 10.0  # 重采样时间
    _.command_curriculum = True  # 命令课程
    _.num_lin_vel_bins = 30  # 线速度分箱数量
    _.num_ang_vel_bins = 30  # 角速度分箱数量
    _.lin_vel_x = [-0.6, 0.6]  # 线速度x范围（更新）
    _.lin_vel_y = [-0.6, 0.6]  # 线速度y范围（更新）
    _.ang_vel_yaw = [-1, 1]  # 角速度偏航范围

    # 配置域随机化
    _ = Cnfg.domain_rand
    _.randomize_base_mass = True  # 随机化基座质量
    _.added_mass_range = [-1, 3]  # 添加质量范围
    _.push_robots = False  # 推动机器人
    _.max_push_vel_xy = 0.5  # 最大推动速度xy
    _.randomize_friction = True  # 随机化摩擦
    _.friction_range = [0.05, 4.5]  # 摩擦范围
    _.randomize_restitution = True  # 随机化恢复系数
    _.restitution_range = [0.0, 1.0]  # 恢复系数范围
    _.restitution = 0.5  # 默认地形恢复系数
    _.randomize_com_displacement = True  # 随机化质心位移
    _.com_displacement_range = [-0.1, 0.1]  # 质心位移范围
    _.randomize_motor_strength = True  # 随机化电机强度
    _.motor_strength_range = [0.9, 1.1]  # 电机强度范围
    _.randomize_Kp_factor = False  # 随机化Kp因子
    _.Kp_factor_range = [0.8, 1.3]  # Kp因子范围
    _.randomize_Kd_factor = False  # 随机化Kd因子
    _.Kd_factor_range = [0.5, 1.5]  # Kd因子范围
    _.rand_interval_s = 6  # 随机化间隔秒
