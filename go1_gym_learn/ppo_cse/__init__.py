import time
from collections import deque
import copy
import os
import pickle

import torch
import glob
from ml_logger import logger
from params_proto import PrefixProto

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage


def class_to_dict(obj) -> dict:
    """
    将对象及其属性转换为字典形式。
    该函数递归地遍历对象的属性，将其转换为字典形式，便于JSON序列化或其他用途。
    对象的私有属性和特定属性（如'terrain'）将被忽略。

    Attributes:
        obj (object): 需要转换的对象。

    Returns:
        dict: 对象属性的字典表示。

    Examples:
        >>> class Example:
        >>>     def __init__(self, name, value):
        >>>         self.name = name
        >>>         self.value = value
        >>>
        >>> example = Example("example", 123)
        >>> print(class_to_dict(example))
        {'name': 'example', 'value': 123}

    Note:
        - 私有属性和以'_'开头的属性不会被转换。
        - 特定属性'terrain'也将被忽略。
    """
    # 检查对象是否有__dict__属性，如果没有直接返回对象本身
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}  # 初始化结果字典
    # 遍历对象的所有属性
    for key in dir(obj):
        # 忽略私有属性和'terrain'属性
        if key.startswith("_") or key == "terrain":
            continue
        element = []  # 初始化属性值，用于存放转换后的值
        val = getattr(obj, key)  # 获取属性值
        # 如果属性值是列表，递归转换列表中的每个元素
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            # 对于非列表属性值，直接递归转换
            element = class_to_dict(val)
        result[key] = element  # 将转换后的值添加到结果字典中
    return result  # 返回转换后的字典


class DataCaches:
    """
    DataCaches类用于管理和存储数据缓存。
    该类初始化时会创建两种类型的缓存：SlotCache和DistCache。
    SlotCache用于按课程分组存储数据，而DistCache用于存储数据的分布情况。

    Attributes:
        slot_cache (SlotCache): 用于按课程分组存储数据的缓存。
        dist_cache (DistCache): 用于存储数据分布情况的缓存。

    Methods:
        __init__(self, curriculum_bins): 类的构造函数，用于初始化数据缓存。

    Examples:
        >>> caches = DataCaches(1)
        创建一个DataCaches实例，其中curriculum_bins设置为1。

    Note:
        - SlotCache和DistCache是从go1_gym_learn.ppo.metrics_caches模块导入的。
    """

    def __init__(self, curriculum_bins):
        """
        初始化DataCaches实例。

        Parameters:
            curriculum_bins (int): 用于SlotCache初始化的课程分组数。
        """
        # 从go1_gym_learn.ppo.metrics_caches模块导入SlotCache和DistCache类
        from go1_gym_learn.ppo.metrics_caches import SlotCache, DistCache

        # 初始化SlotCache实例，传入curriculum_bins参数
        self.slot_cache = SlotCache(curriculum_bins)
        # 初始化DistCache实例
        self.dist_cache = DistCache()


# 创建DataCaches实例，传入参数1表示curriculum_bins设置为1
caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    """
    RunnerArgs类用于定义运行参数。
    该类包含算法运行所需的各种参数，包括算法类名、每个环境的步数、最大迭代次数、日志保存间隔、视频保存间隔、日志频率、恢复运行设置等。

    Attributes:
        algorithm_class_name (str): 算法类名。
        num_steps_per_env (int): 每个环境的步数，即每次迭代中的步数。
        max_iterations (int): 最大迭代次数，即策略更新的次数。
        save_interval (int): 检查潜在保存的迭代间隔。
        save_video_interval (int): 视频保存间隔。
        log_freq (int): 日志记录频率。
        resume (bool): 是否从之前的运行中恢复。
        load_run (int): 加载的运行编号，-1表示最后一次运行。
        checkpoint (int): 加载的模型检查点，-1表示最后保存的模型。
        resume_path (str): 恢复路径，根据load_run和checkpoint更新。
        resume_curriculum (bool): 是否恢复课程。

    Examples:
        >>> runner_args = RunnerArgs()
        创建一个RunnerArgs实例，使用默认参数。

    Note:
        - PrefixProto是一个假设的父类，用于提供某些功能或属性。
        - cli参数在此处未使用，假设用于控制命令行接口的某些行为。
    """

    # runner参数
    algorithm_class_name = 'RMA'  # 算法类名
    num_steps_per_env = 24  # 每个环境的步数，即每次迭代中的步数
    max_iterations = 1500  # 最大迭代次数，即策略更新的次数

    # 日志参数
    save_interval = 400  # 检查潜在保存的迭代间隔
    save_video_interval = 100  # 视频保存间隔
    log_freq = 10  # 日志记录频率

    # 加载和恢复参数
    resume = True  # 是否从之前的运行中恢复
    load_run = 0  # 加载的运行编号，-1表示最后一次运行
    checkpoint = -1  # 加载的模型检查点，-1表示最后保存的模型

    dirs = glob.glob(f"../runs/gait-conditioned-agility/*/train/*")
    logdir = sorted(dirs)[load_run]

    resume_path = logdir[:]  # 根据自己情况修改只要能找到文件即可
    resume_curriculum = True


class Runner:

    def __init__(self, env, device='cpu'):
        """
        初始化模型和环境。

        该方法负责初始化模型、环境以及相关的配置。如果设置为从之前的状态恢复，则会加载预训练的权重和课程状态。

        Attributes:
            env (Environment): 环境对象，包含观察空间和动作空间的定义。
            device (str): 指定模型运行的设备，默认为'cpu'。

        Methods:
            __init__(self, env, device='cpu'): 类的构造函数，用于初始化环境和模型。

        Examples:
            >>> env = YourEnvironment()
            >>> model = YourModel(env)
            初始化模型实例，传入环境对象。

        Note:
            - 需要从ppo模块导入PPO类。
            - 如果启用了恢复功能，需要从ml_logger模块导入ML_Logger类用于加载权重和课程状态。
        """
        # 从ppo模块导入PPO类
        from .ppo import PPO

        self.device = device  # 设置运行设备
        self.env = env  # 设置环境

        # 初始化ActorCritic模型，并将其移至指定设备
        actor_critic = ActorCritic(self.env.num_obs,
                                   self.env.num_privileged_obs,
                                   self.env.num_obs_history,
                                   self.env.num_actions,
                                   ).to(self.device)

        # 从给定的路径恢复训练。
        if RunnerArgs.resume:
            # load pretrained weights from resume_path
            from ml_logger import ML_Logger

            # 从ml_logger模块导入ML_Logger类
            loader = ML_Logger(root="http://127.0.0.1:8081",  # ML_Logger 需要若不该会报错
                               prefix=RunnerArgs.resume_path)

            # 原来的loader.load_torch会报错这样该可以使用
            weights = torch.load(RunnerArgs.resume_path + "/checkpoints/ac_weights_last.pt")

            actor_critic.load_state_dict(state_dict=weights)

            if hasattr(self.env, "curricula") and RunnerArgs.resume_curriculum:
                # load curriculum state
                # 原来的loader.load_pkl会报错，使用pickle读取.pkl文件
                f = open(RunnerArgs.resume_path + '/curriculum/distribution.pkl', 'rb')
                distributions = pickle.load(f)
                # 也进行了修改，为了找到weights_
                distribution_last = distributions["distribution"]
                gait_names = [key[8:] if key.startswith("weights_") else None for key in distribution_last.keys()]
                for gait_id, gait_name in enumerate(self.env.category_names):
                    self.env.curricula[gait_id].weights = distribution_last[f"weights_{gait_name}"]
                    print(gait_name)

        # 初始化PPO算法
        self.alg = PPO(actor_critic, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env  # 设置每个环境的步数

        # 初始化存储和模型
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])

        # 初始化总时间步数、总时间、当前学习迭代次数和最后记录迭代次数
        self.tot_time_steps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = 0

        # 重置环境
        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500,
              eval_expert=False):
        """
        执行学习过程。

        在指定的学习迭代次数内，通过与环境交互并更新算法模型来进行学习。支持在随机的初始episode长度下开始学习，
        并在特定频率下进行评估和课程结构的保存。

        Methods:
            - torch.randint_like: 生成与给定张量形状相同的随机整数张量。
            - torch.cat: 沿着指定维度连接张量。
            - torch.zeros: 返回一个填充了标量值 0 的张量。
            - torch.to: 将张量复制到指定的设备上。
            - torch.jit.script: 将一个模块或函数编译成TorchScript，这是一种可以优化执行的中间表示形式。
            - os.makedirs: 递归创建目录。
            - copy.deepcopy: 创建对象的深拷贝。

        Attributes:
            num_learning_iterations (int): 学习迭代的次数。
            init_at_random_ep_len (bool): 是否在随机的初始episode长度下开始学习。
            eval_freq (int): 评估频率。
            curriculum_dump_freq (int): 课程结构保存频率。
            eval_expert (bool): 是否在评估时使用专家策略。

        Returns:
            None

        Examples:
            >>> agent.learn(1000, init_at_random_ep_len=True, eval_freq=50, curriculum_dump_freq=200, eval_expert=True)

        Note:
            - 该方法假设已经有一个名为`alg`的算法实例和一个名为`env`的环境实例与之关联。
            - 使用`ml_logger`库来记录学习过程中的各种指标和状态。
        """
        from ml_logger import logger  # 导入日志记录器

        # 确保日志记录器的前缀已设置，以避免覆盖整个仪器服务器的数据
        assert logger.prefix, "you will overwrite the entire instrument server"

        # 开始记录学习过程
        logger.start('start', 'epoch', 'episode', 'run', 'step')

        # 如果设置了在随机初始episode长度下开始，初始化环境的episode长度
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))  # 随机初始化环境的episode长度

        # 分割训练和测试环境
        num_train_envs = self.env.num_train_envs  # 获取训练环境的数量

        # 获取初始观察结果
        obs_dict = self.env.get_observations()  # 获取初始观察结果
        obs, privileged_obs, obs_history = (obs_dict["obs"],
                                            obs_dict["privileged_obs"], obs_dict["obs_history"])  # 解包观察结果
        # 将观察结果转移到指定的设备上
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)  # 将观察结果转移到指定设备
        # 切换到训练模式
        self.alg.actor_critic.train()  # 将模型切换到训练模式

        # 初始化奖励和长度缓冲区
        rew_buffer = deque(maxlen=100)  # 初始化奖励缓冲区
        len_buffer = deque(maxlen=100)  # 初始化长度缓冲区
        rew_buffer_eval = deque(maxlen=100)  # 初始化评估奖励缓冲区
        len_buffer_eval = deque(maxlen=100)  # 初始化评估长度缓冲区
        # 初始化当前奖励和episode长度
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 初始化当前奖励和
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 初始化当前episode长度

        # 设置总迭代次数
        tot_iter = self.current_learning_iteration + num_learning_iterations  # 计算总迭代次数
        for it in range(self.current_learning_iteration, tot_iter):  # 迭代学习过程
            start = time.time()  # 记录开始时间
            # 执行一次环境交互回合
            with torch.inference_mode():  # 开启推理模式，减少内存使用
                for i in range(self.num_steps_per_env):  # 对每个环境步骤进行迭代
                    # 为训练环境生成动作
                    actions_train = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],
                                                 obs_history[:num_train_envs])  # 生成训练环境的动作
                    # 根据是否评估专家来为评估环境生成动作
                    if eval_expert:
                        actions_eval = self.alg.actor_critic.act_teacher(obs_history[num_train_envs:],
                                                                         privileged_obs[
                                                                         num_train_envs:])  # 生成评估环境的动作（专家）
                    else:
                        actions_eval = self.alg.actor_critic.act_student(obs_history[num_train_envs:])  # 生成评估环境的动作（学生）
                    # 执行动作并获取结果
                    ret = self.env.step(torch.cat((actions_train, actions_eval), dim=0))  # 执行动作并获取结果
                    obs_dict, rewards, dones, infos = ret  # 解包结果
                    # 更新观察结果
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]  # 更新观察结果

                    # 将结果转移到指定的设备上
                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(
                        self.device)  # 转移结果到指定设备
                    # 处理环境步骤
                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)  # 处理环境步骤

                    # 如果有训练环节的信息，记录到日志
                    if 'train/episode' in infos:
                        with logger.Prefix(metrics="train/episode"):  # 设置日志前缀
                            logger.store_metrics(**infos['train/episode'])  # 记录训练环节信息

                    # 如果有评估环节的信息，记录到日志
                    if 'eval/episode' in infos:
                        with logger.Prefix(metrics="eval/episode"):  # 设置日志前缀
                            logger.store_metrics(**infos['eval/episode'])  # 记录评估环节信息

                    # 如果有课程信息，更新奖励和长度缓冲区
                    if 'curriculum' in infos:
                        cur_reward_sum += rewards  # 更新当前奖励和
                        cur_episode_length += 1  # 更新当前episode长度

                        new_ids = (dones > 0).nonzero(as_tuple=False)  # 获取完成episode的环境索引

                        new_ids_train = new_ids[new_ids < num_train_envs]  # 获取训练环境中完成episode的索引
                        rew_buffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())  # 更新训练环境奖励缓冲区
                        len_buffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())  # 更新训练环境长度缓冲区
                        cur_reward_sum[new_ids_train] = 0  # 重置训练环境当前奖励和
                        cur_episode_length[new_ids_train] = 0  # 重置训练环境当前episode长度

                        new_ids_eval = new_ids[new_ids >= num_train_envs]  # 获取评估环境中完成episode的索引
                        rew_buffer_eval.extend(cur_reward_sum[new_ids_eval].cpu().numpy().tolist())  # 更新评估环境奖励缓冲区
                        len_buffer_eval.extend(cur_episode_length[new_ids_eval].cpu().numpy().tolist())  # 更新评估环境长度缓冲区
                        cur_reward_sum[new_ids_eval] = 0  # 重置评估环境当前奖励和
                        cur_episode_length[new_ids_eval] = 0  # 重置评估环境当前episode长度

                    # 如果有课程分布信息，记录到日志
                    if 'curriculum/distribution' in infos:
                        distribution = infos['curriculum/distribution']  # 获取课程分布信息

            stop = time.time()  # 记录停止时间
            collection_time = stop - start  # 计算环境交互时间

            # 执行学习步骤
            start = stop  # 更新开始时间
            self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])  # 计算回报

            # 如果达到课程结构保存频率，保存课程信息
            if it % curriculum_dump_freq == 0:
                logger.save_pkl({"iteration": it,
                                 **caches.slot_cache.get_summary(),
                                 **caches.dist_cache.get_summary()},
                                path=f"curriculum/info.pkl", append=True)  # 保存课程信息

                if 'curriculum/distribution' in infos:
                    logger.save_pkl({"iteration": it,
                                     "distribution": distribution},
                                    path=f"curriculum/distribution.pkl", append=True)  # 保存课程分布信息

            # 更新算法模型并记录学习时间
            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student = self.alg.update  # 更新模型并获取损失
            stop = time.time()  # 记录停止时间
            learn_time = stop - start  # 计算学习时间

            # 记录学习过程中的各种指标
            logger.store_metrics(
                # 记录总时间，此行代码被注释掉了，原意可能是用于记录从开始到现在的总学习时间
                # total_time=learn_time - collection_time,
                time_elapsed=logger.since('start'),  # 从开始到现在的时间
                time_iter=logger.split('epoch'),  # 当前epoch的时间
                adaptation_loss=mean_adaptation_module_loss,  # 适应模块的平均损失
                mean_value_loss=mean_value_loss,  # 值函数的平均损失
                mean_surrogate_loss=mean_surrogate_loss,  # 代理模型的平均损失
                mean_decoder_loss=mean_decoder_loss,  # 解码器的平均损失
                mean_decoder_loss_student=mean_decoder_loss_student,  # 学生解码器的平均损失
                mean_decoder_test_loss=mean_decoder_test_loss,  # 测试时解码器的平均损失
                mean_decoder_test_loss_student=mean_decoder_test_loss_student,  # 测试时学生解码器的平均损失
                mean_adaptation_module_test_loss=mean_adaptation_module_test_loss  # 测试时适应模块的平均损失
            )

            # 如果设置了保存视频间隔，记录视频
            if RunnerArgs.save_video_interval:
                self.log_video(it)  # 记录视频

            # 更新总时间步数
            self.tot_time_steps += self.num_steps_per_env * self.env.num_envs  # 更新总时间步数
            # 如果达到日志记录频率，记录日志
            if logger.every(RunnerArgs.log_freq, "iteration", start_on=1):
                # if it % Config.log_freq == 0:  # 此行代码被注释掉了，原意可能是另一种日志记录频率的判断方法
                logger.log_metrics_summary(key_values={"timesteps": self.tot_time_steps, "iterations": it})  # 记录日志摘要
                logger.job_running()  # 标记任务正在运行

            # 如果达到保存间隔，保存模型和其他信息
            if it % RunnerArgs.save_interval == 0:
                with logger.Sync():  # 同步日志记录器
                    # 保存演员-评论家模型的权重
                    logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
                    # 复制最新的权重作为最后的权重
                    logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

                    path = './tmp/legged_data'  # 临时数据路径

                    os.makedirs(path, exist_ok=True)  # 创建路径

                    adaptation_module_path = f'{path}/adaptation_module_latest.jit'  # 适应模块的路径
                    adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to(
                        'cpu')  # 深拷贝适应模块并移至CPU
                    traced_script_adaptation_module = torch.jit.script(adaptation_module)  # 将适应模块转换为Torch脚本
                    traced_script_adaptation_module.save(adaptation_module_path)  # 保存适应模块的Torch脚本

                    body_path = f'{path}/body_latest.jit'  # 身体模型的路径
                    body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')  # 深拷贝身体模型并移至CPU
                    traced_script_body_module = torch.jit.script(body_model)  # 将身体模型转换为Torch脚本
                    traced_script_body_module.save(body_path)  # 保存身体模型的Torch脚本

                    # 上传适应模块和身体模型的文件
                    logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
                    logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)

            # 更新当前学习迭代次数
            self.current_learning_iteration += num_learning_iterations

        # 学习结束后，保存模型和其他信息
        with logger.Sync():
            # 保存模型的状态字典
            logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
            # 复制最新的模型权重文件
            logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")

            # 创建存储路径
            path = './tmp/legged_data'
            os.makedirs(path, exist_ok=True)

            # 保存适应模块
            adaptation_module_path = f'{path}/adaptation_module_latest.jit'
            adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
            traced_script_adaptation_module = torch.jit.script(adaptation_module)
            traced_script_adaptation_module.save(adaptation_module_path)

            # 保存身体模型
            body_path = f'{path}/body_latest.jit'
            body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
            traced_script_body_module = torch.jit.script(body_model)
            traced_script_body_module.save(body_path)

            # 上传保存的文件到日志服务器
            logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
            logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)

    def log_video(self, it):
        """
        记录并保存视频。

        根据迭代次数和设置的视频保存间隔，决定是否开始录制视频。支持同时记录评估环境的视频。
        如果有完整的帧数据，则暂停录制并保存视频文件。

        Methods:
            - start_recording: 开始录制训练环境的视频。
            - start_recording_eval: 开始录制评估环境的视频。
            - get_complete_frames: 获取训练环境的完整帧数据。
            - pause_recording: 暂停录制训练环境的视频。
            - get_complete_frames_eval: 获取评估环境的完整帧数据。
            - pause_recording_eval: 暂停录制评估环境的视频。
            - save_video: 保存视频文件。

        Attributes:
            it (int): 当前的迭代次数。

        Returns:
            None

        Examples:
            >>> agent.log_video(1000)

        Note:
            - 该方法假设已经有一个名为`env`的环境实例与之关联，并且环境实例具有录制视频的能力。
            - 使用`ml_logger`库来保存视频文件。
        """
        # 检查是否达到录制视频的间隔
        if it - self.last_recording_it >= RunnerArgs.save_video_interval:
            self.env.start_recording()  # 开始录制训练环境的视频
            if self.env.num_eval_envs > 0:
                self.env.start_recording_eval()  # 如果有评估环境，也开始录制评估环境的视频
            print("START RECORDING")  # 打印开始录制的信息
            self.last_recording_it = it  # 更新最后一次录制视频的迭代次数

        frames = self.env.get_complete_frames()  # 获取训练环境的完整帧数据
        if len(frames) > 0:
            self.env.pause_recording()  # 暂停录制训练环境的视频
            print("LOGGING VIDEO")  # 打印正在记录视频的信息
            logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)  # 保存训练环境的视频文件

        # 如果有评估环境，重复上述过程记录评估环境的视频
        if self.env.num_eval_envs > 0:
            frames = self.env.get_complete_frames_eval()  # 获取评估环境的完整帧数据
            if len(frames) > 0:
                self.env.pause_recording_eval()  # 暂停录制评估环境的视频
                print("LOGGING EVAL VIDEO")  # 打印正在记录评估环境视频的信息
                logger.save_video(frames, f"videos/{it:05d}_eval.mp4", fps=1 / self.env.dt)  # 保存评估环境的视频文件

    def get_inference_policy(self, device=None):
        """
        获取用于推理的策略函数。

        切换算法的actor_critic模型到评估模式，并将其移动到指定的设备上（如果提供了设备）。返回用于推理的动作生成函数。

        Methods:
            - eval: 将模型切换到评估模式。
            - to: 将模型移动到指定的设备上。
            - act_inference: 获取用于推理的动作生成函数。

        Attributes:
            device (str, optional): 指定模型应该被移动到的设备。默认为None，表示不移动模型。

        Returns:
            function: 用于推理的动作生成函数。

        Examples:
            >>> policy = agent.get_inference_policy(device='cuda')
            >>> action = policy(observation)

        Note:
            - 该方法假设已经有一个名为`alg`的算法实例与之关联。
            - 在调用此方法之前，确保已经正确设置了环境和算法的状态。
        """
        # 切换到评估模式（例如，对dropout等进行处理）
        self.alg.actor_critic.eval()
        # 如果指定了设备，将模型移动到该设备上
        if device is not None:
            self.alg.actor_critic.to(device)
        # 返回用于推理的动作生成函数
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        """
        获取专家策略函数。

        此方法将算法模型切换到评估模式，并可选择将其移动到指定的设备上。返回的是专家策略函数，可用于生成专家动作。

        Methods:
            - eval: 切换到评估模式，通常用于禁用特定于训练的行为如Dropout。
            - to: 将模型及其参数移动到指定的设备上。

        Attributes:
            device (str, optional): 目标设备的名称。默认为None，表示不改变当前设备。

        Returns:
            function: 专家策略函数，可用于在给定观察下生成动作。

        Examples:
            >>> agent = YourAgentClass()  # 假设已有一个代理类的实例
            >>> expert_policy = agent.get_expert_policy(device='cuda')  # 获取专家策略并移动到CUDA设备上
            >>> action = expert_policy(observation)  # 使用专家策略生成动作

        Note:
            - 确保在调用此方法前，已经有一个名为`alg`的算法实例，并且其包含`actor_critic`属性。
            - `actor_critic`属性应该有一个名为`act_expert`的方法，用于执行专家策略。
        """
        # 切换到评估模式
        self.alg.actor_critic.eval()
        # 如果指定了设备，则将模型移动到该设备上
        if device is not None:
            self.alg.actor_critic.to(device)
        # 返回专家策略函数
        return self.alg.actor_critic.act_expert
