# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

from typing import Tuple

import numpy as np
import torch
from isaacgym.torch_utils import quat_apply, normalize
from torch import Tensor


def quat_apply_yaw(quat, vec):
    """
    使用指定的四元数旋转向量，但仅应用偏航（yaw）分量。

    该函数首先将输入的四元数复制并重置其roll和pitch分量（即四元数的前两个分量）为0，
    仅保留偏航（yaw）分量，然后对修改后的四元数进行归一化处理，最后使用这个仅包含偏航分量的四元数
    来旋转输入向量。

    Attributes:
        quat (torch.Tensor): 输入的四元数，形状为 (N, 4)，其中N是四元数的数量。
        vec (torch.Tensor): 需要被旋转的向量，形状为 (N, 3)。

    Returns:
        torch.Tensor: 旋转后的向量，形状为 (N, 3)。

    Examples:
        >>> quat = torch.tensor([[0., 0., 0., 1.]])
        >>> vec = torch.tensor([[1., 0., 0.]])
        >>> rotated_vec = quat_apply_yaw(quat, vec)
        >>> print(rotated_vec)

    Note:
        - 该函数假设输入的四元数和向量都是批处理形式，即第一维是批次维度。
        - 需要先导入或定义`normalize`和`quat_apply`函数，分别用于四元数的归一化和应用四元数旋转向量。
    """
    quat_yaw = quat.clone().view(-1, 4)  # 复制输入的四元数，并确保其形状为 (N, 4)
    quat_yaw[:, :2] = 0.  # 将四元数的roll和pitch分量重置为0
    quat_yaw = normalize(quat_yaw)  # 对只包含偏航分量的四元数进行归一化
    return quat_apply(quat_yaw, vec)  # 使用归一化后的四元数旋转输入向量


def wrap_to_pi(angles):
    """
    将角度值规范化到[-π, π]区间内。

    此函数首先将输入角度通过取模操作限制在[0, 2π]区间内，然后对于那些大于π的角度，
    通过减去2π来将它们转换到[-π, π]的范围。这样处理后的角度值便于进行角度差的计算和比较。

    Attributes:
        angles (np.ndarray): 输入的角度值数组，可以是任意形状的。

    Returns:
        np.ndarray: 规范化到[-π, π]区间内的角度值数组，形状与输入数组相同。

    Examples:
        >>> angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, -np.pi/2, -np.pi, -3*np.pi/2])
        >>> wrap_to_pi(angles)
        array([ 0.        ,  1.57079633,  3.14159265, -1.57079633,  0.        ,
               -1.57079633, -3.14159265,  1.57079633])

    Note:
        - 输入角度值应以弧度为单位。
        - 此函数对于向量化操作非常有效，可以直接处理整个numpy数组而无需循环。
    """
    angles %= 2 * np.pi  # 将角度限制在[0, 2π]区间内
    angles -= 2 * np.pi * (angles > np.pi)  # 对大于π的角度减去2π，以规范化到[-π, π]
    return angles  # 返回规范化后的角度值


def torch_rand_sqrt_float(lower, upper, shape, device):
    """
    生成一个随机浮点数张量，其值的平方根分布在指定的范围内。

    此函数首先生成一个形状为 `shape` 的随机张量，其值在 [-1, 1] 范围内均匀分布。
    然后，对于张量中的每个值，如果值为负，则取其负数的平方根并保持符号不变；如果值为正，则直接取其平方根。
    接着，将结果映射到 [0, 1] 范围内，并根据指定的 `lower` 和 `upper` 边界调整其范围。
    最终返回这个调整后的张量。

    Attributes:
        lower (float): 生成的随机数的下界。
        upper (float): 生成的随机数的上界。
        shape (Tuple[int, int]): 生成的张量的形状。
        device (str): 生成的张量所在的设备。

    Returns:
        torch.Tensor: 调整后的随机浮点数张量，其平方根分布在 [lower, upper] 范围内。

    Examples:
        >>> torch_rand_sqrt_float(0.0, 1.0, (2, 3), 'cpu')
        tensor([[0.3457, 0.7891, 0.2345],
                [0.6578, 0.1234, 0.9876]])

    Note:
        - 该函数假设 `lower` < `upper`。
        - 返回的张量中的数值不是均匀分布的，而是其平方根在 [lower, upper] 范围内均匀分布。
    """
    # 在 [-1, 1] 范围内生成随机数张量
    r = 2 * torch.rand(*shape, device=device) - 1
    # 对负数取平方根并保持符号，对正数直接取平方根
    r = torch.where(r < 0., -torch.sqrt(-r), torch.sqrt(r))
    # 将结果映射到 [0, 1] 范围内
    r = (r + 1.) / 2.
    # 调整张量的范围到 [lower, upper]
    return (upper - lower) * r + lower


def get_scale_shift(range):
    """
    计算将输入范围映射到[-1, 1]区间所需的缩放和平移参数。

    此函数接受一个表示输入范围的元组（最小值，最大值），计算并返回将该范围线性映射到[-1, 1]区间所需的缩放因子和平移量。

    Attributes:
        range (Tuple[float, float]): 输入范围的最小值和最大值。

    Returns:
        Tuple[float, float]: 缩放因子和平移量。

    Examples:
        >>> scale, shift = get_scale_shift((0, 10))
        >>> print(f"Scale: {scale}, Shift: {shift}")

    Note:
        - 该函数假设输入范围是有效的，即最小值小于最大值。
        - 返回的缩放因子用于将输入值乘以该因子，平移量用于将结果加或减以该值，以实现映射。
    """
    # 计算缩放因子，使得范围映射到[-1, 1]
    scale = 2. / (range[1] - range[0])
    # 计算平移量，以便范围中心对齐到0
    shift = (range[1] + range[0]) / 2.
    return scale, shift  # 返回缩放因子和平移量
