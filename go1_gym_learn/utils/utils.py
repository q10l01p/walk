# License: see [LICENSE, LICENSES/rsl_rl/LICENSE]
import torch


def split_and_pad_trajectories(tensor, dones):
    """
    将张量按完成标志分割成轨迹，并填充至最长轨迹长度，同时返回对应的有效部分掩码。

    此函数接受一个包含多个环境轨迹的张量和完成标志，将其分割成单独的轨迹，
    并将每条轨迹填充至与最长轨迹相同的长度。返回填充后的轨迹和一个掩码张量，
    掩码张量中的True表示原始轨迹的有效部分。

    Args:
        tensor (torch.Tensor): 输入张量，维度为[时间, 环境数量, 附加维度]。
        dones (torch.Tensor): 完成标志张量，维度为[时间, 环境数量, 1]。

    Returns:
        tuple: 包含填充后的轨迹和对应的有效部分掩码的元组。

    Examples:
        >>> tensor = torch.tensor([...])
        >>> dones = torch.tensor([...])
        >>> padded_trajectories, masks = split_and_pad_trajectories(tensor, dones)

    Note:
        - 输入的张量假定维度顺序为[时间, 环境数量, 附加维度]。
    """
    # 克隆完成标志并将最后一个标志设置为1，确保每个轨迹都能被正确结束
    dones = dones.clone()
    dones[-1] = 1
    # 调整张量维度顺序为(环境数量, 时间, ...)
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # 通过计算连续非完成元素的数量来获取轨迹长度
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # 提取单独的轨迹
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    # 对轨迹进行填充，使其长度一致
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)

    # 生成轨迹有效部分的掩码
    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """
    执行split_and_pad_trajectories()的逆操作。

    此函数接受填充后的轨迹和对应的有效部分掩码，将填充的轨迹还原为原始轨迹的形状。
    它通过应用掩码来选择有效部分，并将这些部分重新组织成原始轨迹的形状。

    Args:
        trajectories (torch.Tensor): 填充后的轨迹张量。
        masks (torch.Tensor): 对应的有效部分掩码张量。

    Returns:
        torch.Tensor: 还原后的轨迹张量。

    Examples:
        >>> padded_trajectories = torch.tensor([...])
        >>> masks = torch.tensor([...])
        >>> trajectories = unpad_trajectories(padded_trajectories, masks)

    Note:
        - 假定输入的轨迹张量维度顺序为[时间, 环境数量, 附加维度]。
    """
    # 在应用掩码之前和之后进行转置，以便正确地重塑张量
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)
