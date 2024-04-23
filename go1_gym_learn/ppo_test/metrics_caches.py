from collections import defaultdict

from ml_logger import logger
import numpy as np
import torch


class DistCache:
    def __init__(self):
        """
        初始化一个分布式缓存对象。

        该对象用于存储和更新键值对数据，特别是用于记录和计算平均值。

        Attributes:
            cache (defaultdict): 一个默认字典，用于存储键值对和它们的计数。
        """
        self.cache = defaultdict(lambda: 0)  # 初始化默认字典，未找到的键默认值为0

    def log(self, **key_vals):
        """
        记录或更新键值对数据。

        对于每个键值对，该方法更新其平均值和计数。如果键是新的，则初始化其值和计数。

        Args:
            **key_vals: 可变数量的键值对参数，键为字符串，值为数值。

        Examples:
            >>> cache = DistCache()
            >>> cache.log(loss=0.5, accuracy=0.8)
            >>> cache.log(loss=0.3, accuracy=0.85)
            >>> print(cache.get_summary())
        """
        for k, v in key_vals.items():  # 遍历所有键值对
            count = self.cache[k + '@counts'] + 1  # 更新或初始化计数
            self.cache[k + '@counts'] = count  # 保存新的计数
            self.cache[k] = v + (count - 1) * self.cache[k]  # 更新累计值
            self.cache[k] /= count  # 计算新的平均值

    def get_summary(self):
        """
        获取并清除所有记录的平均值。

        返回一个包含所有键和它们平均值的字典，并清除缓存。

        Returns:
            dict: 包含所有键和它们平均值的字典。

        Examples:
            >>> cache = DistCache()
            >>> cache.log(loss=0.5, accuracy=0.8)
            >>> summary = cache.get_summary()
            >>> print(summary)
        """
        ret = {
            k: v
            for k, v in self.cache.items()
            if not k.endswith("@counts")  # 选择不以"@counts"结尾的键，即原始数据键
        }
        self.cache.clear()  # 清除缓存
        return ret  # 返回平均值字典


if __name__ == '__main__':
    """
    当模块被直接运行时，以下代码将执行。

    该代码段创建一个DistCache实例，并使用两组数据（线速度和角速度）进行测试。
    它首先记录一组全为1的线速度和全为0的角速度，然后记录一组全为0的线速度和角速度。
    最后，它打印出这些值的平均值。
    """
    cl = DistCache()  # 创建DistCache实例
    lin_vel = np.ones((11, 11))  # 创建一个全为1的线速度矩阵
    ang_vel = np.zeros((5, 5))  # 创建一个全为0的角速度矩阵
    cl.log(lin_vel=lin_vel, ang_vel=ang_vel)  # 记录第一组数据
    lin_vel = np.zeros((11, 11))  # 创建一个全为0的线速度矩阵
    ang_vel = np.zeros((5, 5))  # 创建另一个全为0的角速度矩阵
    cl.log(lin_vel=lin_vel, ang_vel=ang_vel)  # 记录第二组数据
    print(cl.get_summary())  # 打印平均值


class SlotCache:
    def __init__(self, n):
        """
        初始化一个具有指定槽位数量的缓存对象。

        该对象用于存储和更新键值对数据，特别是用于记录和计算每个槽位的平均值。

        Args:
            n (int): 缓存中槽位的数量。

        Attributes:
            n (int): 缓存中槽位的数量。
            cache (defaultdict): 一个默认字典，用于存储键值对和它们的计数，每个键对应一个长度为n的numpy数组。
        """
        self.n = n  # 缓存槽位的数量
        self.cache = defaultdict(lambda: np.zeros([n]))  # 初始化默认字典，未找到的键默认值为长度为n的零数组

    def log(self, slots=None, **key_vals):
        """
        记录或更新键值对数据。

        对于每个键值对，该方法更新其在指定槽位的平均值和计数。如果键是新的，则初始化其值和计数。

        Args:
            slots (list, optional): 指定需要更新的槽位的id列表。默认为None，表示更新所有槽位。
            **key_vals: 可变数量的键值对参数，键为字符串，值为数值。

        Examples:
            >>> cache = SlotCache(n=10)
            >>> cache.log(slots=[0, 1, 2], lin_vel=1, ang_vel=0)
            >>> cache.log(slots=[3, 4, 5], lin_vel=0, ang_vel=1)
            >>> print(cache.get_summary())
        """
        if slots is None:
            slots = range(self.n)  # 如果未指定槽位，则默认更新所有槽位

        for k, v in key_vals.items():  # 遍历所有键值对
            counts = self.cache[k + '@counts'][slots] + 1  # 更新或初始化计数
            self.cache[k + '@counts'][slots] = counts  # 保存新的计数
            self.cache[k][slots] = v + (counts - 1) * self.cache[k][slots]  # 更新累计值
            self.cache[k][slots] /= counts  # 计算新的平均值

    def get_summary(self):
        """
        获取并清除所有记录的平均值。

        返回一个包含所有键和它们平均值的字典，并清除缓存。

        Returns:
            dict: 包含所有键和它们平均值的字典。

        Examples:
            >>> cache = SlotCache(n=10)
            >>> cache.log(slots=[0, 1, 2], lin_vel=1, ang_vel=0)
            >>> summary = cache.get_summary()
            >>> print(summary)
        """
        ret = {
            k: v
            for k, v in self.cache.items()
            if not k.endswith("@counts")  # 选择不以"@counts"结尾的键，即原始数据键
        }
        self.cache.clear()  # 清除缓存
        return ret  # 返回平均值字典


if __name__ == '__main__':
    """
    当模块被直接运行时，以下代码将执行。

    该代码段创建一个SlotCache实例，并使用两组数据（线速度和角速度）进行测试。
    它首先记录一组特定槽位的线速度和角速度，然后记录一组全槽位的线速度为1。
    """
    cl = SlotCache(100)  # 创建一个具有100个槽位的SlotCache实例
    reset_env_ids = [2, 5, 6]  # 指定需要更新的槽位id
    lin_vel = [0.1, 0.5, 0.8]  # 线速度值列表
    ang_vel = [0.4, -0.4, 0.2]  # 角速度值列表
    cl.log(reset_env_ids, lin_vel=lin_vel, ang_vel=ang_vel)  # 记录特定
