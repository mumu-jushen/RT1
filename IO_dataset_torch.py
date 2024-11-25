import os
import json
import glob
from PIL import Image
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm as tqdm


def build_dataset(
    data_path,
    time_sequence_length=6,
    predicting_next_ts=True,
    num_train_episode=200,
    num_val_episode=100,
    cam_view=["front"],
    language_embedding_size=512,
):
    """
    This function is for building the training and validation dataset

    Parameters:
    - data_path(str): locates the path where the dataset is stored
            the dataset path should have the following file structures:
                - [robotname]_[taskname]
                    - [cam_view_0]
                        - data_000
                            - rgb # where those image stored
                                - image_001.png
                                - image_002.png
                                - ...
                            - results.csv # robot actions stored
                            - results_raw.csv # joint and target object position stored
                        - data_001
                        - ...
                    - [cam_view_1]
                        - data_000
                        - data_001
                        - ...
                    - ...
    - time_sequence_length(int) : number of history length input for RT-1 model,
        6 means current frame image and past 5 frames of images will be packed and input to RT-1
    - predicting_next_ts(bool) : in our dataset's results.csv and results_raw.csv, we stored current frame's action and joint status.
        if we want to predict next frame's action, this option needs to be True and result in the 1 step offset reading on csv files
        this differs between the samplings method of different dataset.
    - num_train_episode(int) : specifies numbers of training episodes
    - num_train_episode(int) : specifies numbers of validation episodes
    - cam_view(list of strs) : camera views used for training.

    Returns:
    - train_dataset(torch.utils.data.Dataset)
    - val_dataset(torch.utils.data.Dataset)
    """

    with open(os.path.join(data_path, cam_view[0], "dataset_info.json"), "r") as f:
        info = json.load(f)
    episode_length = info["episode_length"]
    episode_dirs = sorted(glob.glob(data_path + "/" + cam_view[0] + "/*/"))
    assert len(episode_dirs) == len(
        episode_length
    ), "length of episode directories and episode length not equal, check dataset's dataset_info.json"
    perm_indice = torch.randperm(len(episode_dirs)).tolist()
    dirs_lengths = dict(
        episode_dirs=np.array(episode_dirs)[perm_indice],
        episode_length=np.array(episode_length)[perm_indice],
    )
    train_episode_dirs = dirs_lengths["episode_dirs"][:num_train_episode]
    train_episode_length = dirs_lengths["episode_length"][:num_train_episode]
    val_episode_dirs = dirs_lengths["episode_dirs"][
        num_train_episode : num_train_episode + num_val_episode
    ]
    val_episode_length = dirs_lengths["episode_length"][
        num_train_episode : num_train_episode + num_val_episode
    ]

    train_dataset = IODataset(
        episode_dirs=train_episode_dirs,
        episode_length=train_episode_length,
        time_sequence_length=time_sequence_length,
        predicting_next_ts=predicting_next_ts,
        cam_view=cam_view,
        language_embedding_size=language_embedding_size,
    )
    val_dataset = IODataset(
        episode_dirs=val_episode_dirs,
        episode_length=val_episode_length,
        time_sequence_length=time_sequence_length,
        predicting_next_ts=predicting_next_ts,
        cam_view=cam_view,
        language_embedding_size=language_embedding_size,
    )
    return train_dataset, val_dataset


class IODataset(Dataset):
    def __init__(
        self,
        episode_dirs,
        episode_length,
        time_sequence_length=6,
        predicting_next_ts=True,
        cam_view=["front"],
        robot_dof=9,
        language_embedding_size=512,
    ):
        # 初始化数据集的一些基本参数
        self._cam_view = cam_view  # 相机视角列表
        self.predicting_next_ts = predicting_next_ts  # 是否预测下一时间步的数据
        self._time_sequence_length = time_sequence_length  # 时间序列长度
        self._episode_length = episode_length  # 每个episode的长度列表
        self.querys = self.generate_history_steps(episode_length)  # 生成历史时间步的索引
        self._episode_dirs = episode_dirs  # 存储每个episode数据的目录
        self.keys_image = self.generate_fn_lists(self._episode_dirs)  # 生成所有图像文件的列表
        self.values, self.num_zero_history_list = self.organize_file_names()  # 组织数据文件名称和相关信息
        self._robot_dof = robot_dof  # 机器人的自由度数量
        self._language_embedding_size = language_embedding_size  # 语言嵌入的维度大小

    def generate_fn_lists(self, episode_dirs):
        """
        生成数据集中所有图像的路径列表
        参数:
        - episode_dirs(list of strs): 图像存储的目录列表, 格式为:
            - [robotname]_[taskname]
                - [cam_view_0]
                    - data_000
                    - data_001
                    - data_002
                    - ...
        返回值:
        - keys(list of strs): 所有图像文件名的列表
        """
        keys = []
        for ed in episode_dirs:
            # 获取目录下所有PNG图像文件的路径并排序
            image_files = sorted(glob.glob(f"{ed}rgb/*.png"))
            keys.append(image_files)  # 将图像文件列表添加到keys列表中
        return keys

    def generate_history_steps(self, episode_length):
        """
        生成当前帧及历史帧的时间步索引
        参数:
        - episode_length(list of int): 每个episode的长度列表
        返回值:
        - querys(list of tensors): 每个数据点的历史时间步索引
        """
        querys = []
        for el in episode_length:
            q = torch.cat(
                (
                    [
                        # 为每个时间步生成历史索引，负值表示需要填充
                        torch.arange(el)[:, None] - i
                        for i in range(self._time_sequence_length)
                    ]
                ),
                dim=1,
            )
            q[q < 0] = -1  # 将小于0的索引替换为-1，表示需要填充
            querys.append(q.flip(1))  # 翻转时间维度，确保时间步顺序
        return querys

    def organize_file_names(self):
        """
        组织每个数据点的相关信息，包括需要填充的零的数量、episode目录、图像文件名等
        返回值:
        - values(list): 每个数据点的详细信息字典列表，包括：
            - num_zero_history: 初始时间步缺少历史数据时填充的零的数量
            - episode_dir: 该数据点对应的episode目录
            - img_fns: 该数据点对应的图像文件名列表
            - query_index: 该数据点在episode中的索引
            - episode_length: 该数据点所在的episode的总长度
        - num_zero_history_list(list): 每个数据点的填充零的数量列表
        """
        values = []
        num_zero_history_list = []
        for i, (query, key_img, ed) in enumerate(
            zip(self.querys, self.keys_image, self._episode_dirs)
        ):
            for q in query:
                img_fns = []
                for img_idx in q:
                    # 将每个历史时间步的图像文件名添加到列表中，如果时间步小于0则添加None
                    img_fns.append(key_img[img_idx] if img_idx >= 0 else None)
                num_zero_history = (q < 0).sum()  # 计算需要填充的零的数量
                num_zero_history_list.append(int(num_zero_history))
                values.append(
                    dict(
                        num_zero_history=num_zero_history,
                        episode_dir=ed,
                        img_fns=img_fns,
                        query_index=q,
                        episode_length=self._episode_length[i],
                    )
                )
        return values, num_zero_history_list

    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.values)

    def __getitem__(self, idx):
        # 获取指定索引的数据点
        value = self.values[idx]
        img = self.get_image(value["img_fns"])  # 获取图像数据
        lang = self.get_language_instruction()  # 获取语言嵌入
        ee_pos_cmd, ee_rot_cmd, gripper_cmd, joint, tar_obj_pose = self.get_ee_data(
            value["episode_dir"], value["query_index"], value["num_zero_history"]
        )  # 获取末端执行器数据、关节数据和目标物体位姿
        terminate_episode = self.get_episode_status(
            value["episode_length"], value["query_index"], value["num_zero_history"]
        )  # 获取当前帧的episode状态
        sample_obs = {
            "image": img.float().permute(0, 3, 1, 2),  # 将图像数据的通道维度移动到第2维，以适应模型的输入格式
            "natural_language_embedding": torch.tensor(lang).float(),  # 语言嵌入
            "joint_position": torch.tensor(joint).float(),  # 机器人关节位置
            "tar_obj_pose": torch.tensor(tar_obj_pose).float(),  # 目标物体的位姿
        }
        sample_action = {
            "world_vector": torch.tensor(ee_pos_cmd).float(),  # 末端执行器位置命令
            "rotation_delta": torch.tensor(ee_rot_cmd).float(),  # 末端执行器旋转命令
            "gripper_closedness_action": torch.tensor(gripper_cmd).float(),  # 夹爪开闭状态
            "terminate_episode": torch.tensor(terminate_episode.argmax(-1)).long(),  # 终止episode的标志
        }

        return sample_obs, sample_action

    def get_image(self, img_fns):
        """
        获取当前帧和历史帧的图像数据
        参数:
        - img_fns(list of strs): 图像文件名列表
        返回值:
        - torch.Tensor: 图像数据张量，形状为[time_sequence_length, height, width, channels]
        """
        imgs = []
        for img_fn in img_fns:
            img_multi_view = []
            for c_v in self._cam_view:
                img_multi_view.append(
                    np.array(Image.open(img_fn.replace(self._cam_view[0], c_v)))
                    if img_fn != None
                    else np.zeros_like(Image.open(img_fns[-1]))
                )
            img = np.concatenate(img_multi_view, axis=0)  # 将多个视角的图像在通道维度拼接
            imgs.append(torch.from_numpy(img[:, :, :3]))  # 提取前3个通道（RGB）并转换为张量
        return torch.stack(imgs, dim=0) / 255.0  # 将图像标准化到0-1之间，并沿时间维度堆叠

    def get_ee_data(self, episode_dir, query_index, pad_step_num):
        """
        读取末端执行器的相关数据，包括机器人关节状态和目标物体的位姿
        参数:
        - episode_dir(str): 存储结果文件的目录
        - query_index(tensor): 数据点的索引，填充的零对应索引-1
        - pad_step_num(int): 填充的零的时间步数量
        返回值:
        - ee_pos_cmd(np.array): 末端执行器的位置信息（x, y, z）
        - ee_rot_cmd(np.array): 末端执行器的旋转信息（rx, ry, rz）
        - gripper_cmd(np.array): 机器人夹爪的开闭状态
        - joint(np.array): 机器人的关节状态
        - tar_obj_pose: 目标物体的位置信息（x, y, z, rx, ry, rz, rw）
        """
        start_idx = query_index[(query_index > -1).nonzero()[0, 0]]  # 获取起始索引
        end_idx = query_index[-1]  # 获取结束索引
        visual_data_filename = f"{episode_dir}result.csv"
        raw_data = pd.read_csv(visual_data_filename)  # 读取视觉结果数据
        visual_data_filename_raw = f"{episode_dir}result_raw.csv"
        raw_raw_data = pd.read_csv(visual_data_filename_raw)  # 读取原始关节数据
        if self.predicting_next_ts:
            """
            如果预测的是下一个时间步的数据，则将第一行移动到最后一行
            """
            first_row = raw_data.iloc[0]
            raw_data = raw_data.iloc[1:]
            raw_data = pd.concat([raw_data, first_row.to_frame().T], ignore_index=True)
            first_row = raw_raw_data.iloc[0]
            raw_raw_data = raw_raw_data.iloc[1:]
            raw_raw_data = pd.concat(
                [raw_raw_data, first_row.to_frame().T], ignore_index=True
            )
        # 为末端执行器的位置、旋转、夹爪、关节状态和目标物体位姿创建零填充
        ee_pos_cmd = np.zeros([pad_step_num, 3])  # 位置命令有3个维度 [x, y, z]
        ee_rot_cmd = np.zeros([pad_step_num, 3])  # 旋转命令有3个维度 [rx, ry, rz]
        gripper_cmd = np.zeros([pad_step_num, 1])  # 夹爪有1个维度（开或关）
        joint = np.zeros([pad_step_num, 9])  # 机器人的关节状态有9个自由度
        tar_obj_pose = np.zeros([pad_step_num, 7])  # 目标物体的位姿信息有7个维度 [x,y,z,rx,ry,rz,w]

        # 追加实际的数据到填充的数组中
        ee_pos_cmd = np.vstack(
            (
                ee_pos_cmd,
                raw_data.loc[
                    start_idx:end_idx,
                    [f"ee_command_position_{ax}" for ax in ["x", "y", "z"]],
                ].to_numpy(),
            )
        )
        ee_rot_cmd = np.vstack(
            (
                ee_rot_cmd,
                raw_data.loc[
                    start_idx:end_idx,
                    [f"ee_command_rotation_{ax}" for ax in ["x", "y", "z"]],
                ].to_numpy(),
            )
        )
        joint = np.vstack(
            (
                joint,
                raw_raw_data.loc[
                    start_idx:end_idx,
                    [f"joint_{str(ax)}" for ax in range(self._robot_dof)],
                ].to_numpy(),
            )
        )
        tar_obj_pose = np.vstack(
            (
                tar_obj_pose,
                raw_raw_data.loc[
                    start_idx:end_idx,
                    [
                        f"tar_obj_pose_{ax}"
                        for ax in ["x", "y", "z", "rx", "ry", "rz", "rw"]
                    ],
                ].to_numpy(),
            )
        )
        # 读取夹爪状态并添加到填充数组中
        gripper_data = (
            raw_data.loc[start_idx:end_idx, "gripper_closedness_commanded"]
            .to_numpy()
            .reshape(-1, 1)
        )
        gripper_cmd = np.vstack((gripper_cmd, gripper_data))
        return ee_pos_cmd, ee_rot_cmd, gripper_cmd, joint, tar_obj_pose

    def get_language_instruction(self):
        """
        获取语言指令的嵌入，单任务模型中语言嵌入是一个常数
        如果是多任务模型，可以修改为实际的语言指令嵌入
        """
        return np.zeros([self._time_sequence_length, self._language_embedding_size])

    def get_episode_status(self, episode_length, query_index, pad_step_num):
        """
        确定当前帧及历史帧在episode中的位置：开始、中间或结束
        参数:
        - episode_length(int): 当前episode的长度
        - query_index(tensor): 数据点的索引，填充的零对应索引-1
        - pad_step_num(int): 填充的零的时间步数量
        返回值:
        - episode_status(np.array): 指定历史中每帧的位置状态（开始、中间、结束）
        """
        start_idx = query_index[(query_index > -1).nonzero()[0, 0]]  # 获取起始索引
        end_idx = query_index[-1]  # 获取结束索引
        episode_status = np.zeros([pad_step_num, 4], dtype=np.int32)  # 状态为4维 [start, middle, end, padding]
        episode_status[:, -1] = 1  # 填充的时间步设置为1
        for i in range(start_idx, end_idx + 1):
            status = np.array(
                [i == 0, i not in [0, episode_length - 2], i == episode_length - 2, 0],
                dtype=np.int32,
            )
            episode_status = np.vstack((episode_status, status))
        if pad_step_num > 0:
            episode_status[pad_step_num] = np.array([1, 0, 0, 0])  # 第一个填充时间步设置为start
        return episode_status

def load_config_from_json(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)
    return config


if __name__ == "__main__":
    args = load_config_from_json("train_config.json")
    dataset, _ = build_dataset(
        data_path=args["data_path"],
        time_sequence_length=args["time_sequence_length"],
        predicting_next_ts=args["predicting_next_ts"],
        num_train_episode=args["num_train_episode"],
        num_val_episode=args["num_val_episode"],
        cam_view=args["cam_view"],
        language_embedding_size=args["network_configs"]["language_embedding_size"],
    )
    # dataset = dataset[:100]

    wv_x = []
    wv_y = []
    wv_z = []
    rd_x = []
    rd_y = []
    rd_z = []
    from maruya24_rt1.tokenizers import action_tokenizer
    from gym import spaces
    from collections import OrderedDict
    import matplotlib.pyplot as plt

    output_tensor_space = spaces.Dict(
        OrderedDict(
            [
                ("terminate_episode", spaces.Discrete(4)),
                (
                    "world_vector",
                    spaces.Box(low=-0.025, high=0.025, shape=(3,), dtype=np.float32),
                ),
                (
                    "rotation_delta",
                    spaces.Box(
                        low=-np.pi / 20,
                        high=np.pi / 20,
                        shape=(3,),
                        dtype=np.float32,
                    ),
                ),
                (
                    "gripper_closedness_action",
                    spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                ),
            ]
        )
    )
    at = action_tokenizer.RT1ActionTokenizer(
        output_tensor_space, vocab_size=256  # action space
    )
    dataloader = DataLoader(dataset, batch_size=64, num_workers=64)
    for i, batch in tqdm(enumerate(dataloader), total=len(dataset) // 64):
        batch = at.tokenize(batch[1])
        for i in range(batch.size(0)):
            wv_x.append(int(batch[i, -1, 1]))
            wv_y.append(int(batch[i, -1, 2]))
            wv_z.append(int(batch[i, -1, 3]))
            rd_x.append(int(batch[i, -1, 4]))
            rd_y.append(int(batch[i, -1, 5]))
            rd_z.append(int(batch[i, -1, 6]))
        # print(batch)
    plt.subplot(2, 3, 1)
    plt.title("world_vector_x")
    plt.hist(wv_x, bins=256, range=(0, 256))
    plt.xlim(0, 256)
    plt.subplot(2, 3, 2)
    plt.title("world_vector_y")
    plt.hist(wv_y, bins=256, range=(0, 256))
    plt.xlim(0, 256)
    plt.subplot(2, 3, 3)
    plt.title("world_vector_z")
    plt.hist(wv_z, bins=256, range=(0, 256))
    plt.xlim(0, 256)
    plt.subplot(2, 3, 4)
    plt.title("rotation_delta_x")
    plt.hist(rd_x, bins=256, range=(0, 256))
    plt.xlim(0, 256)
    plt.subplot(2, 3, 5)
    plt.title("rotation_delta_y")
    plt.hist(rd_y, bins=256, range=(0, 256))
    plt.xlim(0, 256)
    plt.subplot(2, 3, 6)
    plt.title("rotation_delta_z")
    plt.hist(rd_z, bins=256, range=(0, 256))
    plt.xlim(0, 256)
    plt.show()
