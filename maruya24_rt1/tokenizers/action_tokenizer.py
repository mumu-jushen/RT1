# Subject to the terms and conditions of the Apache License, Version 2.0 that the original code follows, 
# I have retained the following copyright notice written on it.

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# You can find the original code from here[https://github.com/google-research/robotics_transformer].


from gym import spaces
import torch

from typing import Dict, Union
"""
Please define action space using gym.spaces.Dict.

As an example, if an action is:
terminate = [0, 1] # this is one hot vector.
world_vector = [0.9, 0.8, -0.3]
rotation_delta = [-0.1, 0.2, .6]
gripper_closedness = 0.9

then action space will look like
action_space = gym.spaces.Dict(
    {
        'terminate': gym.spaces.Discrete(2),
        'world_vector': gym.spaces.Box(low= -1.0, high= 1.0, shape=(3,), dtype=np.float32),
        'rotation_delta': gym.spaces.Box(low= -np.pi / 2  , high= np.pi / 2, shape=(3,), dtype=np.float32),
        'gripper_closedness_action': gym.spaces.Box(low= -1.0  , high= 1.0, shape=(1,), dtype=np.float32)
    }
)
or 
action_space = gym.spaces.Dict(
            OrderedDict([
                ('terminate', gym.spaces.Discrete(2)), 
                ('world_vector', gym.spaces.Box(low= -1.0, high= 1.0, shape=(3,), dtype=np.float32)),
                ('rotation_delta', gym.spaces.Box(low= -np.pi / 2., high= np.pi / 2., shape=(3,), dtype=np.float32)),
                ('gripper_closedness_action', gym.spaces.Box(low= -1.0  , high= 1.0, shape=(1,), dtype=np.float32))
                ])
        )
Please use OrderedDict if you want gym.spaces.Dict to keep order of actions.

This action_space is just information about each action.
These information are very convenient when interpreting, examining, cliping, and processing the action
because the action is dictionary with key names which are the same as the action_space.

action value will be look like
action = {
    'terminate': 1,
    'world_vector': [0.9, 0.8, -0.3],
    'rotation_delta': [-0.1, 0.2, .6],
    'gripper_closedness_action': [0.9]
}
Note that values are int and numpy 1-d arrays.
"""
class RT1ActionTokenizer():
    def __init__(self,
                 action_space: spaces.Dict,
                 vocab_size: int):
        """
        初始化一个RT1ActionTokenizer。

        参数:
        - action_space: OpenAI Gym中定义的动作空间字典，表示期望的动作。
        - vocab_size: 将动作离散化的桶的数量（词汇表大小）。
        """

        self._action_space = action_space  # 存储动作空间字典
        self._vocab_size = vocab_size  # 存储词汇表大小，用于离散化动作
        self._action_order = list(action_space.keys())  # 确定动作token化的顺序

        self._tokens_per_action = 0
        # 计算每个动作所需的token数量
        for action in self._action_order:
            # 如果动作是Discrete类型，则它是一个token
            if isinstance(self._action_space[action], spaces.Discrete):
                self._tokens_per_action += 1

            # 如果动作是Box类型，使用shape确定token数量
            elif isinstance(self._action_space[action], spaces.Box):
                action_shape = self._action_space[action].shape
                if len(action_shape) != 1:
                    raise ValueError(
                        f'仅支持单维度的动作形状，得到的是 {action_shape}')
                self._tokens_per_action += action_shape[0]
            else:
                raise ValueError('假设动作空间由gym.spaces.Discrete或gym.spaces.Box定义')

    @property
    def tokens_per_action(self) -> int:
        """返回每个动作的token数量。"""
        return self._tokens_per_action

    def tokenize(self, action: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        将一个动作token化。

        参数:
        - action: 一个包含动作的字典，其中键为动作名称，值为动作的张量。

        返回:
        - action_tokens: 一个表示动作的token的张量，形状为(tokens_per_action)或(batch, tokens_per_action)。
        """
        action_tokens = []
        # 按照self._action_order的顺序进行token化
        for k in self._action_order:
            a = action[k]  # 获取当前动作的值
            a_space = self._action_space[k]  # 获取当前动作的空间类型

            if isinstance(a_space, spaces.Discrete):
                # 如果是Discrete类型，动作值应该小于词汇表大小
                assert torch.all(a < self._vocab_size), "Discrete类型的动作应小于词汇表大小。"
                token = a  # Discrete类型的动作已经是token，形状为()或(batch,)
                token = a.unsqueeze(-1)  # 调整形状为(1,)或(batch, 1)，Discrete动作只会产生一个token

            else:  # 如果是Box类型，动作值的形状为(action_size)或(batch, action_size)
                low = torch.tensor(a_space.low).to(a.device)
                high = torch.tensor(a_space.high).to(a.device)
                a = torch.clamp(a, low, high)  # 将动作值限制在允许范围内
                # 对动作值进行归一化
                token = (a - low) / (high - low)
                # 将动作值离散化到vocab_size范围内
                token = token * (self._vocab_size - 1)
                token = token.to(torch.int32)  # 将动作值转换为整数，形状为(action_size)或(batch, action_size)
            
            action_tokens.append(token)  # 将token添加到结果列表中

        # 将所有动作的token连接在一起，结果形状为(tokens_per_action)或(batch, tokens_per_action)
        action_tokens = torch.concat(action_tokens, dim=-1)
        return action_tokens

    # action_tokens的形状为(tokens_per_action)或(batch, tokens_per_action)
    def detokenize(self, action_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        将一个动作的token反向token化。

        参数:
        - action_tokens: 一个表示动作的token的张量，形状为(tokens_per_action)或(batch, tokens_per_action)。

        返回:
        - action: 一个包含反向token化后动作的字典。
        """
        action = {}
        token_index = 0
        # action_tokens按self._action_order顺序排列
        # 因此我们也按self._action_order顺序进行反向token化
        for k in self._action_order:
            space = self._action_space[k]
            if isinstance(space, spaces.Discrete):
                # Discrete类型的动作已经是tokens。
                # action_tokens[k]的形状为(1,)或(batch,)。
                action[k] = action_tokens[..., token_index]
                # 如果模型输出的token超出了允许范围，则将其设置为默认值，即0 token。
                action[k] = torch.where(action[k] > space.n, torch.zeros_like(action[k]), action[k])
                token_index += 1

            else:
                # 对于Box类型的动作，反向token化并进行去归一化
                actions = []
                action_dim = space.shape[0]  # 获取动作的维度
                for j in range(action_dim):
                    a = action_tokens[..., token_index:token_index + 1]  # 形状为(1,)或(batch, 1)
                    a = a.to(torch.float32)  # 将int32转换为float32
                    # 反向归一化
                    a = a / (self._vocab_size - 1)
                    a = a * (space.high[j] - space.low[j]) + space.low[j]
                    actions.append(a)
                    token_index += 1
                # 将所有维度的动作连接在一起，形状为(action_dim)或(batch, action_dim)
                action[k] = torch.concat(actions, dim=-1)
        return action
