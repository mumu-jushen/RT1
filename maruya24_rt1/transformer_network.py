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

from maruya24_rt1.tokenizers import action_tokenizer
from maruya24_rt1.tokenizers import image_tokenizer
from maruya24_rt1.transformer import Transformer
from maruya24_rt1.film_efficientnet import preprocessors

from typing import Optional, Tuple, Union, Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F


# This is a robotics transformer network.
# TransformerNetwork类定义了一个用于机器人控制任务的基于Transformer的神经网络。
class TransformerNetwork(nn.Module):
    """基于Transformer的演员网络。"""

    def __init__(
        self,
        input_tensor_space: spaces.Dict,  # 输入张量空间，通常是一个包含多个传感器数据的字典，例如图像、语言嵌入等。
        output_tensor_space: spaces.Dict,  # 输出张量空间，通常是一个包含机器人动作数据的字典，例如世界坐标、旋转变化、夹爪动作等。
        train_step_counter: int = 0,  # 训练步数计数器，初始为0。
        vocab_size: int = 256,  # 词汇表大小，通常是输出层的维度，也对应输入层的维度。
        token_embedding_size: int = 512,  # 图像tokenizer输出的嵌入维度。这将用于EfficientNetEncoder类中的1x1卷积。
        language_embedding_size: int = 512,  # 语言嵌入的维度，也用于FiLM层。
        num_layers: int = 1,  # Transformer的层数。
        layer_size: int = 4096,  # 这是attention机制中的key_dim，对应每个attention头的query、key和value的维度。
        num_heads: int = 8,  # 注意力头的数量。
        feed_forward_size: int = 512,  # feed-forward网络的维度，对应Transformer部分的d_model。
        dropout_rate: float = 0.1,  # dropout的比例。
        time_sequence_length: int = 1,  # 时间序列的长度。
        crop_size: int = 236,  # 图像裁剪的尺寸。
        use_token_learner: Optional[bool] = True,  # 是否使用Token Learner模块。
        return_attention_scores: bool = False,  # 是否返回注意力得分，通常用于调试和可视化。
        num_encoders=1,  # 编码器的数量。
        using_proprioception=False,  # 是否使用本体感知数据（如机器人自身的状态）。
    ):
        super().__init__()

        # 初始化一些内部参数和变量
        self._input_tensor_space = input_tensor_space
        self._output_tensor_space = output_tensor_space
        self._train_step_counter = train_step_counter
        self._actions = None
        self._returns = None
        self._vocab_size = vocab_size
        self._token_embedding_size = token_embedding_size
        self._language_embedding_size = language_embedding_size
        self._time_sequence_length = time_sequence_length
        self._crop_size = crop_size
        self.num_encoders = num_encoders

        # 创建Transformer模型
        # 创建图像tokenizer模块，用于将图像转换为tokens
        self._image_tokenizers = nn.ModuleDict()
        self.language_embedding = nn.Embedding(512, language_embedding_size)
        for idx_encoder in range(num_encoders):
            self._image_tokenizers[str(idx_encoder)] = (
                image_tokenizer.RT1ImageTokenizer(
                    embedding_output_dim=self._token_embedding_size,
                    language_embedding_size=self._language_embedding_size,
                    use_token_learner=use_token_learner,
                    num_tokens=8,
                )
            )
            self._image_tokenizers[str(idx_encoder)].eval()

        # 创建动作tokenizer，用于将动作数据转换为tokens
        self._action_tokenizer = action_tokenizer.RT1ActionTokenizer(
            output_tensor_space, vocab_size=self._vocab_size
        )

        self.using_proprioception = using_proprioception
        if self.using_proprioception:
            # 创建使用本体感知数据的Transformer
            self._transformer = Transformer(
                num_layers=num_layers,
                layer_size=layer_size,
                num_heads=num_heads,
                feed_forward_size=feed_forward_size,
                dropout_rate=dropout_rate,
                vocab_size=self._vocab_size,
                input_token_emb_dim=self._token_embedding_size + 1,
                return_attention_scores=return_attention_scores,
            )
        else:
            # 创建不使用本体感知数据的Transformer
            self._transformer = Transformer(
                num_layers=num_layers,
                layer_size=layer_size,
                num_heads=num_heads,
                feed_forward_size=feed_forward_size,
                dropout_rate=dropout_rate,
                vocab_size=self._vocab_size,
                input_token_emb_dim=self._token_embedding_size,
                return_attention_scores=return_attention_scores,
            )

        # 获取每个动作和上下文图像的token数量
        self._tokens_per_action = self._action_tokenizer.tokens_per_action
        self._tokens_per_context_image = self._image_tokenizers[
            "0"
        ].tokens_per_context_image

        # 生成损失掩码和注意力掩码
        self._generate_masks()

        # 定义损失函数
        self.loss_weight = torch.ones(vocab_size)
        self._loss_object = nn.CrossEntropyLoss(
            reduction="mean", weight=self.loss_weight
        )

        self._attention_scores = []
        self._use_token_learner = use_token_learner

        # 定义状态空间，用于存储Transformer的状态
        self._state_space = spaces.Dict(
            {
                "context_image_tokens": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        time_sequence_length,
                        self._tokens_per_context_image,
                        token_embedding_size,
                    ),
                    dtype=np.float32,
                ),
                "context_pos_orn": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(time_sequence_length, self._tokens_per_context_image),
                    dtype=np.float32,
                ),
                "action_tokens": spaces.MultiDiscrete(
                    np.full((time_sequence_length, self._tokens_per_action), vocab_size)
                ),
                "seq_idx": spaces.Discrete(time_sequence_length + 1),
            }
        )

    @property
    def attention_scores(self) -> List[torch.Tensor]:
        """返回注意力得分。通常用于调试和可视化。"""
        return self._attention_scores

    def _get_action_index_for_token(self, k):
        """返回与给定位置`k`相关联的动作索引。

        如果k不是动作token，则返回-1。
        如果k属于序列中的第一个动作，则返回0，依此类推。

        参数:
            k: 表示序列中位置的整数。

        返回值:
            此位置所属动作的索引，如果该位置属于图像token，则返回-1。
        """
        if k < 0 or k >= self._all_num_tokens:
            return -1

        n = k
        if (
            n % self._single_time_step_num_tokens < self._tokens_per_context_image
        ):  # 检查k是否为上下文图像token
            return -1
        return int(
            n / self._single_time_step_num_tokens
        )  # 返回k所属的时间步索引。

    def _generate_masks(self):
        """生成用于动作预测损失和注意力可视化的掩码。"""
        # 每个时间步 = [图像, 动作]
        self._single_time_step_num_tokens = (
            self._tokens_per_action + self._tokens_per_context_image
        )

        # 完整的序列 = [前缀上下文 + N个时间步 + 后缀上下文]
        self._all_num_tokens = (
            self._time_sequence_length * self._single_time_step_num_tokens
        )

        # 为动作预测损失创建掩码
        self._action_tokens_mask = []
        for n in range(0, self._all_num_tokens, self._single_time_step_num_tokens):
            for x in range(0, self._tokens_per_action, 1):
                self._action_tokens_mask.append(x + n + self._tokens_per_context_image)

        # 确保因果性的look ahead掩码。
        # 这是一个下三角矩阵。除0以外的所有元素为1。
        # 0表示掩码。
        self._default_attention_mask = torch.tril(
            torch.ones((self._all_num_tokens, self._all_num_tokens), dtype=torch.uint8)
        )

        action_mask = torch.from_numpy(
            np.ndarray(shape=(self._all_num_tokens, self._all_num_tokens), dtype=int)
        )

        for i in range(self._all_num_tokens):
            for j in range(self._all_num_tokens):
                action_i = self._get_action_index_for_token(i)
                action_j = self._get_action_index_for_token(j)
                mask = 0
                if (
                    action_i != -1 and action_j != -1
                ):  # 检查i和j是否都是动作。
                    # 忽略前一个时间步的动作。
                    if action_j < action_i:
                        mask = 1
                    # 如果不是自回归，忽略当前时间步的动作。
                    if action_j == action_i and j <= i:
                        mask = 1
                action_mask[i, j] = mask
        action_mask = action_mask.to(self._default_attention_mask.device)
        self._default_attention_mask -= action_mask

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        network_state: Dict[
            str, torch.Tensor
        ],  # network_state保留观测tokens、动作tokens和序列索引。
    ):
        """调用Transformer网络。

        参数:
            observations: 包含图像和自然语言嵌入的观测数据，格式为字典的Tensors。
            network_state: 包含时间步、图像、动作tokens、步数等的网络状态，格式为字典的Tensors。

        返回值:
            一个元组`(解码后的输出动作, 网络状态)`。
        """

        # 用于确定是训练还是推理调用
        # outer_rank为2 -> [b, t]表示训练期间
        # outer_rank为1 -> [b]表示推理期间
        outer_rank = self._get_outer_rank(observations)
        assert outer_rank in (1, 2), "outer rank 应该为 1 或 2"

        b, t = self._get_batch_size_and_seq_len(network_state)
        # network_state用于推理。
        # b : 批次大小
        # t: 模型的时间序列长度

        # context_image_tokens: (b, t, num_tokens, embedding_dim)
        # action_tokens: (b, t, self._tokens_per_action)
        context_image_tokens, action_tokens, attention_mask = self._get_tokens_and_mask(
            observations, network_state
        )

        self._aux_info = {"action_labels": action_tokens}

        if outer_rank == 1:  # 推理调用
            # 在循环中运行Transformer以逐个生成动作token
            seq_idx = network_state["seq_idx"][0]
            action_t = torch.minimum(
                seq_idx, torch.tensor(self._time_sequence_length - 1)
            )
            # Transformer默认向左移动所有token一步（通常是预测下一个token的默认训练任务...）。
            transformer_shift = -1
            # 我们只想获取在time_step时预测的动作。
            # 这个索引表示在action_t时间步的最后一个观察token的输出。
            start_index = (
                transformer_shift
                + self._tokens_per_context_image
                + action_t * (self._single_time_step_num_tokens)
            )
            current_action_tokens = []
            action_predictions_logits = []
            # 重复推理tokens_per_action次。
            for k in range(self._tokens_per_action):
                action_index = start_index + k
                # token: (1, 1)
                # token_logits: (1, 1 vocab_size)
                token, token_logits = self._transformer_call_and_slice(
                    context_image_tokens,
                    action_tokens,
                    attention_mask=attention_mask,
                    batch_size=b,
                    slice_start=action_index,  # 切片单个动作维度
                )
                action_predictions_logits.append(token_logits)

                current_action_tokens.append(token)

                # 将预测的token添加到action_tokens
                action_tokens = action_tokens.view(
                    b, -1
                )  # [b, t, self._tokens_per_action] -> [b, t * self._tokens_per_action]
                action_start_index = (action_t * self._tokens_per_action) + k
                # 用预测的token替换action_tokens[:, action_start_index]。注意，这不是插入操作。
                action_tokens = torch.concat(
                    [
                        action_tokens[:, :action_start_index],
                        token,
                        action_tokens[:, action_start_index + 1 :],
                    ],
                    dim=1,
                )
                action_tokens = action_tokens.view(
                    b, t, self._tokens_per_action
                )  # [b, t * self._tokens_per_action] -> [b, t, self._tokens_per_action]

            self._aux_info.update(
                {
                    # action_predictions_logits是
                    # [1, self._tokens_per_action, self._vocab_size]
                    "action_predictions_logits": torch.concat(
                        action_predictions_logits, 1
                    )
                }
            )

            predicted_tokens_for_output = torch.concat(
                current_action_tokens, 1
            )  # [1, self._tokens_per_action]
            one_state_action_tokens = predicted_tokens_for_output.unsqueeze(
                1
            )  # [1, 1, self._tokens_per_action]

            # 将预测的动作tokens添加到network_state['action_tokens']
            state_action_tokens = network_state[
                "action_tokens"
            ]  # (1, time_sequence_length, self._tokens_per_action)
            # 用预测的tokens替换state_action_tokens[:, action_t, ...]。注意，这不是插入操作。
            network_state["action_tokens"] = torch.concat(
                [
                    state_action_tokens[:, :action_t, ...],
                    one_state_action_tokens,
                    state_action_tokens[:, action_t + 1 :, ...],
                ],
                dim=1,
            )

            # 增加时间步数以用于下一次推理调用。
            # network_state['seq_idx']永远不会超过time_sequence_length。
            network_state["seq_idx"] = torch.minimum(
                seq_idx + 1, torch.tensor(self._time_sequence_length)
            )[None]

            self._loss = torch.tensor(0.0)

        else:
            # 训练调用 --> 只需运行一次Transformer的前向传递
            # output_tokens: (bs, t*num_tokens, vocab_size)
            output_tokens = self._transformer_call(
                context_image_tokens,
                action_tokens,
                attention_mask=attention_mask,
                batch_size=b,
            )

            # 聚合所有预测的动作以计算动作损失。使用高级索引提取所有预测的动作。
            predicted_action_index = torch.tensor(self._action_tokens_mask) - 1
            action_logits = output_tokens[
                :, predicted_action_index
            ]  # (bs, t*tokens_per_action, vocab_size)
            action_logits_for_training = action_logits.view(
                b, t, self._tokens_per_action, -1
            )  # (bs, t, self._tokens_per_action, vocab_size)

            # 只取最后一个动作作为动作输出。
            # action_logits_for_output为[b, self._tokens_per_action, emb]
            action_logits_for_output = action_logits_for_training[
                :, -1
            ]  # 这将获取该训练中的最后一个时间步的动作。
            # predicted_tokens_for_output为[b, self._tokens_per_action]
            predicted_tokens_for_output = torch.argmax(action_logits_for_output, dim=-1)
            action_logits_for_training = action_logits_for_training.permute(0, 3, 1, 2)
            num_items = float(b * t) * self._single_time_step_num_tokens
            # action_logits_for_training: (b, t, self._tokens_per_action, vocab_size)
            # action_tokens, (b, t, self._tokens_per_action)
            # action_loss: (b, t)
            action_loss = self._loss_object(
                action_logits_for_training, action_tokens
            )  # (b, t, self._tokens_per_action)

            self._loss = action_loss

            # 存储动作标签和预测结果以便于可视化
            self._aux_info.update(
                {
                    "action_predictions": torch.argmax(
                        action_logits_for_training, dim=-1
                    ),
                    "action_loss": action_loss,
                    "actor_loss_mask": torch.ones((b), dtype=torch.float32),
                }
            )

        # 解码输出动作
        output_actions = self._action_tokenizer.detokenize(predicted_tokens_for_output)

        # output_actions是最后的动作。
        # network_stape是用于下次推理的过去状态。
        return output_actions, network_state

    def _get_outer_rank(self, observations: Dict[str, torch.Tensor]) -> int:
        # 用于确定是训练还是推理调用
        # outer_rank为2 -> [b, t]表示训练期间
        # outer_rank为1 -> [b]表示推理期间

        for k in observations.keys():
            obs_value = observations[k]
            obs_value_shape = obs_value.shape

            obs_space = self._input_tensor_space[k]
            obs_space_shape = obs_space.shape
            break
        return len(obs_value_shape) - len(obs_space_shape)

    def _get_batch_size_and_seq_len(self, network_state):
        image_shape = network_state["context_image_tokens"].shape
        b = image_shape[0]
        t = image_shape[1]
        return b, t

    def _transformer_call(
        self,
        context_image_tokens: torch.Tensor,  # (b, t, num token, emb_dim)
        action_tokens: torch.Tensor,  # (b, t, self._tokens_per_action)
        batch_size: int,
        attention_mask: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # 组装输入token序列
        input_token_sequence = self._assemble_input_token_sequence(
            context_image_tokens, action_tokens, batch_size
        )  # [b, t*num_tokens, emb_dim]
        # 运行Transformer
        output_tokens, self._attention_scores = self._transformer(
            input_token_sequence, attention_mask
        )  # (bs, t*num_tokens, vocab_size)
        return output_tokens

    # input_token_sequence = [context_image_tokens + action_tokens]
    def _assemble_input_token_sequence(
        self, context_image_tokens, action_tokens, batch_size
    ):
        # 嵌入动作tokens
        # action_tokens = F.one_hot(action_tokens, num_classes=self._vocab_size).to(torch.float32)
        # action_tokens = self._action_token_emb(action_tokens) # [b, t , num_action_tokens, emb_dim]

        # 将动作tokens设置为与上下文图像tokens相同的大小，这里不使用自回归条件
        action_tokens = torch.zeros_like(
            context_image_tokens
        )
        # 组装token序列
        input_token_sequence = torch.concat(
            (context_image_tokens, action_tokens), dim=2
        )
        if self.using_proprioception:
            input_token_sequence = input_token_sequence.view(
                batch_size, -1, self._token_embedding_size + 1
            )
        else:
            input_token_sequence = input_token_sequence.view(
                batch_size, -1, self._token_embedding_size
            )  # [b, t*num_tokens, emb_dim]
        return input_token_sequence

    # 调用Transformer，切片输出，返回预测的token。
    def _transformer_call_and_slice(
        self, *args, slice_start: int = 0, slice_length: int = 1, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = self._transformer_call(*args, **kwargs)

        slice_end = slice_start + slice_length
        token_logits = output_tokens[
            :, slice_start:slice_end, :
        ]  # (b, slice_length, vocab_size)
        token = torch.argmax(token_logits, dim=-1)

        return token, token_logits

    def _get_tokens_and_mask(
        self,
        observations: Dict[str, torch.Tensor],
        network_state: Dict[str, torch.Tensor],
    ):
        # 对所有输入进行token化
        context_image_tokens, network_state = self._tokenize_images(
            observations, network_state
        )
        # token化动作数据
        action_tokens = self._tokenize_actions(observations, network_state)

        # 生成Transformer注意力掩码
        attention_mask = self._default_attention_mask

        return (context_image_tokens, action_tokens, attention_mask)

    # 在训练期间，我们不会使用network_state。
    # 在训练期间，这将只用于将图像和上下文转换为tokens。
    def _tokenize_images(self, observations, network_state):
        image = observations["image"]  # [b, t, c, h, w] or [b, c, h, w]
        outer_rank = self._get_outer_rank(observations)

        if outer_rank == 1:  # 这是一个推理调用
            seq_idx = network_state["seq_idx"][0]  # 0 ~ time_sequence_length
            time_step = torch.minimum(
                seq_idx, torch.tensor(self._time_sequence_length - 1)
            )
            image = image.unsqueeze(1)  # [b, c, h, w] -> [b, 1, c, h, w]

        image_shape = image.shape
        b = image_shape[0]
        input_t = image_shape[1]
        c = image_shape[2]
        h = image_shape[3]
        w = image_shape[4]

        # 从观测中提取上下文
        context = self._extract_context_from_observation(
            observations, input_t
        )  # [b, t, emb-size] or None
        context = context.mean(dim=-1)

        # 预处理图像
        image = image.view(
            (b * input_t, c, h, w)
        )
        image = image.view((b, input_t, c, h, w))

        # 获取图像tokens
        context_image_tokens = []
        for i in range(self.num_encoders):
            img = image[:, :, :, i * 256 : (i + 1) * 256, :]
            with torch.no_grad():
                context_image_tokens.append(
                    self._image_tokenizers[str(i)](img, context=context)
                )  # (batch, t, num_tokens, embedding_dim)
        context_image_tokens = sum(context_image_tokens)
        num_tokens = context_image_tokens.shape[2]
        if self.using_proprioception:
            context_pos_orn = torch.cat(
                (observations["position"], observations["orientation"]), dim=-1
            )
            context_pos_orn = context_pos_orn.view(b, input_t, context_pos_orn.size(-1))

        if outer_rank == 1:  # 这是一个推理调用
            state_image_tokens = network_state[
                "context_image_tokens"
            ]  # (1, time_sequence_length, tokens_per_context_image, token_embedding_size)
            # network_state作为此调用的输入是上次调用的输出。
            # 因此，我们需要在时间轴上将所有图像向左移动1个位置，以与此调用中的时间维度对齐。
            state_image_tokens = (
                torch.roll(state_image_tokens, -1, 1)
                if seq_idx == self._time_sequence_length
                else state_image_tokens
            )
            context_image_tokens = torch.concat(
                [
                    state_image_tokens[:, :time_step, ...],
                    context_image_tokens,
                    state_image_tokens[
                        :, time_step + 1 :, ...
                    ],  # 如果time_step == time_sequence_lengths -1，这将是空张量。
                ],
                dim=1,
            )
            network_state["context_image_tokens"] = context_image_tokens
            if self.using_proprioception:
                state_pos_orn = network_state["context_pos_orn"]
                state_pos_orn = (
                    torch.roll(state_pos_orn, -1, 1)
                    if seq_idx == self._time_sequence_length
                    else state_pos_orn
                )
                context_pos_orn = torch.concat(
                    [
                        state_pos_orn[:, :time_step, ...],
                        context_pos_orn,
                        state_pos_orn[
                            :, time_step + 1 :, ...
                        ],
                    ],
                    dim=1,
                )
                network_state["context_pos_orn"] = context_pos_orn
        if self.using_proprioception:
            context_image_tokens = torch.cat(
                (context_image_tokens, context_pos_orn.unsqueeze(-1)), dim=-1
            )

        return context_image_tokens, network_state

    def _tokenize_actions(self, observations, network_state):
        outer_rank = self._get_outer_rank(observations)

        if outer_rank == 1:  # 这是一个推理调用
            action_tokens = network_state["action_tokens"]
            seq_idx = network_state["seq_idx"][0]
            action_tokens = (
                torch.roll(action_tokens, -1, 1)
                if seq_idx == self._time_sequence_length
                else action_tokens
            )
        else:
            assert outer_rank == 2
            if (
                self._actions is None
            ):  # 如果没有动作需要被token化，则创建一个全零的张量。
                b, t = self._get_batch_size_and_seq_len(network_state)
                action_tokens = torch.zeros(
                    shape=[b, t, self._tokens_per_action], dtype=torch.int32
                )
            else:
                action_tokens = self._action_tokenizer.tokenize(self._actions)
        return action_tokens

    # 从观测中提取上下文。大小为[b, t, emb-size]。
    def _extract_context_from_observation(self, observations, seq_len):
        """从观测中提取上下文。"""
        context = None
        if "natural_language_embedding" in observations:
            outer_rank = self._get_outer_rank(observations)

            context = self.language_embedding(
                observations["natural_language_embedding"].long()
            )  # [b, t, emb-size] or [b, emb-size]
            if outer_rank == 1:
                context = torch.tile(context[:, None], [1, seq_len, 1])
                # [b, emb-size] ->  [b, 1, emb-size] -> [b, seq_len, emb-size]
        return context

    # 只有通过此函数才能将动作传递给此类。
    def set_actions(self, actions: Dict[str, torch.Tensor]):
        """设置将被token化并用于Transformer网络的动作。

        参数:
        actions: 将被token化并用于Transformer网络的动作。例如：
            动作 terminate = 1 world_vector = [0.9, 0.8, -0.3]
            旋转变化 rotation_delta = [-0.1, 0.2, .6] 夹爪闭合度 gripper_closedness = 0.9
        """
        self._actions = actions

    def get_actor_loss(self) -> torch.Tensor:
        return self._loss

    def get_aux_info(self) -> Dict[str, Any]:
        return self._aux_info


if __name__ == "__main__":
    net = TransformerNetwork()
