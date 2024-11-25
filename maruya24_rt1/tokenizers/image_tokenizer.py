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

"""A FiLM Efficientnet contextual image tokenizer used in Robotics Transformer 1.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from maruya24_rt1.film_efficientnet.pretrained_efficientnet_encoder import (
    EfficientNetEncoder,
)
from maruya24_rt1.tokenizers.token_learner import TokenLearnerModule


class RT1ImageTokenizer(nn.Module):
    def __init__(
        self,
        embedding_output_dim: int = 512,  # 输出的嵌入维度
        language_embedding_size: int = 512,  # 语言嵌入的维度
        use_token_learner: bool = False,  # 是否使用Token Learner模块
        num_tokens: int = 8,  # 使用Token Learner模块时的token数量
    ):
        super().__init__()

        # 初始化EfficientNetEncoder，它是用于图像编码的主要网络
        # 该编码器将图像和可选的上下文（如自然语言嵌入）转换为嵌入表示
        self._tokenizer = EfficientNetEncoder(
            token_embedding_size=embedding_output_dim,  # 输出的嵌入维度
            language_embedding_size=language_embedding_size,  # 语言嵌入的维度
            early_film=True,  # 使用Film层进行早期融合
            pooling=False,  # 不进行池化
        )

        # 如果启用了Token Learner，则初始化相关模块
        self._use_token_learner = use_token_learner
        if self._use_token_learner:
            self._num_tokens = num_tokens  # Token Learner模块的token数量
            self._token_learner = TokenLearnerModule(
                inputs_channels=embedding_output_dim,  # 输入通道数，与嵌入维度相同
                num_tokens=self._num_tokens  # Token Learner模块的token数量
            )

    @property
    def tokens_per_context_image(self) -> int:
        """每个上下文图像的token数量。"""
        if self._use_token_learner:
            num_tokens = self._num_tokens  # 如果使用Token Learner，则返回指定的token数量
        else:
            num_tokens = 100  # 否则，返回默认的100个token
        return num_tokens

    # 注意：上下文在时间轴上是相同的。
    # 这意味着 (b, 0, embedding_dim) == (b, 1, embedding_dim) == (b, 2, embedding_dim) ...
    def forward(
        self, image: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """将图像转换为tokens。

        参数:
        image: 形状为 (b, t, 3, h, w) 的图像，用于token化。
        context: 可选的上下文向量（例如，自然语言嵌入）。期望形状为 (b, t, embedding_dim)。
        
        返回值:
        tokens: 形状为 (batch, t, num_tokens_per_timestep, embedding_dim) 的token。
        """
        b, t, c, h, w = image.shape  # 获取图像的批次大小、时间步数、通道数、高度和宽度

        # 将时间轴折叠到批次轴中，以简化处理
        image = image.view(b * t, c, h, w)
        if context is not None:
            context = context.view(b * t, -1)  # 如果有上下文，则也折叠时间轴

        # 使用EfficientNetEncoder对图像进行编码，并将上下文（如果有）传入
        tokens = self._tokenizer(image, context=context)  # 输出形状为 [b * t, 512, 10, 10]

        if self._use_token_learner:
            # 如果启用了Token Learner模块，对编码后的tokens进行进一步处理
            tokens = self._token_learner(tokens)  # 输出形状为 [b * t, num_token, 512]
            # 将之前折叠到批次轴中的时间轴展开回来
            tokens = tokens.view(b, t, tokens.shape[1], -1)
            return tokens  # 最终输出形状为 [b, t, num_token, 512]
        else:
            # 如果未启用Token Learner模块，继续处理tokens
            tokens = tokens.view(b, t, 512, -1)  # 将时间轴展开回来，形状为 [b, t, 512, 10 * 10]
            tokens = tokens.transpose(2, 3)  # 交换维度，使tokens的形状为 [b, t, 10 * 10, 512]
            return tokens  # 最终输出形状为 [b, t, 100, 512]，即100个tokens，每个token的嵌入维度为512
