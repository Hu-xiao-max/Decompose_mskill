"""
Diffusion Policy模型实现
基于扩散模型的机器人策略学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional
from einops import rearrange


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, dim: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim)
        )
        
        self.block1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
        self.block2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        
        # 添加时间嵌入
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb
        
        h = self.block2(h)
        return x + h


class CrossAttentionBlock(nn.Module):
    """交叉注意力块，用于融合视觉和状态信息"""
    
    def __init__(self, dim: int, context_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim)
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x_norm = self.norm(x)
        context_norm = self.context_norm(context)
        
        q = self.to_q(x_norm)
        k = self.to_k(context_norm)
        v = self.to_v(context_norm)
        
        # 重塑为多头注意力格式
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # 计算注意力
        attn = torch.softmax(torch.einsum('bhid,bhjd->bhij', q, k) * self.scale, dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return x + self.to_out(out)


class VisionEncoder(nn.Module):
    """视觉编码器，使用ResNet骨干网络"""
    
    def __init__(self, input_dim: int = 3, feature_dim: int = 512):
        super().__init__()
        
        # 简化的ResNet-like结构
        self.conv_layers = nn.Sequential(
            # 第一层
            nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 第二层
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第三层
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 第四层
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(512, feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, channels, height, width]
        batch_size, seq_len = x.shape[:2]
        
        # 重塑为[batch_size * seq_len, channels, height, width]
        x = rearrange(x, 'b s c h w -> (b s) c h w')
        
        # 通过卷积层
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        
        # 重塑回[batch_size, seq_len, feature_dim]
        features = rearrange(features, '(b s) d -> b s d', b=batch_size, s=seq_len)
        
        return features


class StateEncoder(nn.Module):
    """状态编码器，处理机器人状态信息"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256, output_dim: int = 512):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DiffusionPolicy(nn.Module):
    """Diffusion Policy主模型"""
    
    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 4,
        vision_feature_dim: int = 512,
        state_dim: int = 15,  # 7关节 + 7末端位姿 + 1夹爪
        hidden_dim: int = 512,
        num_diffusion_steps: int = 100,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_diffusion_steps = num_diffusion_steps
        
        # 编码器
        self.vision_encoder = VisionEncoder(feature_dim=vision_feature_dim)
        self.state_encoder = StateEncoder(state_dim, output_dim=vision_feature_dim)
        
        # 时间嵌入
        time_emb_dim = hidden_dim * 4
        self.time_embedding = SinusoidalPositionEmbedding(time_emb_dim)
        
        # 动作嵌入
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        
        # 上下文融合
        context_dim = vision_feature_dim * 2  # 视觉 + 状态
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        
        # Transformer层
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': CrossAttentionBlock(hidden_dim, hidden_dim, num_heads, dropout),
                'self_attn': nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                'mlp': ResidualBlock(hidden_dim, time_emb_dim, dropout),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim)
            })
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 噪声调度器
        self.register_buffer('betas', self._cosine_beta_schedule(num_diffusion_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """余弦噪声调度"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, 
                images: torch.Tensor, robot_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            noisy_actions: [batch_size, action_horizon, action_dim]
            timesteps: [batch_size]
            images: [batch_size, seq_len, 3, H, W]
            robot_states: [batch_size, seq_len, state_dim]
        """
        batch_size = noisy_actions.shape[0]
        
        # 编码视觉和状态信息
        vision_features = self.vision_encoder(images)  # [B, seq_len, vision_dim]
        state_features = self.state_encoder(robot_states)  # [B, seq_len, vision_dim]
        
        # 融合视觉和状态特征
        context_features = torch.cat([vision_features, state_features], dim=-1)  # [B, seq_len, 2*vision_dim]
        context_features = self.context_projection(context_features)  # [B, seq_len, hidden_dim]
        
        # 全局池化得到全局上下文
        global_context = torch.mean(context_features, dim=1, keepdim=True)  # [B, 1, hidden_dim]
        
        # 时间嵌入
        time_emb = self.time_embedding(timesteps)  # [B, time_emb_dim]
        
        # 动作嵌入
        x = self.action_embedding(noisy_actions)  # [B, action_horizon, hidden_dim]
        
        # Transformer层
        for layer in self.layers:
            # 交叉注意力（动作序列关注上下文）
            x_norm = layer['norm1'](x)
            x = x + layer['cross_attn'](x_norm, global_context)
            
            # 自注意力（动作序列内部注意力）
            x_norm = layer['norm2'](x)
            attn_out, _ = layer['self_attn'](x_norm, x_norm, x_norm)
            x = x + attn_out
            
            # MLP + 时间嵌入
            # 为每个时间步广播时间嵌入
            time_emb_expanded = time_emb.unsqueeze(1).expand(-1, self.action_horizon, -1)
            x = layer['mlp'](x, time_emb_expanded)
        
        # 输出预测的噪声
        predicted_noise = self.output_projection(x)
        
        return predicted_noise
    
    def add_noise(self, actions: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """为动作添加噪声"""
        noise = torch.randn_like(actions)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # 广播到正确的形状
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1)
        
        noisy_actions = sqrt_alphas_cumprod_t * actions + sqrt_one_minus_alphas_cumprod_t * noise
        
        return noisy_actions, noise
    
    @torch.no_grad()
    def sample(self, images: torch.Tensor, robot_states: torch.Tensor, 
               num_inference_steps: int = 50) -> torch.Tensor:
        """采样生成动作序列"""
        batch_size = images.shape[0]
        device = images.device
        
        # 初始化随机噪声
        actions = torch.randn(batch_size, self.action_horizon, self.action_dim, device=device)
        
        # 创建采样时间步
        timesteps = torch.linspace(self.num_diffusion_steps - 1, 0, num_inference_steps, device=device).long()
        
        for i, t in enumerate(timesteps):
            # 预测噪声
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            predicted_noise = self(actions, t_batch, images, robot_states)
            
            # 计算去噪参数
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            if i < len(timesteps) - 1:
                alpha_cumprod_t_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=device)
            
            # DDPM采样公式
            pred_original_sample = (actions - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            if i < len(timesteps) - 1:
                # 不是最后一步，继续去噪
                pred_sample_direction = torch.sqrt(1 - alpha_cumprod_t_prev) * predicted_noise
                actions = torch.sqrt(alpha_cumprod_t_prev) * pred_original_sample + pred_sample_direction
                
                # 添加噪声（非确定性采样）
                if t > 0:
                    noise = torch.randn_like(actions)
                    actions = actions + torch.sqrt(beta_t) * noise
            else:
                # 最后一步
                actions = pred_original_sample
        
        return actions


def create_diffusion_policy(
    action_dim: int = 7,
    action_horizon: int = 4,
    state_dim: int = 15,
    **kwargs
) -> DiffusionPolicy:
    """创建Diffusion Policy模型"""
    return DiffusionPolicy(
        action_dim=action_dim,
        action_horizon=action_horizon,
        state_dim=state_dim,
        **kwargs
    )


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_diffusion_policy().to(device)
    
    batch_size = 2
    seq_len = 8
    action_horizon = 4
    
    # 模拟输入数据
    images = torch.randn(batch_size, seq_len, 3, 224, 224).to(device)
    robot_states = torch.randn(batch_size, seq_len, 15).to(device)
    actions = torch.randn(batch_size, action_horizon, 7).to(device)
    timesteps = torch.randint(0, 100, (batch_size,)).to(device)
    
    # 添加噪声
    noisy_actions, noise = model.add_noise(actions, timesteps)
    
    # 前向传播
    predicted_noise = model(noisy_actions, timesteps, images, robot_states)
    
    print(f"输入动作形状: {actions.shape}")
    print(f"噪声动作形状: {noisy_actions.shape}")
    print(f"预测噪声形状: {predicted_noise.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试采样
    sampled_actions = model.sample(images, robot_states, num_inference_steps=10)
    print(f"采样动作形状: {sampled_actions.shape}")
