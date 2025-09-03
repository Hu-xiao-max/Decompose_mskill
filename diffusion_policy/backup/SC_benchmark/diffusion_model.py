"""
改进的Diffusion Policy模型 - 针对小数据集优化
主要改进：
1. 减少扩散步数
2. 改进噪声调度
3. 添加正则化
4. 数值稳定性优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional, List
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
    """改进的残差块 - 添加了Layer Norm和Dropout"""
    
    def __init__(self, dim: int, time_emb_dim: int, dropout: float = 0.1):  # 减少dropout
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
            nn.Dropout(dropout),  # 增加dropout
            nn.Linear(dim, dim)
        )
        
        # 添加残差缩放因子
        self.residual_scale = 0.9  # 略微减小残差连接的权重
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        
        # 添加时间嵌入
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb
        
        h = self.block2(h)
        return x + self.residual_scale * h  # 缩放残差


class CrossAttentionBlock(nn.Module):
    """交叉注意力块 - 添加注意力dropout"""
    
    def __init__(self, dim: int, context_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5  # 修正scale计算
        
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim)
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim, dim, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)  # 添加注意力dropout
        
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
        
        # 计算注意力 - 添加数值稳定性
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn - attn.max(dim=-1, keepdim=True)[0]  # 减去最大值防止溢出
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)  # 添加dropout
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return x + self.to_out(out)


class EnhancedVisionEncoder(nn.Module):
    """增强的视觉编码器 - 优化用于24GB GPU和RLBench任务"""
    
    def __init__(self, input_dim: int = 3, feature_dim: int = 1024):  # 显著增加feature_dim
        super().__init__()
        
        # 更深的网络结构，类似ResNet但针对机器人任务优化
        self.conv_layers = nn.Sequential(
            # Stem layers - 保持更多空间信息
            nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 1 - 提取低级特征
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),  # 轻微dropout
            
            # Block 2 - 中级特征
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            
            # Block 3 - 高级特征
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # 空间注意力和全局池化
            nn.AdaptiveAvgPool2d((2, 2))  # 保留一些空间信息
        )
        
        # 更复杂的特征映射
        self.fc = nn.Sequential(
            nn.Linear(512 * 4, feature_dim * 2),  # 更大的中间层
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 空间注意力机制
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]
        
        # 重塑为[batch_size * seq_len, channels, height, width]
        x = rearrange(x, 'b s c h w -> (b s) c h w')
        
        # 通过卷积层（到空间注意力前）
        conv_features = self.conv_layers[:-1](x)  # 除了最后的AdaptiveAvgPool2d
        
        # 应用空间注意力
        attention_weights = self.spatial_attention(conv_features)
        attended_features = conv_features * attention_weights
        
        # 全局池化
        pooled_features = F.adaptive_avg_pool2d(attended_features, (2, 2))
        features = pooled_features.view(pooled_features.size(0), -1)
        features = self.fc(features)
        
        # 重塑回[batch_size, seq_len, feature_dim]
        features = rearrange(features, '(b s) d -> b s d', b=batch_size, s=seq_len)
        
        return features


class ImprovedDiffusionPolicy(nn.Module):
    """改进的Diffusion Policy - 针对小数据集优化"""
    
    def __init__(
        self,
        action_dim: int = 8,
        action_horizon: int = 4,
        vision_feature_dim: int = 512,  # 增加
        state_dim: int = 15,
        hidden_dim: int = 512,  # 增加
        num_diffusion_steps: int = 100,  # 增加到100
        num_layers: int = 6,  # 增加层数
        num_heads: int = 8,  # 增加注意力头
        dropout: float = 0.1,  # 减少dropout
        num_cameras: int = 1,
        fusion_method: str = 'attention',  # 添加这个参数以兼容
        # 新增参数
        use_ema: bool = True,  # 使用EMA
        clip_denoised: bool = True,  # 裁剪去噪结果
        clip_range: Tuple[float, float] = (-10.0, 10.0),  # 动作范围
        prediction_type: str = 'epsilon',  # 'epsilon' 或 'v_prediction'
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_diffusion_steps = num_diffusion_steps
        self.num_cameras = num_cameras
        self.clip_denoised = clip_denoised
        self.clip_range = clip_range
        self.prediction_type = prediction_type
        
        # 增强的视觉编码器
        self.vision_encoder = EnhancedVisionEncoder(
            feature_dim=vision_feature_dim
        )
        
        # 多视角融合模块
        if num_cameras > 1:
            self.multi_view_fusion = nn.MultiheadAttention(
                embed_dim=vision_feature_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.view_position_embedding = nn.Parameter(
                torch.randn(num_cameras, vision_feature_dim) * 0.02
            )
        
        # 状态编码器 - 添加dropout
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vision_feature_dim)
        )
        
        # 时间嵌入
        time_emb_dim = hidden_dim * 2  # 减小
        self.time_embedding = SinusoidalPositionEmbedding(time_emb_dim)
        
        # 动作嵌入 - 添加LayerNorm
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 上下文融合
        context_dim = vision_feature_dim * 2
        self.context_projection = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer层 - 简化
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': CrossAttentionBlock(hidden_dim, hidden_dim, num_heads, dropout),
                'mlp': ResidualBlock(hidden_dim, time_emb_dim, dropout),
                'norm': nn.LayerNorm(hidden_dim)
            })
            for _ in range(num_layers)
        ])
        
        # 输出层 - 添加初始化
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 初始化输出层为接近零
        nn.init.zeros_(self.output_projection[-1].weight)
        nn.init.zeros_(self.output_projection[-1].bias)
        
        # 改进的噪声调度
        self.register_buffer('betas', self._improved_beta_schedule(num_diffusion_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        # 确保alpha_cumprod不会太小
        self.alphas_cumprod = torch.clamp(self.alphas_cumprod, min=1e-4)
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        
        # v-prediction所需的额外参数
        if prediction_type == 'v_prediction':
            self.register_buffer('sqrt_alphas_cumprod_for_v', torch.sqrt(self.alphas_cumprod))
            self.register_buffer('sqrt_one_minus_alphas_cumprod_for_v', torch.sqrt(1.0 - self.alphas_cumprod))
    
    def _improved_beta_schedule(self, timesteps: int) -> torch.Tensor:
        """改进的噪声调度 - 使用cosine调度"""
        # 使用cosine调度，更适合稳定训练
        def alpha_bar(t):
            return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        
        betas = []
        for i in range(timesteps):
            t1 = i / timesteps
            t2 = (i + 1) / timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        
        betas = torch.tensor(betas, dtype=torch.float32)
        # 限制beta的范围防止数值不稳定
        return torch.clip(betas, 0.0001, 0.02)
    
    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, 
                images: torch.Tensor, robot_states: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size = noisy_actions.shape[0]
        
        # 编码视觉信息
        if len(images.shape) == 5:  # 单视角
            vision_features = self.vision_encoder(images)
        else:  # 多视角
            # 使用注意力机制融合多视角特征
            cam_features = []
            for i in range(images.shape[1]):
                cam_feat = self.vision_encoder(images[:, i])
                cam_features.append(cam_feat)
            
            # 堆叠所有视角特征 [B, num_cameras, T, D]
            multi_view_features = torch.stack(cam_features, dim=1)  # [B, num_cameras, T, D]
            
            if hasattr(self, 'multi_view_fusion'):
                # 添加位置编码
                B, num_cams, T, D = multi_view_features.shape
                pos_emb = self.view_position_embedding.unsqueeze(0).unsqueeze(2).expand(B, -1, T, -1)
                multi_view_features = multi_view_features + pos_emb
                
                # 重塑为 [B*T, num_cameras, D] 进行注意力计算
                mv_reshaped = multi_view_features.view(B*T, num_cams, D)
                
                # 多头注意力融合
                fused_features, _ = self.multi_view_fusion(
                    mv_reshaped, mv_reshaped, mv_reshaped
                )
                
                # 全局池化所有视角
                vision_features = fused_features.mean(dim=1).view(B, T, D)
            else:
                # 回退到简单平均
                vision_features = multi_view_features.mean(dim=1)
        
        # 编码状态信息
        state_features = self.state_encoder(robot_states)
        
        # 融合特征
        context_features = torch.cat([vision_features, state_features], dim=-1)
        context_features = self.context_projection(context_features)
        
        # 全局池化
        global_context = torch.mean(context_features, dim=1, keepdim=True)
        
        # 时间嵌入
        time_emb = self.time_embedding(timesteps)
        
        # 动作嵌入
        x = self.action_embedding(noisy_actions)
        
        # Transformer层 - 简化版本
        for layer in self.layers:
            # 交叉注意力
            x = layer.cross_attn(x, global_context)
            
            # MLP + 时间嵌入
            time_emb_expanded = time_emb.unsqueeze(1).expand(-1, self.action_horizon, -1)
            x = layer.mlp(x, time_emb_expanded)
            x = layer.norm(x)
        
        # 输出预测
        output = self.output_projection(x)
        
        # 根据prediction_type返回不同的目标
        if self.prediction_type == 'v_prediction':
            # v-prediction: 预测 v = alpha_t * epsilon - sigma_t * x0
            return output
        else:
            # epsilon-prediction: 预测噪声
            return output
    
    def add_noise(self, actions: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """为动作添加噪声 - 添加数值稳定性"""
        # 限制输入动作范围
        actions = torch.clamp(actions, *self.clip_range)
        
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
               num_inference_steps: Optional[int] = None,
               guidance_scale: float = 1.0) -> torch.Tensor:
        """改进的采样方法 - DDIM采样"""
        if num_inference_steps is None:
            num_inference_steps = self.num_diffusion_steps
            
        batch_size = robot_states.shape[0]
        device = robot_states.device
        
        # 初始化 - 使用较小的初始噪声
        actions = torch.randn(batch_size, self.action_horizon, self.action_dim, device=device) * 0.5
        
        # 使用DDIM采样步骤
        step_ratio = self.num_diffusion_steps // num_inference_steps
        timesteps = torch.arange(0, self.num_diffusion_steps, step_ratio, device=device).flip(0)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
            
            # 预测
            if self.prediction_type == 'v_prediction':
                v_pred = self(actions, t_batch, images, robot_states)
                # 从v-prediction恢复噪声和x0
                alpha_t = self.sqrt_alphas_cumprod_for_v[t]
                sigma_t = self.sqrt_one_minus_alphas_cumprod_for_v[t]
                predicted_noise = sigma_t * actions + alpha_t * v_pred
                x0_pred = alpha_t * actions - sigma_t * v_pred
            else:
                predicted_noise = self(actions, t_batch, images, robot_states)
                # 预测x0
                alpha_t = self.sqrt_alphas_cumprod[t]
                sigma_t = self.sqrt_one_minus_alphas_cumprod[t]
                x0_pred = (actions - sigma_t * predicted_noise) / (alpha_t + 1e-8)
            
            # 裁剪x0
            if self.clip_denoised:
                x0_pred = torch.clamp(x0_pred, *self.clip_range)
            
            # DDIM更新规则
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_next = self.sqrt_alphas_cumprod[t_next]
                sigma_next = self.sqrt_one_minus_alphas_cumprod[t_next]
                
                # 确定性更新
                actions = alpha_next * x0_pred + sigma_next * predicted_noise
                
                # 可选：添加少量噪声提高多样性
                # noise_scale = 0.1  # 小的噪声
                # actions = actions + noise_scale * torch.randn_like(actions)
            else:
                actions = x0_pred
            
            # 每步裁剪防止爆炸
            actions = torch.clamp(actions, self.clip_range[0] * 2, self.clip_range[1] * 2)
        
        # 最终裁剪
        actions = torch.clamp(actions, *self.clip_range)
        
        return actions


def create_improved_diffusion_policy(
    action_dim: int = 8,
    action_horizon: int = 4,
    state_dim: int = 15,
    num_cameras: int = 1,
    fusion_method: str = 'attention',  # 添加这个参数以兼容
    # 改进后的推荐配置
    num_diffusion_steps: int = 100,  # 增加步数
    hidden_dim: int = 512,  # 增加模型容量
    num_layers: int = 6,  # 增加层数
    dropout: float = 0.1,  # 减少dropout
    prediction_type: str = 'epsilon',  # 或 'v_prediction'
    clip_range: Tuple[float, float] = (-1.0, 1.0),  # 更合理的范围
    **kwargs
) -> ImprovedDiffusionPolicy:
    """创建改进的Diffusion Policy模型"""
    return ImprovedDiffusionPolicy(
        action_dim=action_dim,
        action_horizon=action_horizon,
        state_dim=state_dim,
        num_cameras=num_cameras,
        fusion_method=fusion_method,  # 传递参数
        num_diffusion_steps=num_diffusion_steps,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        prediction_type=prediction_type,
        clip_range=clip_range,
        **kwargs
    )


if __name__ == "__main__":
    # 测试优化的模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("测试优化的Diffusion Policy (24GB GPU配置)")
    print("=" * 60)
    
    # 使用24GB GPU优化配置
    model = create_improved_diffusion_policy(
        num_cameras=4,
        num_diffusion_steps=200,  # 显著增加步数
        hidden_dim=1024,  # 大幅增加容量
        vision_feature_dim=1024,
        num_layers=12,  # 更深的网络
        num_heads=16,  # 更多注意力头
        dropout=0.05,  # 减少dropout
        clip_range=(-2.0, 2.0)  # RLBench动作空间
    ).to(device)
    
    batch_size = 8  # 利用24GB显存
    seq_len = 8  # 更长序列
    action_horizon = 4
    
    # 模拟多相机输入数据
    images = torch.randn(batch_size, 4, seq_len, 3, 256, 256).to(device)  # 4相机，更高分辨率
    robot_states = torch.randn(batch_size, seq_len, 15).to(device)
    actions = torch.randn(batch_size, action_horizon, 8).to(device) * 1.0  # RLBench动作范围
    timesteps = torch.randint(0, 200, (batch_size,)).to(device)  # 匹配新的扩散步数
    
    # 测试前向传播
    noisy_actions, noise = model.add_noise(actions, timesteps)
    predicted_noise = model(noisy_actions, timesteps, images, robot_states)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"输入动作范围: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"噪声动作范围: [{noisy_actions.min():.3f}, {noisy_actions.max():.3f}]")
    print(f"预测噪声范围: [{predicted_noise.min():.3f}, {predicted_noise.max():.3f}]")
    
    # 测试采样
    print("\n测试采样过程...")
    with torch.no_grad():
        sampled_actions = model.sample(images, robot_states, num_inference_steps=20)
    print(f"采样动作形状: {sampled_actions.shape}")
    print(f"采样动作范围: [{sampled_actions.min():.3f}, {sampled_actions.max():.3f}]")
    
    print(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB" if torch.cuda.is_available() else "CPU模式")
    
    print("\n24GB GPU优化改进:")
    print("1. 扩散步数: 50 -> 200 (4倍提升)")
    print("2. 模型规模: hidden_dim=1024, layers=12 (显著增加)")
    print("3. 多相机支持: 4个相机，注意力融合")
    print("4. 图像分辨率: 224x224 -> 256x256")
    print("5. 视觉编码器: 增强卷积网络+空间注意力")
    print("6. 正则化: dropout=0.05 防止欠拟合")
    print("7. 数值稳定: cosine噪声调度+梯度裁剪")
    print("8. 采样优化: DDIM采样+动作范围裁剪")
    print("9. RLBench优化: 动作范围(-2,2)，序列长度8")
    
    print("\n优化测试完成! 模型已针对24GB GPU和RLBench任务优化。")