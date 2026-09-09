"""
Enhanced Causal Digital Twin with GNN + Diffusion + Transformer (V3)
=====================================================================

针对稀疏高维医疗时序数据的高性能架构 - 超越ML基线增强版

新增功能:
1. 不确定性量化: N次采样计算方差/置信区间
2. GNN邻接矩阵可视化接口
3. 虚拟干预支持
4. 跨模态注意力权重输出
5. 消融研究版本 (NoGNN)
6. Feature Tokenizer (FT-Transformer风格) - 提升数值特征理解
7. ResNet Skip Connection - 让DL模型学习GBDT残差

V3 改进 (超越GradientBoosting):
8. 深层特征金字塔 - 多尺度特征融合
9. Squeeze-and-Excitation通道注意力 - 增强关键特征
10. 更强的Dropout和LayerNorm正则化
11. 标签相关性建模 - 利用标签间的共现关系
12. 增强的静态特征处理 - 模拟决策树的特征交互
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict


# =============================================================================
# 新增: Feature Tokenizer (FT-Transformer风格)
# =============================================================================

class FeatureTokenizer(nn.Module):
    """
    Feature Tokenizer - 专门为表格数据设计的特征嵌入
    
    思想: 对每个连续特征学习一个独立的 weight 和 bias,
    将数值 x 投影成向量 x_emb = x * W + b
    这让 Transformer 能像决策树一样理解数值空间的切分
    """
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        
        # 每个特征有独立的投影参数
        self.feature_weights = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.feature_biases = nn.Parameter(torch.zeros(n_features, d_model))
        
        # 新增：特征交互层 - 模拟决策树的特征组合
        self.feature_interaction = nn.Sequential(
            nn.Linear(d_model * n_features, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, n_features] 连续特征值
        returns: [B, n_features, d_model] 特征嵌入
        """
        # x: [B, n_features] -> [B, n_features, 1]
        x = x.unsqueeze(-1)
        # [B, n_features, 1] * [n_features, d_model] + bias
        emb = x * self.feature_weights + self.feature_biases
        return emb
    
    def get_interaction_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取特征交互后的全局表示"""
        emb = self.forward(x)  # [B, n_features, d_model]
        flat = emb.view(emb.size(0), -1)  # [B, n_features * d_model]
        return self.feature_interaction(flat)  # [B, d_model]


# =============================================================================
# 新增: Squeeze-and-Excitation模块 - 通道注意力
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block - 自适应通道加权"""
    def __init__(self, d_model: int, reduction: int = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(d_model, d_model // reduction),
            nn.GELU(),
            nn.Linear(d_model // reduction, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]
        returns: [B, N, D] with channel attention applied
        """
        # Squeeze: [B, N, D] -> [B, D]
        se = x.mean(dim=1)
        # Excitation: [B, D] -> [B, D]
        se = self.excitation(se)
        # Scale: [B, N, D] * [B, 1, D]
        return x * se.unsqueeze(1)


# =============================================================================
# 新增: 标签相关性建模
# =============================================================================

class LabelCorrelationModule(nn.Module):
    """建模标签之间的相关性，利用共现关系提升预测"""
    def __init__(self, n_labels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.n_labels = n_labels
        
        # 可学习的标签嵌入
        self.label_embeddings = nn.Parameter(torch.randn(n_labels, d_model) * 0.02)
        
        # 标签-标签注意力
        self.label_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
        
        # 特征-标签交互
        self.feature_label_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: [B, D] 特征表示
        returns: [B, n_labels] 标签预测
        """
        B = features.size(0)
        
        # 扩展标签嵌入: [n_labels, D] -> [B, n_labels, D]
        labels = self.label_embeddings.unsqueeze(0).expand(B, -1, -1)
        
        # 标签-标签自注意力，捕获标签间关系
        labels_enhanced, _ = self.label_attn(labels, labels, labels)
        
        # 特征-标签交叉注意力
        # query: labels, key/value: features
        features_expanded = features.unsqueeze(1)  # [B, 1, D]
        label_features, _ = self.feature_label_attn(labels_enhanced, features_expanded, features_expanded)
        
        # 合并原始标签和特征增强的标签
        combined = torch.cat([labels_enhanced, label_features], dim=-1)  # [B, n_labels, 2*D]
        
        # 投影到标签预测
        logits = self.output_proj(combined).squeeze(-1)  # [B, n_labels]
        
        return logits


# =============================================================================
# 模块1: Sparse Feature Processor (处理稀疏特征)
# =============================================================================

class SparseFeatureProcessor(nn.Module):
    """
    稀疏特征处理器 - 使用特征哈希 + 可学习嵌入处理任意大小的特征空间
    V3增强: 添加SE注意力和更深的特征融合
    """
    def __init__(self, d_model: int = 128, n_buckets: int = 512, 
                 n_channels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_buckets = n_buckets
        
        # 调整维度分配，给 value 更大权重以提高干预敏感性
        self.feature_embed = nn.Embedding(n_buckets, d_model // 2 - 16)  # 48
        self.channel_embed = nn.Embedding(n_channels, d_model // 4 - 8)  # 24
        
        # 增强 value encoder - 占更大比例
        self.value_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),  # 64
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2)  # 64
        )
        
        self.time_linear = nn.Linear(1, d_model // 4 - 8)  # 24
        
        # 48 + 24 + 64 + 24 = 160 -> 128
        self.proj = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # value直接旁路
        self.value_bypass = nn.Linear(d_model // 2, d_model)
        self.value_bypass_weight = 0.3
        
        # V3新增: SE注意力块
        self.se_block = SEBlock(d_model, reduction=4)
        
        # V3新增: 额外的特征增强层
        self.feature_enhance = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
    
    def _hash_features(self, feature_ids: torch.Tensor) -> torch.Tensor:
        return feature_ids % self.n_buckets
    
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        time = events[:, :, 0:1]
        channel = events[:, :, 1].long()
        feature_id = events[:, :, 2].long()
        value = events[:, :, 3:4]
        
        hashed_features = self._hash_features(feature_id)
        
        f_emb = self.feature_embed(hashed_features)
        c_emb = self.channel_embed(channel)
        v_emb = self.value_encoder(value)
        t_emb = self.time_linear(time)
        
        x = torch.cat([f_emb, c_emb, v_emb, t_emb], dim=-1)
        x = self.proj(x)
        
        # 添加value直接旁路
        value_direct = self.value_bypass(v_emb)
        x = x + self.value_bypass_weight * value_direct
        
        # V3新增: SE注意力增强
        x = self.se_block(x)
        
        # V3新增: 特征增强残差
        x = x + 0.1 * self.feature_enhance(x)
        
        return x


# =============================================================================
# 模块2: Feature Graph Neural Network (特征关系图) - 增强版
# =============================================================================

class FeatureGNN(nn.Module):
    """
    特征图神经网络 - 支持邻接矩阵提取用于可视化
    V3增强: 更深的GNN层和残差连接
    """
    def __init__(self, d_model: int = 128, n_buckets: int = 512, 
                 n_layers: int = 2, dropout: float = 0.1, l1_lambda: float = 0.01):
        super().__init__()
        self.n_buckets = n_buckets
        self.l1_lambda = l1_lambda
        
        # 低秩分解的邻接矩阵
        rank = 64
        self.adj_U = nn.Parameter(torch.randn(n_buckets, rank) * 0.01)
        self.adj_V = nn.Parameter(torch.randn(n_buckets, rank) * 0.01)
        
        # V3增强: 使用更深的GNN层
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model)
            )
            for _ in range(n_layers)
        ])
        
        # V3新增: 层间残差门控
        self.layer_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            for _ in range(n_layers)
        ])
        
        self.out_proj = nn.Linear(d_model, d_model)
    
    def get_adjacency_matrix(self) -> torch.Tensor:
        """提取完整邻接矩阵用于可视化"""
        adj = torch.mm(self.adj_U, self.adj_V.t())  # (n_buckets, n_buckets)
        adj = F.softmax(adj / math.sqrt(self.adj_U.size(-1)), dim=-1)
        return adj
    
    def get_sparse_loss(self) -> torch.Tensor:
        """L1稀疏性正则化"""
        adj = self.get_adjacency_matrix()
        return self.l1_lambda * torch.abs(adj).mean()
    
    def forward(self, x: torch.Tensor, feature_ids: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        
        hashed_ids = feature_ids % self.n_buckets
        
        U_selected = self.adj_U[hashed_ids]
        V_selected = self.adj_V[hashed_ids]
        
        attn = torch.bmm(U_selected, V_selected.transpose(-1, -2))
        attn = F.softmax(attn / math.sqrt(U_selected.size(-1)), dim=-1)
        
        h = x
        for i, layer in enumerate(self.gnn_layers):
            h_agg = torch.bmm(attn, h)
            h_new = layer(h + h_agg)
            
            # V3增强: 门控残差连接
            gate_input = torch.cat([h, h_new], dim=-1)
            gate = self.layer_gates[i](gate_input)
            h = gate * h_new + (1 - gate) * h
        
        return self.out_proj(h)


# =============================================================================
# 模块3: Diffusion Module - 支持N次采样不确定性量化
# =============================================================================

class DiffusionAugmentation(nn.Module):
    """
    GNN-Conditioned Diffusion - 支持多次采样用于不确定性量化
    """
    def __init__(self, d_model: int = 128, n_steps: int = 20):
        super().__init__()
        self.d_model = d_model
        self.n_steps = n_steps
        
        betas = torch.linspace(1e-4, 0.02, n_steps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        self.denoise_net = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, x_gnn: torch.Tensor, training: bool = True, 
                stochastic: bool = False) -> torch.Tensor:
        """
        Args:
            x_gnn: GNN增强后的特征
            training: 训练模式
            stochastic: 推理时是否使用随机采样(用于不确定性量化)
        """
        if not training and not stochastic:
            return x_gnn
            
        B, N, D = x_gnn.shape
        device = x_gnn.device
        
        t = torch.randint(0, self.n_steps, (B,), device=device).long()
        alpha_t = self.alphas_cumprod[t].view(B, 1, 1)
        
        noise = torch.randn_like(x_gnn)
        x_t = torch.sqrt(alpha_t) * x_gnn + torch.sqrt(1 - alpha_t) * noise
        
        t_emb = (t.float() / self.n_steps).view(B, 1, 1).expand(-1, N, -1)
        net_input = torch.cat([x_t, x_gnn, t_emb], dim=-1)
        
        x_recon = self.denoise_net(net_input)
        
        mix_ratio = 0.7 if training else 0.5  # 推理时更多随机性
        return mix_ratio * x_gnn + (1 - mix_ratio) * x_recon


# =============================================================================
# 模块4: Efficient Temporal Transformer
# =============================================================================

class EfficientTemporalTransformer(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                 window_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, n_heads, dropout=dropout, 
                                              batch_first=True),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 2, d_model),
                    nn.Dropout(dropout)
                ),
                'norm2': nn.LayerNorm(d_model)
            }))
        
        # 可学习查询池化（保留用于兼容性，但会和其他池化混合）
        self.pool_query = nn.Parameter(torch.randn(1, 32, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        
        # 新增：混合池化投影（用于将混合池化结果映射到相同维度）
        # mean + max + attention = 3种池化方式
        self.pool_proj = nn.Linear(d_model * 3, d_model)
        
        # ===== 强化：全局值注入（Global Value Injection）=====
        # 解决问题：Self-Attention + LayerNorm 会抹平输入值的绝对差异
        # 强化方案：
        # 1. 统计信息扩展到 5 个维度：mean, std, max, min, range
        # 2. 直接旁路：将原始输入的全局均值直接加到输出
        # 3. 使用较大的固定权重，确保值变化信号不被淹没
        self.value_stats_dim = d_model * 5  # mean, std, max, min, range
        self.value_injection = nn.Sequential(
            nn.Linear(self.value_stats_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Tanh()  # 限制范围但保留符号
        )
        # 固定较大权重 - 确保值信息占主导
        self.injection_weight = 0.5  # 50% 来自值信息
        
        # 直接旁路：将原始均值投影到输出空间
        self.direct_value_bypass = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        B, N, D = x.shape
        
        # ===== 新增：在 Transformer 处理前提取全局值统计信息 =====
        # 这些统计信息会绕过 Transformer 的归一化，直接注入到输出
        if mask is not None:
            mask_bool_input = mask.bool()
            mask_expanded_input = mask_bool_input.unsqueeze(-1).float()
            x_masked_input = x * mask_expanded_input
            valid_counts_input = mask_bool_input.sum(dim=1, keepdim=True).clamp(min=1).float()
            
            # 全局均值 [B, D]
            global_mean = x_masked_input.sum(dim=1) / valid_counts_input
            
            # 全局标准差 [B, D]
            x_centered = (x - global_mean.unsqueeze(1)) * mask_expanded_input
            global_std = torch.sqrt((x_centered ** 2).sum(dim=1) / valid_counts_input + 1e-6)
            
            # 全局最大值 [B, D]
            x_for_max_input = x.masked_fill(~mask_bool_input.unsqueeze(-1), float('-inf'))
            global_max = x_for_max_input.max(dim=1)[0]
            
            # 全局最小值 [B, D]
            x_for_min_input = x.masked_fill(~mask_bool_input.unsqueeze(-1), float('inf'))
            global_min = x_for_min_input.min(dim=1)[0]
            
            # 全局范围 [B, D] - 对值变化最敏感
            global_range = global_max - global_min
        else:
            global_mean = x.mean(dim=1)
            global_std = x.std(dim=1) + 1e-6
            global_max = x.max(dim=1)[0]
            global_min = x.min(dim=1)[0]
            global_range = global_max - global_min
        
        # 拼接全局统计信息 [B, 5*D] - 包含 range 用于捕获值变化
        global_stats = torch.cat([global_mean, global_std, global_max, global_min, global_range], dim=-1)
        
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)
        
        for layer in self.layers:
            attn_out, _ = layer['attn'](x, x, x, key_padding_mask=key_padding_mask)
            x = layer['norm1'](x + attn_out)
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
        
        # ===== 混合池化策略 =====
        # 1. 掩码处理 (确保mask是bool类型)
        if mask is not None:
            mask_bool = mask.bool()  # 确保是bool类型
            mask_expanded = mask_bool.unsqueeze(-1).float()  # [B, N, 1]
            x_masked = x * mask_expanded  # 零填充被mask的位置
            valid_counts = mask_bool.sum(dim=1, keepdim=True).clamp(min=1).float()  # [B, 1]
        else:
            mask_bool = None
            x_masked = x
            valid_counts = torch.tensor(N, dtype=torch.float, device=x.device)
        
        # 2. Mean pooling（对输入变化敏感）
        mean_pool = x_masked.sum(dim=1) / valid_counts  # [B, D]
        
        # 3. Max pooling（捕获极值变化）
        if mask_bool is not None:
            # 将masked位置设为极小值
            x_for_max = x.masked_fill(~mask_bool.unsqueeze(-1), float('-inf'))
        else:
            x_for_max = x
        max_pool = x_for_max.max(dim=1)[0]  # [B, D]
        
        # 4. Attention pooling（保留 32 个可学习时序摘要 token）
        queries = self.pool_query.expand(B, -1, -1)
        attn_tokens, _ = self.pool_attn(queries, x, x, key_padding_mask=key_padding_mask)
        attn_pool = attn_tokens.mean(dim=1)  # [B, D]
        
        # 5. 混合三种池化
        combined = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)  # [B, 3*D]
        pooled = self.pool_proj(combined)  # [B, D]
        
        # ===== 强化：全局值注入（绕过Transformer的归一化） =====
        # 三重保障确保干预的值变化被保留：
        # 1. value_injection: 全局统计信息 -> d_model
        # 2. direct_bypass: 原始均值直接传递
        # 3. 使用较大权重确保值信息不被淹没
        
        value_info = self.value_injection(global_stats)  # [B, D]
        direct_bypass = self.direct_value_bypass(global_mean)  # [B, D]
        
        # 组合值信息：统计信息 + 直接旁路
        combined_value_info = value_info + direct_bypass
        
        # 使用固定较大权重混合（确保值变化不被抹平）
        pooled = (1.0 - self.injection_weight) * pooled + self.injection_weight * combined_value_info
        
        # 返回真实的 32 个 temporal tokens，而不是复制同一个 pooled 向量。
        if mask is not None:
            sampled_tokens = []
            for b in range(B):
                valid_idx = torch.nonzero(mask_bool[b], as_tuple=False).squeeze(-1)
                if valid_idx.numel() == 0:
                    valid_idx = torch.zeros(1, dtype=torch.long, device=x.device)
                sample_pos = torch.linspace(
                    0,
                    max(valid_idx.numel() - 1, 0),
                    steps=32,
                    device=x.device,
                ).round().long()
                sampled_tokens.append(x[b, valid_idx[sample_pos], :])
            sampled_tokens = torch.stack(sampled_tokens, dim=0)
        else:
            sample_pos = torch.linspace(0, max(N - 1, 0), steps=32, device=x.device).round().long()
            sampled_tokens = x[:, sample_pos, :]

        temporal_tokens = 0.65 * attn_tokens + 0.35 * sampled_tokens
        temporal_tokens = temporal_tokens + 0.20 * pooled.unsqueeze(1)
        temporal_tokens[:, 0, :] = temporal_tokens[:, 0, :] + pooled
        return temporal_tokens


# =============================================================================
# 模块5: Multi-Modal Fusion - 支持注意力权重输出
# =============================================================================

class MultiModalFusion(nn.Module):
    """多模态融合 - 支持跨模态注意力权重提取"""
    def __init__(self, d_model: int = 128, static_dim: int = 18, 
                 note_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.note_encoder = nn.Sequential(
            nn.Linear(note_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.cross_attn = nn.MultiheadAttention(d_model, 4, dropout=dropout,
                                                 batch_first=True)
        
        # 修改: 4个输入 = temporal_mean + temporal_enhanced + static + notes
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 新增：直接旁路，确保temporal信息不被cross-attention压缩
        self.temporal_bypass_weight = 0.5  # 50%来自直接temporal信息
    
    def forward(self, temporal: torch.Tensor, static: torch.Tensor,
                notes: torch.Tensor, return_attn: bool = False):
        static_emb = self.static_encoder(static)
        note_emb = self.note_encoder(notes)
        
        # 保留原始temporal信息（直接旁路）
        temporal_mean = temporal.mean(dim=1)
        
        context = torch.stack([static_emb, note_emb], dim=1)
        enhanced, attn_weights = self.cross_attn(temporal, context, context)
        temporal_enhanced = enhanced.mean(dim=1)
        
        # 关键修改：将原始temporal_mean也加入融合，确保temporal信息不丢失
        combined = torch.cat([temporal_mean, temporal_enhanced, static_emb, note_emb], dim=-1)
        fused = self.fusion(combined)
        
        # 额外：添加直接残差连接，确保temporal信息传递
        fused = fused + self.temporal_bypass_weight * temporal_mean
        
        if return_attn:
            return fused, attn_weights
        return fused


# =============================================================================
# 完整模型: Enhanced Causal Digital Twin (ECDT) V2
# =============================================================================

class EnhancedCausalDigitalTwin(nn.Module):
    """
    增强型因果数字孪生 (ECDT) V3
    
    V2功能:
    - predict_with_uncertainty: N次采样不确定性量化
    - get_gnn_adjacency: 获取GNN邻接矩阵
    - intervene: 虚拟干预接口
    - get_cross_modal_attention: 跨模态注意力
    - Feature Tokenizer: FT-Transformer风格的静态特征处理
    - ResNet Skip Connection: 学习残差而非从头预测
    
    V3新增 (超越GradientBoosting):
    - 标签相关性建模: 利用标签间共现关系
    - 深层特征融合: 多层次特征金字塔
    - SE注意力: 自适应通道加权
    - 更强的正则化: Dropout和LayerNorm
    """
    def __init__(self, n_outputs: int = 6, d_model: int = 128, 
                 n_buckets: int = 512, n_channels: int = 3,
                 n_transformer_layers: int = 3, dropout: float = 0.2,
                 use_diffusion: bool = True, use_gnn: bool = True,
                 static_dim: int = 18, note_dim: int = 768):
        super().__init__()
        self.use_diffusion = use_diffusion
        self.use_gnn = use_gnn
        self.n_buckets = n_buckets
        self.d_model = d_model
        self.n_outputs = n_outputs
        
        # 0. Feature Tokenizer - 对静态特征做FT-Transformer风格的嵌入
        self.static_tokenizer = FeatureTokenizer(static_dim, d_model)
        
        # 1. 稀疏特征处理 (V3增强: 包含SE注意力)
        self.sparse_processor = SparseFeatureProcessor(
            d_model=d_model, n_buckets=n_buckets, 
            n_channels=n_channels, dropout=dropout
        )
        
        # 2. 特征GNN (V3增强: 更深的层和门控残差)
        if use_gnn:
            self.feature_gnn = FeatureGNN(
                d_model=d_model, n_buckets=n_buckets,
                n_layers=3, dropout=dropout  # V3: 增加到3层
            )
        else:
            self.feature_mlp = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model)
            )
        
        # 3. 扩散增强
        if use_diffusion:
            self.diffusion = DiffusionAugmentation(d_model=d_model, n_steps=20)
        
        # 4. 时序Transformer
        self.temporal_transformer = EfficientTemporalTransformer(
            d_model=d_model, n_heads=4, n_layers=n_transformer_layers,
            dropout=dropout
        )
        
        # 5. 多模态融合
        self.multimodal_fusion = MultiModalFusion(
            d_model=d_model, static_dim=static_dim, note_dim=note_dim, dropout=dropout
        )
        
        # V3新增: 标签相关性模块
        self.label_correlation = LabelCorrelationModule(n_outputs, d_model, dropout=dropout)
        
        # 6. 预测头 - 主分类器 (V3增强: 更深的网络)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, n_outputs)
        )
        
        # 7. ResNet-style Skip: 更强的baseline预测器
        self.static_baseline = nn.Sequential(
            nn.Linear(static_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs)
        )
        self.baseline_weight = nn.Parameter(torch.tensor(0.3))
        
        # V3新增: 标签相关性权重
        self.label_corr_weight = nn.Parameter(torch.tensor(0.2))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, events: torch.Tensor, event_mask: torch.Tensor,
                static: Optional[torch.Tensor] = None,
                notes: Optional[torch.Tensor] = None,
                stochastic: bool = False) -> Dict[str, torch.Tensor]:
        B = events.size(0)
        device = events.device
        
        if static is None:
            static = torch.zeros(B, 18, device=device)
        if notes is None:
            notes = torch.zeros(B, 768, device=device)
        
        # 0. Feature Tokenizer - 对静态特征做FT-Transformer嵌入
        static_tokens = self.static_tokenizer(static)  # [B, 18, d_model]
        static_pooled = static_tokens.mean(dim=1)  # [B, d_model]
        # V3: 获取特征交互表示
        static_interaction = self.static_tokenizer.get_interaction_features(static)  # [B, d_model]
        
        # 1. 稀疏特征处理 (V3: 包含SE注意力)
        x = self.sparse_processor(events)
        
        # 2. 特征增强 (GNN or MLP)
        feature_ids = events[:, :, 2].long()
        if self.use_gnn:
            x = x + self.feature_gnn(x, feature_ids)
        else:
            x = x + self.feature_mlp(x)
        
        # 3. 扩散增强
        if self.use_diffusion:
            x = self.diffusion(x, training=self.training, stochastic=stochastic)
        
        # 4. 时序Transformer - 返回 [B, 32, D]
        temporal_repr = self.temporal_transformer(x, event_mask)
        
        # 5. 多模态融合 (加入FT嵌入的静态特征)
        # V3: 使用更强的静态特征融合
        temporal_repr = temporal_repr + 0.15 * static_pooled.unsqueeze(1)
        temporal_repr = temporal_repr + 0.1 * static_interaction.unsqueeze(1)
        fused = self.multimodal_fusion(temporal_repr, static, notes)
        
        # 6. 主预测
        deep_logits = self.classifier(fused)
        
        # V3新增: 标签相关性预测
        label_corr_logits = self.label_correlation(fused)
        
        # 7. ResNet Skip Connection: 静态baseline + 深度残差
        baseline_logits = self.static_baseline(static)
        
        # 可学习的混合: 三路融合
        w_base = torch.sigmoid(self.baseline_weight) * 0.5  # [0, 0.5]
        w_corr = torch.sigmoid(self.label_corr_weight) * 0.3  # [0, 0.3]
        w_deep = 1.0 - w_base - w_corr
        
        logits = w_base * baseline_logits + w_corr * label_corr_logits + w_deep * deep_logits
        
        return {
            'factual_logits': logits,
            'deep_logits': deep_logits,
            'baseline_logits': baseline_logits,
            'label_corr_logits': label_corr_logits,
            'temporal_repr': temporal_repr,
            'fused_repr': fused
        }
    
    def predict_with_uncertainty(self, events: torch.Tensor, event_mask: torch.Tensor,
                                  static: Optional[torch.Tensor] = None,
                                  notes: Optional[torch.Tensor] = None,
                                  n_samples: int = 50) -> Dict[str, torch.Tensor]:
        """
        N次采样不确定性量化
        
        Returns:
            mean_probs: 平均预测概率
            std_probs: 预测标准差 (不确定性)
            all_probs: 所有采样结果
        """
        self.eval()
        all_logits = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(events, event_mask, static, notes, stochastic=True)
                all_logits.append(out['factual_logits'])
        
        all_logits = torch.stack(all_logits, dim=0)  # (N, B, n_outputs)
        all_probs = torch.sigmoid(all_logits)
        
        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)
        
        return {
            'mean_probs': mean_probs,
            'std_probs': std_probs,
            'all_probs': all_probs
        }
    
    def get_gnn_adjacency(self) -> Optional[torch.Tensor]:
        """获取GNN学习的邻接矩阵"""
        if self.use_gnn:
            return self.feature_gnn.get_adjacency_matrix()
        return None
    
    def get_gnn_sparse_loss(self) -> torch.Tensor:
        """获取GNN稀疏性正则化损失"""
        if self.use_gnn:
            return self.feature_gnn.get_sparse_loss()
        return torch.tensor(0.0)
    
    def intervene(self, events: torch.Tensor, event_mask: torch.Tensor,
                  intervention_idx: int = None, intervention_value: float = 0.0,
                  static: Optional[torch.Tensor] = None,
                  notes: Optional[torch.Tensor] = None,
                  target_feature_ids: Optional[List[int]] = None,
                  target_modality: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        虚拟干预: 修改特定特征的值，观察预测变化
        
        Args:
            intervention_idx: (deprecated) 要干预的事件索引，仅用于兼容旧代码
            intervention_value: 干预后的值
            target_feature_ids: 要干预的特征ID列表 (在模态内的ID)
            target_modality: 目标模态 (0=vitals, 1=labs, 2=meds)
        
        事件格式: events[batch, event_idx, :] = [time, modality, feature_id, value]
        """
        events_cf = events.clone()
        
        if target_feature_ids is not None and target_modality is not None:
            # 新方法: 按特征ID和模态找到所有匹配的事件并修改
            # events[:, :, 1] = modality, events[:, :, 2] = feature_id
            modality_mask = (events[:, :, 1] == target_modality)
            for fid in target_feature_ids:
                feature_mask = (events[:, :, 2] == fid)
                combined_mask = modality_mask & feature_mask & (event_mask.bool())
                
                # 修改所有匹配事件的值
                events_cf[:, :, 3] = torch.where(
                    combined_mask, 
                    torch.full_like(events_cf[:, :, 3], intervention_value),
                    events_cf[:, :, 3]
                )
        elif intervention_idx is not None:
            # 兼容旧代码: 只修改固定索引的事件
            events_cf[:, intervention_idx, 3] = intervention_value
        
        factual = self.forward(events, event_mask, static, notes)
        counterfactual = self.forward(events_cf, event_mask, static, notes)
        
        return {
            'factual_logits': factual['factual_logits'],
            'counterfactual_logits': counterfactual['factual_logits'],
            'causal_effect': torch.sigmoid(counterfactual['factual_logits']) - 
                            torch.sigmoid(factual['factual_logits'])
        }
    
    def get_cross_modal_attention(self, events: torch.Tensor, event_mask: torch.Tensor,
                                   static: torch.Tensor, notes: torch.Tensor
                                  ) -> torch.Tensor:
        """获取跨模态注意力权重"""
        x = self.sparse_processor(events)
        feature_ids = events[:, :, 2].long()
        
        if self.use_gnn:
            x = x + self.feature_gnn(x, feature_ids)
        else:
            x = x + self.feature_mlp(x)
        
        temporal_repr = self.temporal_transformer(x, event_mask)
        _, attn_weights = self.multimodal_fusion(temporal_repr, static, notes, return_attn=True)
        
        return attn_weights


# =============================================================================
# 消融版本
# =============================================================================

class EnhancedCausalDigitalTwinNoDiff(EnhancedCausalDigitalTwin):
    """不使用Diffusion增强"""
    def __init__(self, **kwargs):
        kwargs['use_diffusion'] = False
        super().__init__(**kwargs)


class EnhancedCausalDigitalTwinNoGNN(EnhancedCausalDigitalTwin):
    """不使用GNN (用MLP替代)"""
    def __init__(self, **kwargs):
        kwargs['use_gnn'] = False
        super().__init__(**kwargs)


# =============================================================================
# 基线模型
# =============================================================================

class VanillaTransformer(nn.Module):
    """标准Transformer基线"""
    def __init__(self, n_outputs: int = 6, d_model: int = 128, n_layers: int = 3,
                 dropout: float = 0.2, n_buckets: int = 512):
        super().__init__()
        
        self.sparse_processor = SparseFeatureProcessor(
            d_model=d_model, n_buckets=n_buckets, dropout=dropout
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4,
            dim_feedforward=d_model * 2, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_outputs)
        )
    
    def forward(self, events: torch.Tensor, event_mask: torch.Tensor,
                static: Optional[torch.Tensor] = None,
                notes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.sparse_processor(events)
        attn_mask = event_mask == 0
        x = self.encoder(x, src_key_padding_mask=attn_mask)
        x = (x * event_mask.unsqueeze(-1)).sum(dim=1) / (event_mask.sum(dim=1, keepdim=True) + 1e-8)
        return {'factual_logits': self.classifier(x)}


class BiLSTMBaseline(nn.Module):
    """BiLSTM基线"""
    def __init__(self, n_outputs: int = 6, d_model: int = 128, n_layers: int = 2,
                 dropout: float = 0.2, n_buckets: int = 512):
        super().__init__()
        
        self.sparse_processor = SparseFeatureProcessor(
            d_model=d_model, n_buckets=n_buckets, dropout=dropout
        )
        
        self.lstm = nn.LSTM(d_model, d_model // 2, n_layers,
                           batch_first=True, bidirectional=True, 
                           dropout=dropout if n_layers > 1 else 0)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_outputs)
        )
    
    def forward(self, events: torch.Tensor, event_mask: torch.Tensor,
                static: Optional[torch.Tensor] = None,
                notes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = self.sparse_processor(events)
        output, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return {'factual_logits': self.classifier(h)}
