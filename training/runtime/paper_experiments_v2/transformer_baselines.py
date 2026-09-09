"""
医疗Transformer基线模型实现
============================

解决审稿人质疑: "都2025年了，基线居然只有LSTM？"

实现的基线:
1. BEHRT (BERT for EHR) - Li et al., 2020
2. Med-BERT (Medical Concept BERT) - Rasmy et al., 2021  
3. ClinicalBERT (fine-tuned BERT) - Huang et al., 2019
4. TemporalTransformer (简化版时序Transformer)

这些是2020-2023年医疗预测领域的SOTA模型

参考文献:
- Li et al. "BEHRT: Transformer for Electronic Health Records" (Nature Scientific Reports, 2020)
- Rasmy et al. "Med-BERT: pre-trained contextualized embeddings" (npj Digital Medicine, 2021)
- Huang et al. "ClinicalBERT: Modeling Clinical Notes" (CHIL, 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import math


# =============================================================================
# 基础组件
# =============================================================================

class PositionalEncoding(nn.Module):
    """标准位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = self._build_pe(max_len, device=torch.device("cpu"), dtype=torch.float32)
        self.register_buffer('pe', pe)

    def _build_pe(self, max_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pe = torch.zeros(max_len, self.d_model, device=device, dtype=dtype)
        position = torch.arange(0, max_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype) *
            (-math.log(10000.0) / self.d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pe = self.pe
        if pe.size(1) < seq_len or pe.device != x.device or pe.dtype != x.dtype:
            pe = self._build_pe(max(seq_len, pe.size(1)), device=x.device, dtype=x.dtype)
            self.pe = pe
        x = x + pe[:, :seq_len, :]
        return self.dropout(x)


class LearnablePositionalEmbedding(nn.Module):
    """可学习位置嵌入 (BERT风格)"""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力"""
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        q = self.q_linear(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.out_linear(context)
        
        return output, attn_weights


class TransformerEncoderLayer(nn.Module):
    """标准Transformer编码器层"""
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x


# =============================================================================
# 1. BEHRT: BERT for Electronic Health Records
# =============================================================================

class BEHRT(nn.Module):
    """
    BEHRT: Transformer for Electronic Health Records
    
    参考: Li et al., "BEHRT: Transformer for Electronic Health Records" 
          Nature Scientific Reports, 2020
    
    特点:
    - 使用医疗概念嵌入 (诊断、药物、手术)
    - 年龄嵌入 + 时间段嵌入
    - 预训练+微调范式
    
    这里实现微调版本用于ICU预测
    """
    
    def __init__(self, 
                 n_features: int,
                 n_labels: int,
                 d_model: int = 288,
                 n_heads: int = 12,
                 n_layers: int = 6,
                 max_seq_len: int = 512,
                 n_age_buckets: int = 120,  # 0-120岁
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_labels = n_labels
        
        # 特征嵌入
        self.feature_embedding = nn.Linear(n_features, d_model)
        
        # 位置嵌入 (BERT风格)
        self.position_embedding = LearnablePositionalEmbedding(d_model, max_seq_len, dropout)
        
        # 年龄嵌入 (BEHRT特色)
        self.age_embedding = nn.Embedding(n_age_buckets, d_model)
        
        # 时间段嵌入 (day/night, weekday/weekend)
        self.segment_embedding = nn.Embedding(4, d_model)  # 4种组合
        
        # Transformer编码器
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        
        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_labels)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                features: torch.Tensor,
                age: Optional[torch.Tensor] = None,
                segment: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, seq_len, n_features] 特征序列
            age: [B, seq_len] 年龄 (0-119)
            segment: [B, seq_len] 时间段 (0-3)
            mask: [B, seq_len] 有效位置掩码
        
        Returns:
            {'logits': [B, n_labels], 'embeddings': [B, d_model]}
        """
        batch_size, seq_len, _ = features.shape
        
        # 特征嵌入
        x = self.feature_embedding(features)
        
        # 添加[CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 更新mask
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
        
        # 位置嵌入
        x = self.position_embedding(x)
        
        # 年龄嵌入 (可选)
        if age is not None:
            age = torch.clamp(age.long(), 0, 119)
            age_with_cls = torch.cat([
                torch.zeros(batch_size, 1, device=age.device, dtype=torch.long),
                age
            ], dim=1)
            x = x + self.age_embedding(age_with_cls)
        
        # 时间段嵌入 (可选)
        if segment is not None:
            segment = torch.clamp(segment.long(), 0, 3)
            segment_with_cls = torch.cat([
                torch.zeros(batch_size, 1, device=segment.device, dtype=torch.long),
                segment
            ], dim=1)
            x = x + self.segment_embedding(segment_with_cls)
        
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Transformer编码
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # 取[CLS] token的表示
        cls_repr = x[:, 0, :]
        
        # 分类
        logits = self.classifier(cls_repr)
        
        return {
            'logits': logits,
            'embeddings': cls_repr
        }


# =============================================================================
# 2. Med-BERT: Medical Concept BERT
# =============================================================================

class MedBERT(nn.Module):
    """
    Med-BERT: Pre-trained contextualized embeddings for clinical NLP
    
    参考: Rasmy et al., "Med-BERT: pre-trained contextualized embeddings 
          on large-scale structured electronic health records for disease prediction"
          npj Digital Medicine, 2021
    
    特点:
    - 基于ICD/CPT代码的预训练
    - 分层就诊嵌入
    - 疾病预测导向的微调
    
    这里实现用于ICU时序特征的版本
    """
    
    def __init__(self,
                 n_features: int,
                 n_labels: int,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 max_seq_len: int = 256,
                 n_visit_types: int = 10,  # 就诊类型
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_labels = n_labels
        
        # 特征嵌入 (模拟医疗概念嵌入)
        self.concept_embedding = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # 位置编码
        self.position_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 就诊类型嵌入 (Med-BERT特色)
        self.visit_embedding = nn.Embedding(n_visit_types, d_model)
        
        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        # 层次化聚合 (Med-BERT使用分层表示)
        self.hierarchical_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_labels)
        )
    
    def forward(self,
                features: torch.Tensor,
                visit_type: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, seq_len, n_features]
            visit_type: [B, seq_len] 就诊类型
            mask: [B, seq_len]
        """
        batch_size, seq_len, _ = features.shape
        
        # 概念嵌入
        x = self.concept_embedding(features)
        
        # 位置编码
        x = self.position_encoding(x)
        
        # 就诊类型嵌入
        if visit_type is not None:
            visit_type = torch.clamp(visit_type.long(), 0, 9)
            x = x + self.visit_embedding(visit_type)
        
        # Transformer编码
        src_key_padding_mask = (mask == 0) if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # 层次化注意力池化
        attn_scores = self.hierarchical_pool(x).squeeze(-1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        
        # 分类
        logits = self.classifier(pooled)
        
        return {
            'logits': logits,
            'embeddings': pooled,
            'attention_weights': attn_weights
        }


# =============================================================================
# 3. ClinicalBERT (简化版)
# =============================================================================

class ClinicalBERTForSequence(nn.Module):
    """
    ClinicalBERT for Sequence Classification
    
    参考: Huang et al., "ClinicalBERT: Modeling Clinical Notes and 
          Predicting Hospital Readmission"
          CHIL Workshop, 2019
    
    这是一个简化版本，不使用预训练权重，
    但保持相似的架构用于公平比较
    """
    
    def __init__(self,
                 n_features: int,
                 n_labels: int,
                 d_model: int = 768,  # BERT-base
                 n_heads: int = 12,
                 n_layers: int = 12,
                 max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_labels = n_labels
        
        # 输入投影 (模拟token embedding)
        self.input_projection = nn.Linear(n_features, d_model)
        
        # 位置嵌入
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        
        # Token类型嵌入 (简化为单一类型)
        self.token_type_embeddings = nn.Embedding(2, d_model)
        
        # Layer Normalization
        self.embeddings_layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.embeddings_dropout = nn.Dropout(dropout)
        
        # Transformer层
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        # 池化层 (BERT的[CLS] pooler)
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_labels)
        )
    
    def forward(self,
                features: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = features.shape
        device = features.device
        
        # 输入嵌入
        inputs_embeds = self.input_projection(features)
        
        # 位置嵌入
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)
        
        # Token类型嵌入
        if token_type_ids is None:
            token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        # 组合嵌入
        embeddings = inputs_embeds + position_embeds + token_type_embeds
        embeddings = self.embeddings_layer_norm(embeddings)
        embeddings = self.embeddings_dropout(embeddings)
        
        # Transformer编码
        src_key_padding_mask = (mask == 0) if mask is not None else None
        encoded = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # 取第一个位置作为[CLS]
        cls_output = encoded[:, 0, :]
        pooled_output = self.pooler(cls_output)
        
        # 分类
        logits = self.classifier(pooled_output)
        
        return {
            'logits': logits,
            'embeddings': pooled_output
        }


# =============================================================================
# 4. 轻量级时序Transformer基线
# =============================================================================

class TemporalTransformer(nn.Module):
    """
    轻量级时序Transformer
    
    特点:
    - 专为ICU时序数据设计
    - 使用相对位置编码
    - 参数量适中，训练更快
    """
    
    def __init__(self,
                 n_features: int,
                 n_labels: int,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 max_seq_len: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_labels = n_labels
        
        # 特征投影
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])
        
        # 聚合
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.classifier = nn.Linear(d_model, n_labels)
    
    def forward(self,
                features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = features.shape
        
        # 特征投影
        x = self.feature_proj(features)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer编码
        for layer in self.layers:
            x = layer(x, mask)
        
        # 池化
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            x = x * mask_expanded
            pooled = x.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        
        # 分类
        logits = self.classifier(pooled)
        
        return {
            'logits': logits,
            'embeddings': pooled
        }


# =============================================================================
# 基线比较框架
# =============================================================================

class BaselineComparison:
    """
    基线模型比较框架
    """
    
    AVAILABLE_BASELINES = {
        'BEHRT': BEHRT,
        'Med-BERT': MedBERT,
        'ClinicalBERT': ClinicalBERTForSequence,
        'TemporalTransformer': TemporalTransformer,
    }
    
    @classmethod
    def create_model(cls, name: str, n_features: int, n_labels: int, 
                     d_model: int = 128, **kwargs) -> nn.Module:
        """创建基线模型"""
        if name not in cls.AVAILABLE_BASELINES:
            raise ValueError(f"Unknown baseline: {name}. Available: {list(cls.AVAILABLE_BASELINES.keys())}")
        
        model_cls = cls.AVAILABLE_BASELINES[name]
        return model_cls(n_features=n_features, n_labels=n_labels, d_model=d_model, **kwargs)
    
    @classmethod
    def get_model_info(cls, name: str) -> Dict:
        """获取模型信息"""
        info = {
            'BEHRT': {
                'paper': 'Li et al., Nature Scientific Reports 2020',
                'features': ['Age embedding', 'Segment embedding', 'BERT-style'],
                'recommended_params': {'d_model': 288, 'n_layers': 6, 'n_heads': 12}
            },
            'Med-BERT': {
                'paper': 'Rasmy et al., npj Digital Medicine 2021',
                'features': ['Visit embedding', 'Hierarchical pooling', 'Disease-oriented'],
                'recommended_params': {'d_model': 256, 'n_layers': 4, 'n_heads': 8}
            },
            'ClinicalBERT': {
                'paper': 'Huang et al., CHIL 2019',
                'features': ['BERT-base architecture', 'Clinical text pre-training'],
                'recommended_params': {'d_model': 768, 'n_layers': 12, 'n_heads': 12}
            },
            'TemporalTransformer': {
                'paper': 'Lightweight baseline',
                'features': ['Simple Transformer', 'Position encoding', 'Fast training'],
                'recommended_params': {'d_model': 128, 'n_layers': 3, 'n_heads': 4}
            }
        }
        return info.get(name, {})
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 使用示例
# =============================================================================

def demo_baselines():
    """演示基线模型"""
    print("=" * 60)
    print("Transformer基线模型演示")
    print("=" * 60)
    
    # 配置
    n_features = 50
    n_labels = 6
    batch_size = 8
    seq_len = 128
    
    # 模拟数据
    x = torch.randn(batch_size, seq_len, n_features)
    mask = torch.ones(batch_size, seq_len)
    mask[:, -20:] = 0  # 最后20个位置是padding
    
    print(f"\n输入形状: {x.shape}")
    print(f"掩码形状: {mask.shape}")
    
    print("\n" + "-" * 60)
    
    # 使用可被n_heads整除的d_model
    d_model_configs = {
        'BEHRT': 288,  # 288 % 12 = 0
        'Med-BERT': 256,  # 256 % 8 = 0
        'ClinicalBERT': 768,  # 768 % 12 = 0
        'TemporalTransformer': 128  # 128 % 4 = 0
    }
    
    for name in BaselineComparison.AVAILABLE_BASELINES.keys():
        print(f"\n📦 {name}")
        
        info = BaselineComparison.get_model_info(name)
        print(f"   论文: {info.get('paper', 'N/A')}")
        print(f"   特点: {', '.join(info.get('features', []))}")
        
        # 创建模型 - 使用对应的d_model配置
        d_model = d_model_configs.get(name, 128)
        model = BaselineComparison.create_model(name, n_features, n_labels, d_model=d_model)
        n_params = BaselineComparison.count_parameters(model)
        print(f"   参数量: {n_params:,}")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(x, mask=mask)
        
        print(f"   输出logits形状: {output['logits'].shape}")
        print(f"   嵌入形状: {output['embeddings'].shape}")
    
    print("\n" + "=" * 60)
    print("✓ 所有基线模型可正常运行")
    print("=" * 60)


if __name__ == "__main__":
    demo_baselines()
