"""
Enhanced Causal Digital Twin V5 (ECDT-V5)
=========================================

Upgrade for SOTA performance & External Validation
Features:
1. Gated Cross-Modal Fusion (Bidirectional Attention)
2. Knowledge-Enhanced GNN (Prior Injection)
3. Domain Adversarial Training (Gradient Reversal)
4. Calibration-Ready Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, List, Tuple
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .enhanced_cdt import (
    FeatureTokenizer, 
    SparseFeatureProcessor, 
    DiffusionAugmentation, 
    EfficientTemporalTransformer,
    LabelCorrelationModule
)

# =============================================================================
# Component 1: Gradient Reversal Layer for Domain Adaptation
# =============================================================================

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

# =============================================================================
# Component 2: Knowledge-Enhanced GNN
# =============================================================================

class KnowledgeGNN(nn.Module):
    """
    Knowledge-Graph Enhanced GNN
    Fuses data-driven adjacency with prior knowledge (e.g. UMLS, ICD co-occurrence)
    """
    def __init__(self, d_model: int = 128, n_buckets: int = 512, 
                 n_layers: int = 3, dropout: float = 0.1, l1_lambda: float = 0.01):
        super().__init__()
        self.n_buckets = n_buckets
        self.l1_lambda = l1_lambda
        
        # 1. Prior Adjacency (Initialize with identity, can be loaded with KG later)
        # Using a buffer so it's saved but not updated by optimizer
        self.register_buffer('prior_adj', torch.eye(n_buckets)) 
        self.prior_weight = nn.Parameter(torch.tensor(0.5))
        
        # 2. Learnable Adjacency (Low-rank factorization)
        rank = 64
        self.adj_U = nn.Parameter(torch.randn(n_buckets, rank) * 0.01)
        self.adj_V = nn.Parameter(torch.randn(n_buckets, rank) * 0.01)
        
        # 3. GNN Layers (Deeper GCN with Residuals)
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
        
        # Gating mechanism for skip connections
        self.layer_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            for _ in range(n_layers)
        ])
        
        self.out_proj = nn.Linear(d_model, d_model)
    
    def load_prior_knowledge(self, adj_matrix: torch.Tensor):
        """Load external knowledge graph adjacency"""
        if adj_matrix.shape == self.prior_adj.shape:
            self.prior_adj.copy_(adj_matrix)
        else:
            print(f"Warning: KG shape mismatch. Expected {self.prior_adj.shape}, got {adj_matrix.shape}")

    def get_adjacency_matrix(self) -> torch.Tensor:
        """Combine prior and learned adjacency"""
        # Learned Adjacency
        learned_adj = torch.mm(self.adj_U, self.adj_V.t())
        learned_adj = F.softmax(learned_adj / math.sqrt(self.adj_U.size(-1)), dim=-1)
        
        # Prior Adjacency (Normalized)
        # Add self-loops just in case
        prior = self.prior_adj + torch.eye(self.n_buckets, device=self.prior_adj.device)
        # Column normalization
        prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Adaptive Fusion
        alpha = torch.sigmoid(self.prior_weight)
        combined_adj = alpha * prior + (1 - alpha) * learned_adj
        return combined_adj
    
    def get_sparse_loss(self) -> torch.Tensor:
        learned_adj = torch.mm(self.adj_U, self.adj_V.t())
        return self.l1_lambda * torch.abs(learned_adj).mean()
    
    def forward(self, x: torch.Tensor, feature_ids: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        hashed_ids = feature_ids % self.n_buckets
        
        # Get dense adjacency for the batch is expensive, 
        # instead we project features to bucket space, apply GNN, then project back?
        # Current implementation assumes node-to-node attention within the sequence 
        # guided by global adjacency.
        
        # Efficient implementation: Look up relevant rows of Adjacency
        # But for full interaction we need A[id_i, id_j].
        
        # Simplified: We construct a batch-specific attention mask based on A
        # This is O(N^2) per batch item. 
        # Alternatively, use the learned attention from V3 but guided by A.
        
        # Let's stick to the V3 low-rank attention but inject Prior info
        
        U_selected = self.adj_U[hashed_ids] # [B, N, R]
        V_selected = self.adj_V[hashed_ids] # [B, N, R]
        
        # Learned Attention Scores
        attn_scores = torch.bmm(U_selected, V_selected.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(U_selected.size(-1))
        
        # Prior Injection: We need to look up prior_adj[id_i, id_j]
        # Since prior_adj is [Buckets, Buckets], and hashed_ids is [B, N]
        # This gather is tricky without consuming too much memory for large N.
        # Approximation: We only add prior bias to the attention logits
        
        # Since fully gathering prior for every pair is expensive (N*N),
        # we skip explicitly adding prior to the O(N^2) matrix in this efficient implementation
        # and rely on the fusion in get_adjacency_matrix for visualization/regularization,
        # OR we use a separate "Prior Embedding" path.
        
        # Refined Strategy: Use the merged adjacency for global reference, 
        # but for per-sample formulation keep using low-rank.
        # We constrain the low-rank components to be close to the prior?
        # Or we actally add a prior bias term?
        
        # Let's add Prior Bias for small window or specific important pairs?
        # For efficiency, we will proceed with the Learned Attention 
        # but regularize it towards the Prior in the loss function?
        # NO, user wants explicit KG injection.
        
        # Let's add a "Global Knowledge Aggregate" step
        # 1. Project nodes to Bucket Space
        # 2. Apply GCN on full Bucket Graph (N_Buckets x N_Buckets)
        # 3. Project back to Instance Space
        
        # Step 1: Aggregate features by bucket
        # mask: [B, N, Buckets] one-hot
        B, N, D = x.shape
        mask = F.one_hot(hashed_ids, num_classes=self.n_buckets).float() # [B, N, n_b]
        # Normalize mask
        mask_norm = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)
        
        # bucket_features: [B, n_b, D]
        bucket_features = torch.bmm(mask.transpose(1, 2), x)
        bucket_features = bucket_features / (mask.sum(dim=1, keepdim=True).transpose(1, 2) + 1e-8)
        
        # Step 2: Apply GCN on bucket graph
        adj = self.get_adjacency_matrix() # [n_b, n_b]
        # 正确的图卷积: output[b, i, d] = sum_j adj[i,j] * bucket_features[b, j, d]
        # 即 A @ X: [n_b, n_b] @ [n_b, D] for each batch -> einsum
        global_context = torch.einsum('ij,bjd->bid', adj, bucket_features)  # [B, n_b, D]
        
        # Step 3: Distribute back to nodes
        # [B, N, D] = [B, N, n_b] @ [B, n_b, D]
        context_features = torch.bmm(mask, global_context)
        
        # Fuse with local features
        h = x + context_features
        
        # Apply nonlinear layers
        for i, layer in enumerate(self.gnn_layers):
            h_new = layer(h)
            h = h + h_new # Simple residual
        
        return self.out_proj(h)


# =============================================================================
# Component 3a: Original Gated Cross-Modal Fusion (V5, keep for checkpoint compat)
# =============================================================================

class GatedCrossModalFusion(nn.Module):
    """
    Bidirectional Gated Cross-Attention Fusion (original V5 version).
    Kept for backward compatibility with V5 checkpoints.
    """
    def __init__(self, d_model: int = 128, static_dim: int = 18, 
                 note_dim: int = 768, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout)
        )
        self.note_encoder = nn.Sequential(
            nn.Linear(note_dim, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout)
        )
        self.cross_attn_t_sn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn_sn_t = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.global_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.recent_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.temporal_pool = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.recent_pool = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.static_query_t = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.notes_query_t = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.temporal_core = nn.Sequential(
            nn.Linear(d_model * 7, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.residual_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.final_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout)
        )
        self.temporal_bypass_scale = 0.10

    def forward(self, temporal: torch.Tensor, static: torch.Tensor, notes: torch.Tensor):
        s_emb = self.static_encoder(static).unsqueeze(1)
        n_emb = self.note_encoder(notes).unsqueeze(1)
        context_sn = torch.cat([s_emb, n_emb], dim=1)
        t_enhanced, _ = self.cross_attn_t_sn(temporal, context_sn, context_sn)
        t_cross_pool = t_enhanced.mean(dim=1)
        t_orig_pool = temporal.mean(dim=1)
        t_peak_pool = temporal.max(dim=1)[0]

        recent_len = max(4, temporal.size(1) // 4)
        recent_tokens = temporal[:, -recent_len:, :]
        global_ctx, _ = self.temporal_pool(self.global_query.expand(temporal.size(0), -1, -1), temporal, temporal)
        recent_ctx, _ = self.recent_pool(self.recent_query.expand(temporal.size(0), -1, -1), recent_tokens, recent_tokens)

        s_from_t, _ = self.static_query_t(s_emb, temporal, temporal)
        n_from_t, _ = self.notes_query_t(n_emb, temporal, temporal)
        sn_enhanced, _ = self.cross_attn_sn_t(context_sn, temporal, temporal)
        s_enhanced = 0.5 * (sn_enhanced[:, 0, :] + s_from_t.squeeze(1))
        n_enhanced = 0.5 * (sn_enhanced[:, 1, :] + n_from_t.squeeze(1))

        temporal_core = self.temporal_core(
            torch.cat(
                [
                    t_orig_pool,
                    t_cross_pool,
                    global_ctx.squeeze(1),
                    recent_ctx.squeeze(1),
                    t_peak_pool,
                    s_enhanced,
                    n_enhanced,
                ],
                dim=-1,
            )
        )
        static_notes_residual = self.residual_proj(torch.cat([s_emb.squeeze(1), n_emb.squeeze(1)], dim=-1))
        residual_scale = 0.10 + 0.20 * self.residual_gate(torch.cat([temporal_core, static_notes_residual], dim=-1))
        fused = temporal_core + residual_scale * static_notes_residual
        fused = self.final_proj(fused)
        return fused + self.temporal_bypass_scale * t_orig_pool


# =============================================================================
# Component 3b: BidirectionalCrossAttention — one layer of 3-modal interaction
# =============================================================================

class BidirectionalCrossAttention(nn.Module):
    """
    单层三模态双向交叉注意力：
      T ← (S, N)   S ← (T, N)   N ← (T, S)
    残差连接 + LayerNorm + 前馈网络
    """
    def __init__(self, d_model: int = 128, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        make_ca = lambda: nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # T 从 S+N 获取信息
        self.ca_t_from_sn = make_ca()
        # S 从 T+N 获取信息
        self.ca_s_from_tn = make_ca()
        # N 从 T+S 获取信息
        self.ca_n_from_ts = make_ca()

        make_ff = lambda: nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ff_t = make_ff(); self.ff_s = make_ff(); self.ff_n = make_ff()

        make_ln = lambda: nn.LayerNorm(d_model)
        self.ln_t1 = make_ln(); self.ln_t2 = make_ln()
        self.ln_s1 = make_ln(); self.ln_s2 = make_ln()
        self.ln_n1 = make_ln(); self.ln_n2 = make_ln()

    def forward(self, t: torch.Tensor, s: torch.Tensor, n: torch.Tensor):
        # t/s/n: [B, D] — expand to [B, 1, D] for MHA
        t_ = t.unsqueeze(1); s_ = s.unsqueeze(1); n_ = n.unsqueeze(1)

        # t ← s, n
        sn = torch.cat([s_, n_], dim=1)
        t_out, _ = self.ca_t_from_sn(t_, sn, sn)
        t2 = self.ln_t1(t + t_out.squeeze(1))
        t2 = self.ln_t2(t2 + self.ff_t(t2))

        # s ← t, n
        tn = torch.cat([t_, n_], dim=1)
        s_out, _ = self.ca_s_from_tn(s_, tn, tn)
        s2 = self.ln_s1(s + s_out.squeeze(1))
        s2 = self.ln_s2(s2 + self.ff_s(s2))

        # n ← t, s
        ts = torch.cat([t_, s_], dim=1)
        n_out, _ = self.ca_n_from_ts(n_, ts, ts)
        n2 = self.ln_n1(n + n_out.squeeze(1))
        n2 = self.ln_n2(n2 + self.ff_n(n2))

        return t2, s2, n2


# =============================================================================
# Component 3c: GatedCrossModalFusionV2 — upgraded, multi-layer, 3-way gate
# =============================================================================

class GatedCrossModalFusionV2(nn.Module):
    """
    门控多头交叉注意力融合 V2
    ─────────────────────────────────────────────
    • 加深编码器（2×MLP + LayerNorm）
    • n_fusion_layers 层 BidirectionalCrossAttention 堆叠
    • 3-way adaptive gate：softmax 学习 T/S/N 的相对贡献权重
    • 最终 MLP 融合
    """
    def __init__(self, d_model: int = 128, static_dim: int = 18,
                 note_dim: int = 768, n_heads: int = 8,
                 n_fusion_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        # ── 加深各模态编码器 ──
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        self.note_encoder = nn.Sequential(
            nn.Linear(note_dim, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )

        # temporal 池化前投影
        self.temporal_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(),
        )

        # ── 多层双向交叉注意力 ──
        self.cross_layers = nn.ModuleList([
            BidirectionalCrossAttention(d_model, n_heads, dropout)
            for _ in range(n_fusion_layers)
        ])

        # ── 自适应三路门控 ──
        self.modality_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(),
            nn.Linear(d_model, 3), nn.Softmax(dim=-1),
        )

        # ── 最终融合 MLP ──
        self.final_fusion = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, temporal: torch.Tensor, static: torch.Tensor, notes: torch.Tensor):
        # temporal: [B, Seq, D]  →  mean-pooled [B, D]
        t = self.temporal_proj(temporal.mean(dim=1))
        s = self.static_encoder(static)
        n = self.note_encoder(notes)

        # 多层双向交叉注意力（迭代精炼）
        for layer in self.cross_layers:
            t, s, n = layer(t, s, n)

        # 三路自适应门控
        gate_input = torch.cat([t, s, n], dim=-1)            # [B, 3D]
        weights = self.modality_gate(gate_input)              # [B, 3]
        fused = (weights[:, 0:1] * t +
                 weights[:, 1:2] * s +
                 weights[:, 2:3] * n)                        # [B, D]

        return self.final_fusion(fused)

# =============================================================================
# Main Model: ECDT V5
# =============================================================================

class EnhancedCausalDigitalTwinV5(nn.Module):
    def __init__(self, n_outputs: int = 6, d_model: int = 128, 
                 n_buckets: int = 512, n_channels: int = 3,
                 n_transformer_layers: int = 4, dropout: float = 0.2, # Deeper Transformer
                 use_diffusion: bool = True, use_gnn: bool = True,
                 static_dim: int = 18, note_dim: int = 768,
                 use_domain_adaptation: bool = True,
                 use_activation_checkpointing: bool = False):
        super().__init__()
        self.d_model = d_model
        self.use_activation_checkpointing = use_activation_checkpointing
        
        # 0. Components (Reusing efficient V3/V4 components)
        self.static_tokenizer = FeatureTokenizer(static_dim, d_model)
        
        self.sparse_processor = SparseFeatureProcessor(
            d_model=d_model, n_buckets=n_buckets, 
            n_channels=n_channels, dropout=dropout
        )
        
        # 1. New Knowledge GNN
        if use_gnn:
            self.feature_gnn = KnowledgeGNN(
                d_model=d_model, n_buckets=n_buckets,
                n_layers=3, dropout=dropout
            )
        else:
            self.feature_gnn = None
            
        # 2. Diffusion (Same as V3)
        self.use_diffusion = use_diffusion
        if use_diffusion:
            self.diffusion = DiffusionAugmentation(d_model=d_model, n_steps=20)
            
        # 3. Temporal Transformer (V3 Efficient Version is good)
        self.temporal_transformer = EfficientTemporalTransformer(
            d_model=d_model, n_heads=4, n_layers=n_transformer_layers,
            dropout=dropout
        )
        
        # 4. New Gated Fusion
        self.fusion = GatedCrossModalFusion(
            d_model=d_model, static_dim=static_dim, note_dim=note_dim,
            dropout=dropout
        )
        
        # 5. Label Correlation (V3)
        self.label_correlation = LabelCorrelationModule(n_outputs, d_model, dropout=dropout)
        
        # 6. Heads
        # Main Task Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_outputs)
        )
        
        # Domain Adaptation Head
        self.use_domain_adaptation = use_domain_adaptation
        if use_domain_adaptation:
            self.domain_classifier = nn.Sequential(
                GradientReversalLayer(alpha=1.0),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1) # Binary: Source vs Target
            )
            
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _maybe_checkpoint(self, fn, *args):
        if not (self.training and self.use_activation_checkpointing):
            return fn(*args)
        tensor_args = [arg for arg in args if torch.is_tensor(arg)]
        if not tensor_args:
            return fn(*args)
        return activation_checkpoint(fn, *args, use_reentrant=False)

    def forward(self, events: torch.Tensor, event_mask: torch.Tensor,
                static: Optional[torch.Tensor] = None,
                notes: Optional[torch.Tensor] = None,
                stochastic: bool = False,
                return_domain_logits: bool = False) -> Dict[str, torch.Tensor]:
        
        B = events.size(0)
        device = events.device
        
        if static is None: static = torch.zeros(B, 18, device=device)
        if notes is None: notes = torch.zeros(B, 768, device=device)
        
        # --- Feature Processing ---
        x = self.sparse_processor(events) # [B, N, D]
        
        # --- Knowledge GNN ---
        if self.feature_gnn is not None:
            feature_ids = events[:, :, 2].long()
            x = x + self._maybe_checkpoint(lambda a, b: self.feature_gnn(a, b), x, feature_ids)
            
        # --- Diffusion ---
        if self.use_diffusion:
            x = self._maybe_checkpoint(
                lambda a: self.diffusion(a, training=self.training, stochastic=stochastic),
                x
            )
            
        # --- Temporal Modeling ---
        temporal_repr = self._maybe_checkpoint(
            lambda a, b: self.temporal_transformer(a, b),
            x,
            event_mask
        ) # [B, 32, D]
        
        # --- Multimodal Fusion ---
        fused = self._maybe_checkpoint(
            lambda a, b, c: self.fusion(a, b, c),
            temporal_repr,
            static,
            notes
        ) # [B, D]
        
        # --- Predictions ---
        logits = self.classifier(fused)
        label_corr = self.label_correlation(fused)
        
        # Ensemble predictions (Simple average for stability, or learned weight)
        final_logits = logits + 0.2 * label_corr
        
        out = {
            'factual_logits': final_logits,
            'fused_repr': fused
        }
        
        # --- Domain Adaptation ---
        if self.use_domain_adaptation and return_domain_logits:
            domain_logits = self.domain_classifier(fused)
            out['domain_logits'] = domain_logits
            
        return out


# =============================================================================
# ECDT-V6: GatedCrossModalFusionV2 + 进阶域适应
# =============================================================================

class EnhancedCausalDigitalTwinV6(nn.Module):
    """
    ECDT V6 — 全量升级版
    ─────────────────────────────────────────────────────
    升级点 vs V5：
    1. GatedCrossModalFusionV2（多层双向交叉注意力 + 三路门控）
    2. n_heads=8, n_fusion_layers=2
    3. Progressive GRL alpha（初始小，随训练步数增大）
    4. 分类头增加 skip-connection
    5. 静态分支额外 2-layer MLP（已含在 FusionV2 的 static_encoder 中）
    """
    def __init__(self, n_outputs: int = 6, d_model: int = 128,
                 n_buckets: int = 512, n_channels: int = 3,
                 n_transformer_layers: int = 4, dropout: float = 0.2,
                 use_diffusion: bool = True, use_gnn: bool = True,
                 static_dim: int = 18, note_dim: int = 768,
                 use_domain_adaptation: bool = True,
                 n_fusion_layers: int = 2, n_heads: int = 8,
                 grl_alpha: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self._grl_alpha = grl_alpha     # 可在训练中动态更新

        # ── 共享组件 ──
        self.static_tokenizer  = FeatureTokenizer(static_dim, d_model)
        self.sparse_processor  = SparseFeatureProcessor(
            d_model=d_model, n_buckets=n_buckets,
            n_channels=n_channels, dropout=dropout)
        self.feature_gnn = KnowledgeGNN(
            d_model=d_model, n_buckets=n_buckets,
            n_layers=3, dropout=dropout) if use_gnn else None

        self.use_diffusion = use_diffusion
        if use_diffusion:
            self.diffusion = DiffusionAugmentation(d_model=d_model, n_steps=20)

        self.temporal_transformer = EfficientTemporalTransformer(
            d_model=d_model, n_heads=4, n_layers=n_transformer_layers,
            dropout=dropout)

        # ── V2 融合 (核心升级) ──
        self.fusion = GatedCrossModalFusionV2(
            d_model=d_model, static_dim=static_dim, note_dim=note_dim,
            n_heads=n_heads, n_fusion_layers=n_fusion_layers, dropout=dropout)

        # ── 标签相关模块 ──
        self.label_correlation = LabelCorrelationModule(n_outputs, d_model, dropout=dropout)

        # ── 分类头（带 skip） ──
        self.skip_proj = nn.Linear(d_model, d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_outputs)
        )

        # ── 域适应（可进阶 alpha） ──
        self.use_domain_adaptation = use_domain_adaptation
        if use_domain_adaptation:
            self.domain_grl = GradientReversalLayer(alpha=grl_alpha)
            self.domain_disc = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model // 4), nn.ReLU(),
                nn.Linear(d_model // 4, 1)
            )

        self._init_weights()

    def set_grl_alpha(self, alpha: float):
        """在训练中逐步增大 GRL 强度（渐进式域适应）"""
        self._grl_alpha = alpha
        if self.use_domain_adaptation:
            self.domain_grl.alpha = alpha

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, events: torch.Tensor, event_mask: torch.Tensor,
                static: Optional[torch.Tensor] = None,
                notes: Optional[torch.Tensor] = None,
                stochastic: bool = False,
                return_domain_logits: bool = False) -> Dict[str, torch.Tensor]:
        B, device = events.size(0), events.device
        if static is None: static = torch.zeros(B, 18, device=device)
        if notes  is None: notes  = torch.zeros(B, 768, device=device)

        x = self.sparse_processor(events)
        if self.feature_gnn is not None:
            x = x + self.feature_gnn(x, events[:, :, 2].long())
        if self.use_diffusion:
            x = self.diffusion(x, training=self.training, stochastic=stochastic)

        temporal_repr = self.temporal_transformer(x, event_mask)
        fused = self.fusion(temporal_repr, static, notes)            # [B, D]

        # 带 skip 的分类头
        skip = self.skip_proj(fused)
        logits      = self.classifier(fused + skip)
        label_corr  = self.label_correlation(fused)
        final_logits = logits + 0.2 * label_corr

        out = {'factual_logits': final_logits, 'fused_repr': fused}

        if self.use_domain_adaptation and return_domain_logits:
            rev_fused = self.domain_grl(fused)
            out['domain_logits'] = self.domain_disc(rev_fused)

        return out


# =============================================================================
# ECDT-V6-Lite: 轻量版——适合数据少或快速验证
# =============================================================================

class EnhancedCausalDigitalTwinV6Lite(nn.Module):
    """
    ECDT V6-Lite — 轻量对比版
    ─────────────────────────────────────────────────────
    设计思路：同等 d_model=128，靠更少层数/更强正则化降低过拟合
    • 无 GNN（减少过拟合风险）
    • 无 Diffusion（数据增强已在外部完成）
    • 单层 BidirectionalCrossAttention + 3-way gate（n_fusion_layers=1）
    • n_heads=4, n_transformer_layers=2, dropout=0.3
    • 参数量约为 V6 的 60%
    适合比较：V6 vs V6-Lite，分析容量与正则化的 trade-off
    """
    def __init__(self, n_outputs: int = 6, d_model: int = 128,
                 n_buckets: int = 256, n_channels: int = 3,
                 n_transformer_layers: int = 2, dropout: float = 0.3,
                 static_dim: int = 18, note_dim: int = 768,
                 use_domain_adaptation: bool = True):
        super().__init__()
        self.d_model = d_model

        self.sparse_processor = SparseFeatureProcessor(
            d_model=d_model, n_buckets=n_buckets,
            n_channels=n_channels, dropout=dropout)
        self.temporal_transformer = EfficientTemporalTransformer(
            d_model=d_model, n_heads=4, n_layers=n_transformer_layers,
            dropout=dropout)

        # 单层双向交叉注意力融合
        self.fusion = GatedCrossModalFusionV2(
            d_model=d_model, static_dim=static_dim, note_dim=note_dim,
            n_heads=4, n_fusion_layers=1, dropout=dropout)

        self.label_correlation = LabelCorrelationModule(n_outputs, d_model, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, n_outputs)
        )

        self.use_domain_adaptation = use_domain_adaptation
        if use_domain_adaptation:
            self.domain_classifier = nn.Sequential(
                GradientReversalLayer(alpha=0.5),
                nn.Linear(d_model, d_model // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(d_model // 2, 1)
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, events: torch.Tensor, event_mask: torch.Tensor,
                static: Optional[torch.Tensor] = None,
                notes:  Optional[torch.Tensor] = None,
                stochastic: bool = False,
                return_domain_logits: bool = False) -> Dict[str, torch.Tensor]:
        B, device = events.size(0), events.device
        if static is None: static = torch.zeros(B, 18, device=device)
        if notes  is None: notes  = torch.zeros(B, 768, device=device)

        x = self.sparse_processor(events)
        temporal_repr = self.temporal_transformer(x, event_mask)
        fused = self.fusion(temporal_repr, static, notes)

        logits     = self.classifier(fused)
        label_corr = self.label_correlation(fused)
        final_logits = logits + 0.2 * label_corr

        out = {'factual_logits': final_logits, 'fused_repr': fused}
        if self.use_domain_adaptation and return_domain_logits:
            out['domain_logits'] = self.domain_classifier(fused)
        return out
