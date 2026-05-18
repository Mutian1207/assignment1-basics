import torch
import torch.nn as nn
import math
from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding
from einops import rearrange, einsum

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # ✅ 必须加这个 weight，测试会加载它！
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms
        return (x * self.weight).to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__() 
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w3(x)
        hidden = nn.functional.silu(x1) * x2

        return self.w2(hidden)
    
class RoPE(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device=None, dtype=None):
        super().__init__()
        # 1. 计算频率：θ_k = Θ^(2(k-1)/d_k)，k=1,...,d_k/2
        freqs = 1/ (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype).float() / d_k))
        print(freqs.shape)
        # 2. 计算位置编码：pos * θ_k
        # pos shape: (max_seq_len, 1)，freqs shape: (d_k/2, 1) -> (max_seq_len, d_k/2)
        pos = torch.arange(max_seq_len, device=device, dtype=dtype)
        print(pos.shape)
        theta_pos_k = torch.einsum('L, D -> LD', pos, freqs)
        print(theta_pos_k.shape)
        # 3. 将位置编码分成两部分：sin(θ_pos_k) 和 cos(θ_pos_k)
        self.register_buffer('sin', torch.sin(theta_pos_k), persistent=False)
        self.register_buffer('cos', torch.cos(theta_pos_k), persistent=False)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 🔥 修复：适配多头维度 (b, h, s, d)
        sin_pos = self.sin[token_positions].unsqueeze(1)
        cos_pos = self.cos[token_positions].unsqueeze(1)

        x1, x2 = x[..., ::2], x[..., 1::2]
        x1_rot = x1 * cos_pos - x2 * sin_pos
        x2_rot = x1 * sin_pos + x2 * cos_pos
        x_rot = torch.empty_like(x)
        x_rot[..., ::2] = x1_rot
        x_rot[..., 1::2] = x2_rot
        return x_rot
    

def softmax(x:torch.Tensor, i:int):
    if (i >= x.dim()):
        print(f"dim error {i} > {i.dim()}")
        return x
    
    max_val = x.max(dim = i, keepdim=True).values
    exp_x = torch.exp(x - max_val)
    sum_val = exp_x.sum(dim = i, keepdim=True)
    return exp_x / sum_val

class Attention(nn.Module):
    def __init__(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask = None):
        super().__init__()
        self.d_k = q.shape[-1]
        self.d_v = v.shape[-1]
        self.q = q
        self.k = k
        self.v = v
        self.mask = mask
    def forward(self):
        mask_score = torch.where(self.mask, 0.0, -float('inf'))
        k_trans = self.k.transpose(-2,-1)
        pre_softmax_val = torch.matmul(self.q, k_trans)/math.sqrt(self.d_k)
        pre_softmax_val += mask_score
        softmax_query_mat = softmax(pre_softmax_val, -1)
        attention = torch.matmul(softmax_query_mat, self.v)
        return attention
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=0, theta=0.0, token_positions=None):
        super().__init__()
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.d_model = d_model
        self.num_heads = num_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        # RoPE 初始化
        self.rope = None
        if theta > 0:
            self.rope = RoPE(theta, self.d_k, max_seq_len)

        # 可选传入的位置
        self.token_positions = token_positions

    def forward(self, x):
        # QKV 投影
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # 拆多头
        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)
        
        if self.token_positions is None:
            # 不传 → 自动生成
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        else:
            # 传入 → 使用外部给的
            token_positions = self.token_positions

        # RoPE
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # 因果掩码
        seq_len = Q.shape[-2]
        causal_mask = torch.tril(torch.ones(
            seq_len, seq_len, dtype=torch.bool, device=x.device
        ))

        # 注意力计算
        attn = Attention(Q, K, V, causal_mask)
        attn_out = attn.forward()

        # 合并多头
        out = rearrange(attn_out, "b h s d -> b s (h d)")
        return self.Wo(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=0, theta=0.0):
        super().__init__()
        # 注意力层（支持 RoPE）
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta
        )
        
        self.norm1 = RMSNorm(d_model)  # 给 Attention
        self.norm2 = RMSNorm(d_model)  # 给 FFN
        self.ffn = SwiGLU(d_model, d_ff)
    
    def _attention_layer(self, x):
        norm_val = self.norm1.forward(x)
        return x + self.attn.forward(norm_val)
    
    def _activation_layer(self, x):
        norm_val = self.norm2.forward(x)
        return x + self.ffn.forward(norm_val)
    
    def forward(self, x):
        att_x = self._attention_layer(x)
        return self._activation_layer(att_x)

class TransformerLM(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, vocab_size, context_length, num_layers, theta=0.0):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                theta
            )
            for _ in range(num_layers)
        ])
        # layer norm
        self.ln_final = RMSNorm(d_model)
        # linear
        self.linear_final = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, in_indices):
        # in_indices: (batch_size, sequence_length)

        # 1. 词嵌入
        x = self.embed(in_indices)  # (B, S, d_model)
        
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)

        logits = self.linear_final(x)

        return logits
    
def cross_entropy_loss(logits, target_indices):
    # logits: (B, S, vocab_size)
    # target_indices: (B, S)
    max_logits = logits.max(dim = -1, keepdim=True).values
    logits_stable = logits - max_logits
    exp_logits = torch.exp(logits_stable)
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    target_logit = torch.gather(logits_stable, dim=-1, index=target_indices.unsqueeze(-1)).squeeze(-1)
    loss = torch.log(sum_exp) - target_logit
    loss = loss.mean()
    return loss