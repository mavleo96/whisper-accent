import torch
import torch.nn as nn
import torch.nn.functional as F

encoder_dimension_size = {
    "openai/whisper-tiny": 384,
    "openai/whisper-tiny.en": 384,
    "openai/whisper-base": 512,
    "openai/whisper-base.en": 512,
    "openai/whisper-small": 768,
    "openai/whisper-small.en": 768,
    "openai/whisper-medium": 1024,
    "openai/whisper-medium.en": 1024,
    "openai/whisper-large": 1280,
    "openai/whisper-large.en": 1280,
}


class AccentCodebook(nn.Module):
    def __init__(self, num_embeddings, dim_size):
        super(AccentCodebook, self).__init__()
        # Use Embedding layer instead of raw Parameter for safer initialization
        self.codebook = nn.Embedding(num_embeddings, dim_size)
        # Initialize with small values
        nn.init.uniform_(self.codebook.weight, -0.02, 0.02)

        # Simple attention mechanism with linear projections
        self.query_proj = nn.Linear(dim_size, dim_size)
        self.key_proj = nn.Linear(dim_size, dim_size)
        self.value_proj = nn.Linear(dim_size, dim_size)
        self.out_proj = nn.Linear(dim_size, dim_size)
        self.dropout = nn.Dropout(0.1)

        # Initialize layer norm with proper scaling
        self.layer_norm = nn.LayerNorm(dim_size, eps=1e-5)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        # expand codebook to batch size
        if x.ndim == 3:
            B, _, _ = x.shape
            # Get codebook vectors by passing indices
            indices = torch.arange(self.codebook.num_embeddings, device=x.device)
            codebook = self.codebook(indices).unsqueeze(0).expand(B, -1, -1)
        else:
            indices = torch.arange(self.codebook.num_embeddings, device=x.device)
            codebook = self.codebook(indices)

        print("DEBUG: x shape", x.shape)
        print("DEBUG: x dtype", x.dtype)
        print("DEBUG: x nan", torch.isnan(x).sum() / x.numel())
        print("DEBUG: x inf", torch.isinf(x).sum() / x.numel())
        print(
            "DEBUG: x stats",
            x.min().item(),
            x.max().item(),
            x.mean().item(),
            x.std().item(),
        )
        print("DEBUG: codebook shape", codebook.shape)
        print("DEBUG: codebook dtype", codebook.dtype)
        print("DEBUG: codebook nan", torch.isnan(codebook).sum() / codebook.numel())
        print(
            "DEBUG: codebook stats",
            codebook.min().item(),
            codebook.max().item(),
            codebook.mean().item(),
            codebook.std().item(),
        )

        # Project queries, keys, and values
        q = self.query_proj(x)  # [B, L, D] or [L, D]
        k = self.key_proj(codebook)  # [B, N, D] or [N, D]
        v = self.value_proj(codebook)  # [B, N, D] or [N, D]

        print("DEBUG: q shape", q.shape)
        print("DEBUG: k shape", k.shape)
        print("DEBUG: v shape", v.shape)
        print("DEBUG: q nan", torch.isnan(q).sum() / q.numel())
        print("DEBUG: q inf", torch.isinf(q).sum() / q.numel())
        print(
            "DEBUG: q stats",
            q.min().item(),
            q.max().item(),
            q.mean().item(),
            q.std().item(),
        )
        print("DEBUG: k nan", torch.isnan(k).sum() / k.numel())
        print("DEBUG: k inf", torch.isinf(k).sum() / k.numel())
        print(
            "DEBUG: k stats",
            k.min().item(),
            k.max().item(),
            k.mean().item(),
            k.std().item(),
        )
        print("DEBUG: v nan", torch.isnan(v).sum() / v.numel())
        print("DEBUG: v inf", torch.isinf(v).sum() / v.numel())
        print(
            "DEBUG: v stats",
            v.min().item(),
            v.max().item(),
            v.mean().item(),
            v.std().item(),
        )

        # Compute attention scores with scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        print("DEBUG: attn_weights shape", attn_weights.shape)
        print(
            "DEBUG: attn_weights nan",
            torch.isnan(attn_weights).sum() / attn_weights.numel(),
        )
        print(
            "DEBUG: attn_weights inf",
            torch.isinf(attn_weights).sum() / attn_weights.numel(),
        )
        print(
            "DEBUG: attn_weights nan",
            torch.isnan(attn_weights).sum() / attn_weights.numel(),
        )
        print(
            "DEBUG: attn_weights inf",
            torch.isinf(attn_weights).sum() / attn_weights.numel(),
        )
        print(
            "DEBUG: attn_weights stats",
            attn_weights.min().item(),
            attn_weights.max().item(),
            attn_weights.mean().item(),
            attn_weights.std().item(),
        )

        # Apply attention
        output = torch.matmul(attn_weights, v)
        output = self.out_proj(output)

        print("DEBUG: output shape", output.shape)
        print("DEBUG: output nan", torch.isnan(output).sum() / output.numel())
        print("DEBUG: output inf", torch.isinf(output).sum() / output.numel())
        print(
            "DEBUG: output stats",
            output.min().item(),
            output.max().item(),
            output.mean().item(),
            output.std().item(),
        )

        # Add residual connection with layer norm
        final_output = self.layer_norm(x + output)
        # final_output = x + output
        print("DEBUG: final_output shape", final_output.shape)
        print(
            "DEBUG: final_output nan",
            torch.isnan(final_output).sum() / final_output.numel(),
        )
        print(
            "DEBUG: final_output inf",
            torch.isinf(final_output).sum() / final_output.numel(),
        )
        print(
            "DEBUG: final_output stats",
            final_output.min().item(),
            final_output.max().item(),
            final_output.mean().item(),
            final_output.std().item(),
        )

        return final_output, attn_weights


if __name__ == "__main__":
    from hydra import compose, initialize
    from transformers.models.whisper.modeling_whisper import WhisperConfig

    initialize(config_path="../../../configs", version_base=None)
    cfg = compose(
        config_name="baseline_eval.yaml",
        overrides=["model.model_name=openai/whisper-tiny"],
    )

    # TODO: pass configs correctly after refactoring
    dim_size = WhisperConfig.from_pretrained(cfg.model.model_name).d_model
    codebook = AccentCodebook(100, dim_size)

    # Test with batch size
    hidden_state = torch.randn(cfg.data.batch_size, 1500, dim_size)
    output, attn_weights = codebook(hidden_state)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attn_weights.shape)

    # Test without batch size
    hidden_state = torch.randn(1500, dim_size)
    output, attn_weights = codebook(hidden_state)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attn_weights.shape)
    print("nan in output", torch.isnan(output).sum() / output.numel())
    print("nan in attentions", torch.isnan(attn_weights).sum() / attn_weights.numel())
