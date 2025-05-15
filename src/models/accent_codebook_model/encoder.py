from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer


class WhisperEncoderLayerWithCodebook(WhisperEncoderLayer):
    def __init__(self, config, codebook):
        super().__init__(config)
        self.codebook = codebook

    def forward(
        self, hidden_states, attention_mask, layer_head_mask, output_attentions
    ):
        encoder_output = super().forward(
            hidden_states, attention_mask, layer_head_mask, output_attentions
        )

        if output_attentions:
            hidden_states, encoder_attentions = encoder_output
        else:
            hidden_states = encoder_output[0]
            encoder_attentions = None

        hidden_states, _ = self.codebook(hidden_states)

        if output_attentions:
            return hidden_states, encoder_attentions
        return (hidden_states,)


if __name__ == "__main__":
    import torch
    from hydra import compose, initialize
    from transformers import WhisperConfig

    from .codebook import AccentCodebook, encoder_dimension_size

    # TODO: pass configs correctly after refactoring
    with initialize(config_path="../../../configs", version_base=None):
        cfg = compose(
            config_name="baseline_eval.yaml",
            overrides=["model.model_name=openai/whisper-tiny"],
        )

    config = WhisperConfig.from_pretrained(cfg.model.model_name)
    hidden_size = encoder_dimension_size[cfg.model.model_name]
    codebook = AccentCodebook(5, hidden_size)
    layer = WhisperEncoderLayerWithCodebook(config, codebook)

    # Create test inputs
    hidden_states = torch.randn(cfg.data.batch_size, 1500, hidden_size)
    attention_mask = torch.ones(cfg.data.batch_size, 1, 1500, 1500)

    output, attentions = layer(hidden_states, attention_mask, None, True)
    print(f"Output shape: {output.shape}")
    print(f"Encoder attention shape: {attentions[0].shape}")
    print(f"Codebook attention shape: {attentions[1].shape}")
    print("nan in output", torch.isnan(output).sum() / output.numel())
    print("nan in attentions", torch.isnan(attentions[0]).sum() / attentions[0].numel())
    print("nan in attentions", torch.isnan(attentions[1]).sum() / attentions[1].numel())
