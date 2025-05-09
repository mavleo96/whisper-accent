from transformers import WhisperForConditionalGeneration

from .codebook import AccentCodebook
from .encoder import WhisperEncoderLayerWithCodebook

injection_positions = {
    4: [2, 3],
    6: [3, 4],
    12: [4, 6, 8],
    24: [6, 12, 18],
    32: [8, 16, 24],
}


class WhisperWithAccentCodebook(WhisperForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.codebook = AccentCodebook(5, config.d_model)

        for i in injection_positions[config.decoder_layers]:
            encoder_layer = WhisperEncoderLayerWithCodebook(config, self.codebook)
            encoder_layer.load_state_dict(
                self.model.encoder.layers[i].state_dict(), strict=False
            )
            self.model.encoder.layers[i] = encoder_layer


if __name__ == "__main__":
    import torch
    from hydra import compose, initialize
    from transformers import WhisperForConditionalGeneration

    with initialize(config_path="../../../configs", version_base=None):
        cfg = compose(
            config_name="baseline_eval.yaml", overrides=["data.batch_size=32"]
        )

    model = WhisperWithAccentCodebook.from_pretrained(cfg.model.model_name).to("cuda")

    # Create input values in Whisper's expected format (mel spectrograms)
    # Whisper expects [batch_size, n_mels, time]
    input_values = torch.randn(cfg.data.batch_size, 80, 3000).to("cuda")
    attention_mask = torch.randint(0, 2, (cfg.data.batch_size, 3000)).to("cuda")
    labels = torch.randint(
        0, model.config.vocab_size, (cfg.data.batch_size, cfg.data.max_length)
    ).to("cuda")
    decoder_attention_mask = torch.randint(
        0, 2, (cfg.data.batch_size, cfg.data.max_length)
    ).to("cuda")

    output = model(
        input_values,
        attention_mask=attention_mask,
        labels=labels,
        decoder_attention_mask=decoder_attention_mask,
    )
    model.eval()
    print(output.logits)
