import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.constants import NUM_ACCENTS


class WhisperWithAccentToken(WhisperForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        print(
            "Warning: direct initialization of WhisperWithAccentToken is not implemented. Use from_pretrained instead."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        accent_detection_decoder_input_ids,
        accent_token_id_map,
        *model_args,
        **kwargs,
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model.model.decoder.resize_token_embeddings(
            model.config.vocab_size + NUM_ACCENTS
        )
        model.accent_detection_decoder_input_ids = accent_detection_decoder_input_ids
        model.accent_token_id_map = accent_token_id_map
        return model

    def detect_accent(self, input_features, attention_mask):
        B = input_features.size(0)
        L = self.accent_detection_decoder_input_ids.size(0)
        decoder_input_ids = self.accent_detection_decoder_input_ids.repeat(B, 1)
        decoder_input_ids = decoder_input_ids.to(input_features.device)

        output = self.forward(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        accent_logits = output.logits[:, L - 1, list(self.accent_token_id_map.values())]
        # Get predicted accent indices (0 to NUM_ACCENTS-1)
        predicted_accent_indices = accent_logits.argmax(dim=-1)
        # Convert to accent token IDs for decoder input
        predicted_accent_token_ids = torch.tensor(
            [self.accent_token_id_map[idx.item()] for idx in predicted_accent_indices],
            device=self.device,
        ).unsqueeze(1)
        return predicted_accent_token_ids

    def generate(
        self, input_features, attention_mask, decoder_input_ids=None, **kwargs
    ):
        if decoder_input_ids is None:
            # Get accent token IDs for decoder input
            accent_token_ids = self.detect_accent(
                input_features=input_features,
                attention_mask=attention_mask,
            )
            B = input_features.size(0)
            decoder_input_ids = self.accent_detection_decoder_input_ids.repeat(B, 1)
            decoder_input_ids = decoder_input_ids.to(input_features.device)
            decoder_input_ids = torch.cat([decoder_input_ids, accent_token_ids], dim=1)

        output = super().generate(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            **kwargs,
        )

        accent_token_ids = decoder_input_ids[:, -1]
        accent_indices = torch.tensor(
            [
                list(self.accent_token_id_map.keys())[
                    list(self.accent_token_id_map.values()).index(token_id.item())
                ]
                for token_id in accent_token_ids
            ],
            device=accent_token_ids.device,
        )

        return {
            "generated_text": output,
            "accent_ids": accent_indices,
        }


if __name__ == "__main__":
    processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [f"<|accent{i}|>" for i in range(NUM_ACCENTS)]}
    )
    accent_detection_decoder_input_ids = torch.tensor(
        [
            processor.tokenizer.encode("<|startoftranscript|>")[0],
            processor.tokenizer.encode("<|en|>")[0],
            processor.tokenizer.encode("<|transcribe|>")[0],
            # processor.tokenizer.encode("<|notimestamps|>")[0],
        ]
    )

    accent_token_id_map = {
        i: processor.tokenizer.convert_tokens_to_ids(f"<|accent{i}|>")
        for i in range(NUM_ACCENTS)
    }
    print(accent_token_id_map)

    model = WhisperWithAccentToken.from_pretrained(
        "openai/whisper-base.en",
        accent_detection_decoder_input_ids=accent_detection_decoder_input_ids,
        accent_token_id_map=accent_token_id_map,
    )

    if model.generation_config.is_multilingual:
        model.generation_config.language = "<|en|>"
        model.generation_config.task = "transcribe"

    input = {
        "input_features": torch.randn(4, 80, 3000),
        "attention_mask": torch.ones(4, 3000),
    }

    output = model.generate(**input)
    print(output)
    print(model.config.vocab_size)
