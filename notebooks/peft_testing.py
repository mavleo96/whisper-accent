import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Load model and apply LoRA
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
config = LoraConfig(
    r=8,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    # task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# Dummy input
batch = {
    "input_features": torch.randn(2, 80, 3000),  # shape: (batch, feature_dim, time)
    "attention_mask": torch.randint(0, 2, (2, 3000)),  # not always used in Whisper
    "labels": torch.randint(0, 51865, (2, 225)),  # vocab size is 51865
    "decoder_attention_mask": torch.randint(0, 2, (2, 225)),  # decoder attention
}

# Move model and data to the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
batch = {k: v.to(device) for k, v in batch.items()}

# 🔁 Forward pass (for training or loss computation)
with torch.no_grad():
    output = model(
        input_features=batch["input_features"],
        labels=batch["labels"],
        decoder_attention_mask=batch["decoder_attention_mask"],
    )

print("Loss:", output.loss.item())
print("Logits shape:", output.logits.shape)  # (batch_size, seq_len, vocab_size)

# 🧠 Generate method (for inference)
# Whisper expects decoder_start_token_id to be passed
if model.generation_config.decoder_start_token_id is None:
    model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id

with torch.no_grad():
    generated_ids = model.generate(
        input_features=batch["input_features"],
        decoder_attention_mask=batch["decoder_attention_mask"],
        max_new_tokens=50,
    )

print("Generated token IDs:", generated_ids)
