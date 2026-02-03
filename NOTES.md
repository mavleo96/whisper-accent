1. Accent Tokens
- [DONE] _retrieve_init_tokens -> detect_accent -> add before timestamps token
- [DONE] change this to insert before timestamps token
- [DONE] Forward method -> should not use this

2. Tokenizer
- [DONE] add accent tokens to prefix tokens

3. Data Module
- [DONE] BOS token is added inside forward method; data collator should skip this in train loader
- [DONE] Labels should be padded with -100 in the data collator
- [DONE] Can maybe skip decoder attention mask in the data collator
- [DONE] https://huggingface.co/datasets/westbrook/English_Accent_DataSet
- [DONE] Save normalizer.json in the checkpoint

4. Trainer
- Args:
  - [DONE] Training args: add lambda_accent_loss & lambda_diversity_loss
  - [DONE] Dataset args: data_path, num_proc, shuffle
  - [DONE] Push to hub; strategy all checkpoints; hf argument parser
- Trainer:
  - [DONE] subclass Seq2SeqTrainer and override compute loss method (to log each loss separately); not possible
  - [DONE] override create optimizer and scheduler method (to use separate learning rates for embedding and linear layers)
  - [DONE] embedding weight decay should be set to 0
  - [DONE] override compute metrics method (to compute wer & accent accuracy)
  - callback to compute final wer overall and per accent
  - [DONE] need to do retrieve_init_tokens for accent accuracy since generate prediction does not return init_tokens
  - [DONE] train logging; losses: loss, transcription, accent, embedding_diversity
  - [DONE] eval logging; losses: loss, transcription, accent, / metrics: wer, accent accuracy
  - merge and unload model after training as final model
  - AutoModel ; if base model is openai/whisper then loading won't work because of class mismatch; if adapter_config.json is present then loading will not work
    - solution: remove adapter files and save full model
- Training Phases:
  - [DONE] Batch size: 8 x 4 x 1 = 32
  - [DONE] Steps 10K; prev runs were 2K steps
  - tiny is too small; we need to use atleast medium or largev3
  - might need to pretrain to learn english accent embeddings (actually american/canadian)
  - [DONE] optimizer: separate learning rates; embedding learning rate needs to start high and decay fast
  - lower case is correct preprocessing; accent token position also sensible;
- Model:
  - max_length: 448 vs 255
  - [DONE] proj_out: ideal to exclude from LoRA training; proj_out is tied to decoder.embed_tokens; resize token embeddings weight tying is not working


RUNS TO DO:
- Test Runs:
  - Check whisper finetuning on medium.en 2000, 5000
  - Check accent finetuning on accent-medium.en 2000, 5000
  - medium.en 10000
- Ablations (one size medium or large-v3):
  - baseline
  - accent model
  - accent model + accent loss
  - accent model + diversity loss
  - accent model + accent loss + diversity loss
- Training: Westbrook dataset
  - accent-medium
  - accent-medium.en
  - accent-large-v3
- Evaluation: Westbrook dataset
  - tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large, large-v2, large-v3
  - accent-medium, accent-medium.en, accent-large-v3


Overfitting:
- https://huggingface.co/spaces/openai/whisper/discussions/100
- https://github.com/huggingface/community-events/issues/197
- Increase dropout and reduce learning rate & lora rank
- Training Phase:
  - 1000 steps learning the accent embedding 1e-4
  - 2000 steps learning the model 1e-5
