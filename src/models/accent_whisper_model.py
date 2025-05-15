import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.whisper.modeling_whisper import shift_tokens_right
from src.constants import NUM_ACCENTS


class AccentWhisperModel(WhisperForConditionalGeneration):
    """
    Whisper model with accent codebook for accent-aware speech recognition.
    Inherits from WhisperForConditionalGeneration and adds accent-specific processing.
    """
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.__class__ = cls
        
        model.accent_codebook = nn.Parameter(
            torch.randn(NUM_ACCENTS, model.config.d_model)
        )
        model.cross_attention = nn.MultiheadAttention(
            embed_dim=model.config.d_model,
            num_heads=model.config.encoder_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Initialize weights
        model._init_accent_weights()
        
        return model
    
    def __init__(
        self,
        config: WhisperConfig,
    ):
        super().__init__(config)
        
        # Initialize accent codebook
        self.accent_codebook = nn.Parameter(
            torch.randn(NUM_ACCENTS, config.d_model)
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.encoder_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self._init_accent_weights()
    
    def _init_accent_weights(self):
        nn.init.xavier_uniform_(self.accent_codebook)
    
    def forward(
        self,
        input_features,
        attention_mask=None,
        output_attentions=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        accent_ids=None,
        labels=None,
        **kwargs
    ):
        # Get encoder outputs
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        encoder_outputs = self.model.encoder(input_features, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = encoder_outputs.last_hidden_state
        
        # Get accent embeddings
        if accent_ids is None:
            accent_ids = torch.zeros(input_features.size(0), dtype=torch.long, device=input_features.device)
        accent_embeddings = self.accent_codebook[accent_ids]  # [batch_size, d_model]
        
        # Expand accent embeddings to match sequence length
        accent_embeddings = accent_embeddings.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        
        # Apply cross attention between encoder outputs and accent embeddings
        enhanced_states, _ = self.cross_attention(
            query=hidden_states,
            key=accent_embeddings,
            value=accent_embeddings,
            key_padding_mask=None,
            need_weights=False
        )
        hidden_states = hidden_states #+ enhanced_states
        
        # Replace original encoder outputs with enhanced states
        encoder_outputs.last_hidden_state = hidden_states
        
        # Continue with normal Whisper forward pass
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        
        
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            # encoder_attention_mask=attention_mask,
            **kwargs
        )
        
        lm_logits = self.proj_out(decoder_outputs.last_hidden_state)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=enhanced_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    # def generate(self, input_features, attention_mask=None, accent_ids=None, **kwargs):
    #     """
    #     Generate text with accent-aware processing.
    #     """
    #     if accent_ids is None:
    #         accent_ids = torch.zeros(input_features.size(0), dtype=torch.long, device=input_features.device)
        
    #     # Get encoder outputs with accent processing
    #     encoder_outputs = self.model.encoder(input_features, attention_mask=attention_mask)
    #     hidden_states = encoder_outputs.last_hidden_state
        
    #     # Process accent embeddings
    #     accent_embeddings = self.accent_codebook[accent_ids]
    #     accent_embeddings = accent_embeddings.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        
    #     # Apply cross attention
    #     enhanced_states, _ = self.cross_attention(
    #         query=hidden_states,
    #         key=accent_embeddings,
    #         value=accent_embeddings,
    #         key_padding_mask=None,
    #         need_weights=False
    #     )
        
    #     # Replace encoder outputs with enhanced states
    #     encoder_outputs.last_hidden_state = enhanced_states
        
    #     # Continue with normal Whisper generation
    #     return super().generate(
    #         input_features=input_features,
    #         attention_mask=attention_mask,
    #         encoder_outputs=encoder_outputs,
    #         **kwargs
    #     ) 


if __name__ == "__main__":
    # Sample dimensions for debugging
    batch_size = 4
    seq_length = 3000
    num_mel_bins = 80
    max_decoder_length = 448
    
    # Initialize model from pretrained Whisper
    model = AccentWhisperModel.from_pretrained(
        "openai/whisper-base.en",
    )
    
    # Create sample input tensors
    input_features = torch.randn(batch_size, num_mel_bins, seq_length)
    attention_mask = torch.randint(0, 2, (batch_size, seq_length))
    accent_ids = torch.randint(0, NUM_ACCENTS, (batch_size,))
    labels = torch.randint(0, model.config.vocab_size, (batch_size, max_decoder_length))
    decoder_attention_mask = torch.randint(0, 2, (batch_size, max_decoder_length))
    
    # Test forward pass
    outputs = model(
        input_features=input_features,
        attention_mask=attention_mask,
        accent_ids=accent_ids,
        labels=labels,
        decoder_attention_mask=decoder_attention_mask
    )
    
    # Print detailed output information
    print("\nModel Output Statistics:")
    print("-" * 50)
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Logits mean: {outputs.logits.mean().item():.4f}")
    print(f"Logits std: {outputs.logits.std().item():.4f}")
    print(f"Logits min: {outputs.logits.min().item():.4f}")
    print(f"Logits max: {outputs.logits.max().item():.4f}")
    print(f"Number of NaNs in logits: {torch.isnan(outputs.logits).sum().item()}")
    print(f"Number of Infs in logits: {torch.isinf(outputs.logits).sum().item()}")
    
    if outputs.loss is not None:
        print(f"\nLoss value: {outputs.loss.item():.4f}")
        print(f"Loss is NaN: {torch.isnan(outputs.loss).item()}")
    
    # # Test generation
    # generated_ids = model.generate(
    #     input_features=input_features,
    #     attention_mask=attention_mask,
    #     accent_ids=accent_ids
    # )
    
    # print("\nGeneration Statistics:")
    # print("-" * 50)
    # print(f"Generated ids shape: {generated_ids.shape}")
    # print(f"Unique tokens in generation: {torch.unique(generated_ids).shape[0]}")
    # print(f"Number of NaNs in generated ids: {torch.isnan(generated_ids).sum().item()}")
    
    # # Print model configuration
    # print("\nModel Configuration:")
    # print("-" * 50)
    # print(f"Number of accents: {model.num_accents}")
    # print(f"Model dimension: {model.config.d_model}")
    # print(f"Vocabulary size: {model.config.vocab_size}")
    # print(f"Number of attention heads: {model.config.encoder_attention_heads}")
