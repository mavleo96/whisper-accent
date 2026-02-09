from peft import PeftConfig, PeftModel

from src.model import WhisperAccentProcessor
from src.train.train import model_init


def load_model_from_pretrained(model_id):
    config = PeftConfig.from_pretrained(model_id)
    processor = WhisperAccentProcessor.from_pretrained(model_id)

    # Directly loading from base model not possible due to resized embeddings
    # We initialize a new model with the correct tokenizer/special tokens
    # and apply adapter and return merged model
    model = model_init(config.base_model_name_or_path, processor)
    model = PeftModel.from_pretrained(model, model_id)
    model = model.merge_and_unload()

    return model, processor
