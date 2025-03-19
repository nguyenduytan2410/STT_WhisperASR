import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from datasets import load_dataset

# Load dataset
dataset = load_dataset("google/fleurs", "vi_vn", trust_remote_code=True)

# Load processor và model
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="vi", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Tiền xử lý dataset
def preprocess_function(examples):
    audio = examples["audio"]
    examples["input_features"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0].tolist()
    examples["labels"] = processor.tokenizer(examples["transcription"], return_tensors="pt").input_ids[0].tolist()
    return examples

# Áp dụng tiền xử lý
train_dataset = dataset["train"].map(
    preprocess_function,
    remove_columns=dataset["train"].column_names
)

# Data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id
)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper_finetune",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Tích lũy gradient để giảm tải GPU
    learning_rate=1e-5,
    num_train_epochs=1,  # Giảm epoch để thử nghiệm
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="no"  # Bỏ đánh giá nếu không có validation set
)

# Khởi tạo trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

print('bat dau huan luyen')
# Huấn luyện
trainer.train()