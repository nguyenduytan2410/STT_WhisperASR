from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch
import torchaudio
from datasets import load_dataset

# 3.1. Load tokenizer và feature extractor của Whisper
model_name = "openai/whisper-small"  # Hoặc dùng whisper-medium, whisper-large tùy yêu cầu
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="vi", task="transcribe")

# 3.2. Hàm tiền xử lý dữ liệu
def prepare_dataset(batch):
    audio = batch["audio"]
    
    # Resample nếu cần (Fleurs đã chuẩn 16kHz, Whisper cần 16kHz)
    if audio["sampling_rate"] != 16000:
        audio["array"] = torchaudio.transforms.Resample(audio["sampling_rate"], 16000)(torch.tensor(audio["array"])).numpy()
    
    # Trích xuất đặc trưng
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    
    # Token hóa transcript
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

# dataset = dataset.map(prepare_dataset, remove_columns=["audio", "path", "transcription"])
dataset = load_dataset("google/fleurs", "vi_vn", trust_remote_code=True)  # Thay "vi_vn" bằng mã ngôn ngữ bạn muốn
dataset = dataset.map(prepare_dataset, remove_columns=["audio", "path", "transcription"])
print(dataset)

# Bước 4: Huấn luyện mô hình
# 4.1. Load mô hình Whisper
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="vi", task="transcribe")
 
# 4.2. Cấu hình Trainer
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-fleurs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=500,
    num_train_epochs=5,
    save_total_limit=2,
    fp16=False,  # Nếu dùng GPU hỗ trợ FP16
    push_to_hub=False
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 4.3. Bắt đầu huấn luyện
trainer.train()
