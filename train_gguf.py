from unsloth import FastLanguageModel
import torch
from datasets import Dataset
import json

# 1. Modeli yükle
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen3-8B-Q4_K_M.gguf",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,  # 4-bit quantization ile bellekte yerden tasarruf
)

# 2. LoRA adaptörü ekle
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 3. Veri setini yükle
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Eğitim formatına çevir
train_data = []
for item in data["data"]:
    text = f"""### Soru:
{item["question"]}

### Cevap:
{item["answer"]["result"]}

### Açıklama:
{item["answer"]["explanation"]}"""
    
    train_data.append({"text": text})

dataset = Dataset.from_list(train_data)

# 4. Tokenize et
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 5. Eğitim ayarları
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=10,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",
        save_strategy="epoch",
    ),
)

# 6. Eğit
trainer.train()

# 7. Modeli kaydet (GGUF formatında)
model.save_pretrained("education_model")
model.save_pretrained_gguf("education_model", tokenizer, quantization_method="q4_k_m")

print("✅ Model kaydedildi: education_model.gguf")