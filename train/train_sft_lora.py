import os
import sys
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from backend.app.config import settings

def train():
    print(f"Starting SFT training for model: {settings.base_model_name}")
    
    # 1. Load Data
    train_file = Path(settings.processed_data_dir) / "train.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found at {train_file}")
        
    dataset = load_dataset("json", data_files={"train": str(train_file)})
    
    # 2. Load Model & Tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=settings.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        settings.base_model_name,
        quantization_config=bnb_config if settings.load_in_4bit else None,
        device_map=settings.device if settings.device != "cuda" else "auto",
        token=settings.hf_token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(settings.base_model_name, token=settings.hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Configure LoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        r=settings.lora_rank,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    
    # 4. Training Arguments
    output_dir = settings.lora_sft_adapter_path or "./artifacts/lora_sft"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=settings.num_train_epochs,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        evaluation_strategy="no", # Skip eval for now on tiny data
        fp16=True if settings.device == "cuda" else False,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        dataset_text_field="text", # TRL expects a text field, but we have messages. We need a formatting func.
        max_seq_length=settings.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        formatting_func=lambda x: [tokenizer.apply_chat_template(msgs, tokenize=False) for msgs in x["messages"]]
    )

    # 6. Train
    print("Starting training loop...")
    trainer.train()
    
    # 7. Save
    print(f"Saving adapter to {output_dir}")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    train()
