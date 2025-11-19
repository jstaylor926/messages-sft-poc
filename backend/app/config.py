"""Configuration management for Email Style PoC."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"
    openai_max_tokens: int = 1000
    openai_temperature: float = 0.7

    # HuggingFace Configuration
    hf_token: Optional[str] = None
    hf_cache_dir: str = "./cache/huggingface"

    # Model Configuration
    base_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    lora_sft_adapter_path: Optional[str] = "./artifacts/lora_sft"
    lora_dpo_adapter_path: Optional[str] = "./artifacts/lora_dpo"
    infer_adapter_dir: Optional[str] = None  # Switchable adapter directory

    # Training Configuration
    max_seq_length: int = 2048
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 2

    # Data Configuration
    data_dir: str = "./data"
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Inference Configuration
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # System Configuration
    device: str = "cuda"  # or "cpu" or "mps"
    load_in_4bit: bool = True  # QLoRA quantization
    use_flash_attention: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()
