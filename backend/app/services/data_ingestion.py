"""Data ingestion and preprocessing service."""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from ..config import settings


class DataIngestionService:
    """
    Handles data import, redaction, splitting, and JSONL export.

    Responsibilities:
    - PII redaction
    - Dataset splitting
    - Producing consistent, reproducible JSONL files
    """

    def __init__(self):
        self.raw_dir = Path(settings.raw_data_dir)
        self.processed_dir = Path(settings.processed_data_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def redact_pii(self, text: str) -> str:
        """
        Redact PII from text.

        Replaces:
        - Emails -> [EMAIL]
        - Phone numbers -> [PHONE]
        - URLs -> [URL]
        """
        # Email pattern
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            text
        )

        # Phone pattern (basic)
        text = re.sub(
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            '[PHONE]',
            text
        )

        # URL pattern
        text = re.sub(
            r'https?://[^\s]+',
            '[URL]',
            text
        )

        return text

    def normalize_text(self, text: str) -> str:
        """Normalize text (strip extra whitespace, unify quotes)."""
        # Strip extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Unify quote characters
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        return text

    def convert_to_sft_jsonl(
        self,
        input_file: Path,
        output_file: Path,
        system_prompt: str = None
    ) -> int:
        """
        Convert CSV/JSONL to SFT training format.

        Expected input columns: incoming_subject, incoming_body, reply_body

        Returns: number of examples processed
        """
        if system_prompt is None:
            system_prompt = (
                "You are a professional email responder. "
                "Preserve facts; do not invent dates, prices, or names. "
                "Tone: friendly, concise, specific."
            )

        df = pd.read_csv(input_file)
        count = 0

        with open(output_file, 'w') as f:
            for _, row in df.iterrows():
                # Construct messages
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Subject: {row['incoming_subject']}\n\nBody: {row['incoming_body']}"
                    },
                    {"role": "assistant", "content": row['reply_body']}
                ]
                
                # Write to JSONL
                f.write(json.dumps({"messages": messages}) + '\n')
                count += 1
                
        return count

    def convert_to_dpo_jsonl(
        self,
        input_file: Path,
        output_file: Path
    ) -> int:
        """
        Convert preference data to DPO training format.

        Expected input columns: prompt_subject, prompt_body, preferred, non_preferred

        Returns: number of examples processed
        """
        # TODO: Implement DPO conversion
        return 0

    def split_data(
        self,
        input_file: Path,
        train_ratio: float = None,
        val_ratio: float = None,
        test_ratio: float = None
    ) -> Dict[str, Path]:
        """
        Split data into train/val/test sets.

        Returns: dict with paths to train.jsonl, val.jsonl, test.jsonl
        """
        if train_ratio is None:
            train_ratio = settings.train_split
        if val_ratio is None:
            val_ratio = settings.val_split
        if test_ratio is None:
            test_ratio = settings.test_split

        # Read all lines
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 10:
            print("Warning: Dataset too small for splitting. Using all data for train/val/test.")
            train_data = lines
            val_data = lines
            test_data = lines
        else:
            # Shuffle
            from sklearn.model_selection import train_test_split
            
            # First split: Train vs (Val + Test)
            train_data, temp_data = train_test_split(
                lines, 
                train_size=train_ratio, 
                random_state=42
            )
            
            # Second split: Val vs Test
            # Adjust ratio for the remaining data
            remaining_ratio = 1.0 - train_ratio
            val_relative_ratio = val_ratio / remaining_ratio
            
            val_data, test_data = train_test_split(
                temp_data, 
                train_size=val_relative_ratio, 
                random_state=42
            )
        
        # Save splits
        train_path = self.processed_dir / "train.jsonl"
        val_path = self.processed_dir / "val.jsonl"
        test_path = self.processed_dir / "test.jsonl"
        
        with open(train_path, 'w') as f:
            f.writelines(train_data)
            
        with open(val_path, 'w') as f:
            f.writelines(val_data)
            
        with open(test_path, 'w') as f:
            f.writelines(test_data)

        return {
            "train": train_path,
            "val": val_path,
            "test": test_path
        }
