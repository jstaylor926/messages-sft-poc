import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.services.data_ingestion import DataIngestionService
from backend.app.config import settings

def main():
    print("Starting data preparation...")
    service = DataIngestionService()
    
    # Paths
    raw_file = Path(settings.raw_data_dir) / "sft_pairs.csv"
    processed_file = Path(settings.processed_data_dir) / "all.jsonl"
    
    if not raw_file.exists():
        print(f"Error: {raw_file} not found.")
        return
    
    # Convert
    print(f"Converting {raw_file} to JSONL...")
    count = service.convert_to_sft_jsonl(raw_file, processed_file)
    print(f"Converted {count} examples.")
    
    # Split
    print("Splitting data...")
    splits = service.split_data(processed_file)
    
    print("Data preparation complete!")
    print(f"Train: {splits['train']}")
    print(f"Val: {splits['val']}")
    print(f"Test: {splits['test']}")

if __name__ == "__main__":
    main()
