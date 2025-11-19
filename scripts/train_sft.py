import argparse

def train_sft(data_path: str, output_dir: str):
    print(f"Starting SFT training with data from {data_path}...")
    # Placeholder for SFT training logic
    print(f"Training complete. Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    train_sft(args.data_path, args.output_dir)
