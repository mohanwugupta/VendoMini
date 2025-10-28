"""Simple HuggingFace model downloader for Windows."""

import os
import subprocess
import sys
from pathlib import Path

# Set download directory
MODELS_DIR = Path("/scratch/gpfs/JORDANAT/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Models to download
MODELS = [
    # "openai/gpt-oss-20b",
    # "deepseek-ai/DeepSeek-V2.5",
    "meta-llama/Llama-3.3-70B-Instruct",
    # "Qwen/Qwen2.5-72B-Instruct",
    # "Qwen/Qwen3-32B",
    # "deepseek-ai/deepseek-llm-7b-chat"
]

def download_model(repo_id: str):
    """Download a model using huggingface-cli."""
    print(f"\n{'='*60}")
    print(f"📦 Downloading: {repo_id}")
    print(f"{'='*60}")

    try:
        # Convert repo_id to directory name (org/name -> org--name)
        dir_name = repo_id.replace('/', '--')
        model_dir = MODELS_DIR / dir_name
        
        # Use huggingface-cli for more reliable downloads
        cmd = [
            "huggingface-cli", "download",
            repo_id,
            "--local-dir", str(model_dir),
            "--local-dir-use-symlinks", "False"
        ]

        print(f"Running: {' '.join(cmd)}")
        print(f"Downloading to: {model_dir}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Successfully downloaded {repo_id}")
            return True
        else:
            print(f"❌ Error downloading {repo_id}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Exception downloading {repo_id}: {e}")
        return False

def main():
    print("🚀 Starting HuggingFace Model Downloads (CLI method)")
    print(f"📁 Download directory: {MODELS_DIR}")
    print(f"📊 Total models to download: {len(MODELS)}")

    # Authenticate first
    print("\n🔐 Authenticating with HuggingFace...")
    try:
        result = subprocess.run([
            "huggingface-cli", "login", "--token", "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"
        ], capture_output=True, text=True, check=True)
        print("✅ Successfully authenticated with HuggingFace")
    except subprocess.CalledProcessError as e:
        print(f"❌ Authentication failed: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

    downloaded = []
    failed = []

    for repo_id in MODELS:
        if download_model(repo_id):
            downloaded.append(repo_id)
        else:
            failed.append(repo_id)

    # Summary
    print("\n" + "="*60)
    print("📊 DOWNLOAD SUMMARY")
    print("="*60)
    print(f"✅ Successfully downloaded: {len(downloaded)}/{len(MODELS)}")
    for model in downloaded:
        print(f"  ✓ {model}")

    if failed:
        print(f"\n❌ Failed downloads: {len(failed)}")
        for model in failed:
            print(f"  ✗ {model}")

    print(f"\n📁 All models stored in: {MODELS_DIR}")
    print("\n✅ Download process complete!")

if __name__ == "__main__":
    main()