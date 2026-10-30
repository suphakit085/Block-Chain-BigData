"""
Download Elliptic Bitcoin Dataset from Kaggle

This script downloads the dataset from Kaggle and extracts it to the data/raw directory.

Requirements:
    - Kaggle API credentials (~/.kaggle/kaggle.json)
    - kaggle package: pip install kaggle

Usage:
    python scripts/download_data.py
"""

import os
import subprocess
import sys
from pathlib import Path


def check_kaggle_credentials():
    """Check if Kaggle API credentials exist"""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists():
        print("❌ Kaggle API credentials not found!")
        print("\nPlease follow these steps:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Scroll to 'API' section and click 'Create New Token'")
        print("3. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    return True


def download_dataset():
    """Download dataset from Kaggle"""
    # Create data directories
    raw_dir = Path('data/raw')
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 Downloading dataset from Kaggle...")
    print("Dataset: filmlove55za/blockchain-bigdata")
    
    try:
        # Download dataset using Kaggle API
        result = subprocess.run(
            [
                'kaggle', 'datasets', 'download',
                '-d', 'filmlove55za/blockchain-bigdata',
                '-p', str(raw_dir),
                '--unzip'
            ],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("✅ Dataset downloaded successfully!")
        print(f"📁 Files saved to: {raw_dir.absolute()}")
        
        # List downloaded files
        print("\n📋 Downloaded files:")
        for file in raw_dir.glob('*'):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.2f} MB)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading dataset: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Kaggle CLI not found!")
        print("Please install: pip install kaggle")
        return False


def main():
    print("=" * 60)
    print("Elliptic Bitcoin Dataset Downloader")
    print("=" * 60)
    print()
    
    # Check Kaggle credentials
    if not check_kaggle_credentials():
        sys.exit(1)
    
    # Download dataset
    if download_dataset():
        print("\n✨ Setup complete! You can now run the data preparation notebook.")
    else:
        print("\n❌ Download failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
