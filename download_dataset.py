import os
import argparse
import shutil

def download_kaggle_dataset(kaggle_link, dest_folder="data"):
    """
    Downloads a dataset from a Kaggle link or handle and moves it to a local folder.
    Example handle: 'tarunpaparaju/plant-health-prediction'
    """
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Please install it using: pip install kagglehub")
        return None

    print(f"Downloading dataset: {kaggle_link}")
    try:
        # kagglehub.dataset_download expects the dataset handle
        download_path = kagglehub.dataset_download(kaggle_link)
        print(f"Data downloaded to Kaggle cache: {download_path}")
        
        # Create destination folder
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
            
        # Look for CSV files and copy them to our local 'data' folder
        csv_files = []
        for root, dirs, files in os.walk(download_path):
            for file in files:
                if file.endswith('.csv'):
                    source_path = os.path.join(root, file)
                    dest_path = os.path.join(dest_folder, file)
                    shutil.copy2(source_path, dest_path)
                    csv_files.append(dest_path)
                    print(f"Copied {file} to {dest_folder}/")
        
        if not csv_files:
            print("No CSV files found in the downloaded dataset.")
            return None
            
        print("\nDataset successfully downloaded and extracted locally.")
        return csv_files
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Note: Make sure your Kaggle credentials are set up (e.g., ~/.kaggle/kaggle.json)")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset from Kaggle")
    parser.add_argument('--dataset', type=str, default='tarunpaparaju/plant-health-prediction',
                        help='Kaggle dataset handle (e.g., username/dataset-name)')
    
    args = parser.parse_args()
    download_kaggle_dataset(args.dataset)
