import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

def download_and_load_kaggle_dataset(kaggle_link):
    """
    Downloads dataset from a kaggle link or handle.
    Example handle: 'tarunpaparaju/plant-health-prediction'
    """
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Please install it using: pip install kagglehub pandas scikit-learn")
        return None

    print(f"Downloading dataset: {kaggle_link}")
    try:
        # kagglehub.dataset_download expects the dataset handle (e.g., 'username/dataset-name')
        path = kagglehub.dataset_download(kaggle_link)
        print(f"Data downloaded to: {path}")
        
        # Look for CSV files
        csv_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            print("No CSV files found in the dataset.")
            return None
            
        print(f"Loading {csv_files[0]}...")
        df = pd.read_csv(csv_files[0])
        return df
    except Exception as e:
        print(f"Error downloading or loading dataset: {e}")
        print("Note: Make sure your Kaggle credentials are set up (e.g., ~/.kaggle/kaggle.json)")
        return None

def train_kernel_model(df):
    """
    Trains a nonlinear kernel SVM model to predict water stress.
    Requires features: soil moisture, temperature, humidity, vegetation-index.
    """
    print("\nDataset Preview:")
    print(df.head())
    
    # In a real scenario, you need to map your dataset's columns to these variables.
    # Here we assume the user will adjust column names to match their dataset.
    print("\n--- Model Training Phase ---")
    print("Please modify the column names in the script to match your Kaggle dataset.")
    
    # Example column names (adjust these based on the actual Kaggle dataset)
    target_col = 'water_stress' # Or 'label', 'stress_level', etc.
    feature_cols = ['soil_moisture', 'temperature', 'humidity', 'vegetation_index']
    
    # Check if expected columns exist, otherwise use dummy columns for demonstration
    missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
    
    if missing_cols:
        print(f"Warning: The following columns were not found in the dataset: {missing_cols}")
        print("Using the first available numerical columns as features, and the last as target for demonstration.")
        # Auto-select columns for demonstration purposes
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(num_cols) < 2:
            print("Not enough numerical columns to train a model.")
            return
            
        feature_cols = num_cols[:-1]
        target_col = num_cols[-1]
        
        # If classification, target should be categorical or discrete
        if df[target_col].nunique() > 10:
            print("Target column appears continuous. Converting to binary class for classification (above/below mean).")
            df[target_col] = (df[target_col] > df[target_col].mean()).astype(int)
            
    print(f"Features: {feature_cols}")
    print(f"Target: {target_col}")
    
    # Prepare data
    X = df[feature_cols]
    y = df[target_col]
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mode()[0])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Nonlinear Kernel Model (RBF Kernel SVM)
    print("\nTraining Non-Linear Kernel SVM (RBF)...")
    model = SVC(kernel='rbf', gamma='scale', C=1.0)
    model.fit(X_train_scaled, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test_scaled)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def main():
    parser = argparse.ArgumentParser(description="Predict Crop Water Stress using Kernel-Based Classification")
    parser.add_argument('--dataset', type=str, default='tarunpaparaju/plant-health-prediction',
                        help='Kaggle dataset handle (e.g., username/dataset-name)')
    
    args = parser.parse_args()
    
    df = download_and_load_kaggle_dataset(args.dataset)
    if df is not None:
        train_kernel_model(df)

if __name__ == "__main__":
    main()
