import os
import sqlite3
import pandas as pd
import shutil
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

DB_NAME = "crop_data.db"
KAGGLE_HANDLE = "tarunpaparaju/plant-health-prediction" # Example dataset for water stress/plant health
RAW_CSV_NAME = "kaggle_water_stress_dataset.csv"
DATASET_CSV_NAME = "dataset.csv"

def import_from_kaggle():
    """
    Imports the dataset directly from Kaggle using kagglehub.
    """
    print(f"Importing dataset from Kaggle: {KAGGLE_HANDLE}...")
    try:
        import kagglehub
    except ImportError:
        print("Please install kagglehub: pip install kagglehub")
        return False
        
    try:
        # Download from Kaggle
        download_path = kagglehub.dataset_download(KAGGLE_HANDLE)
        
        # Find the downloaded CSV
        csv_files = []
        for root, dirs, files in os.walk(download_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
                    
        if not csv_files:
            print("No CSV found in Kaggle dataset.")
            return False
            
        # Copy and rename files to match required flat structure
        shutil.copy(csv_files[0], RAW_CSV_NAME)
        shutil.copy(csv_files[0], DATASET_CSV_NAME)
        print(f"Successfully imported from Kaggle! Saved as {RAW_CSV_NAME} and {DATASET_CSV_NAME}")
        return True
    except Exception as e:
        print(f"Kaggle import failed: {e}")
        print("Make sure your Kaggle API credentials are set up.")
        return False

def setup_database_and_load_data():
    """
    Connects to SQLite database and loads the CSV data into a table.
    """
    if not os.path.exists(DATASET_CSV_NAME):
        success = import_from_kaggle()
        if not success:
            return None
            
    print(f"Connecting to database: {DB_NAME}...")
    conn = sqlite3.connect(DB_NAME)
    
    print(f"Loading data from {DATASET_CSV_NAME} into database...")
    df = pd.read_csv(DATASET_CSV_NAME)
    
    # Write the data to a SQLite table named 'sensor_readings'
    df.to_sql('sensor_readings', conn, if_exists='replace', index=False)
    print("Data successfully loaded into the 'sensor_readings' table.\n")
    
    return conn

def analyze_data_from_db(conn):
    """
    Queries the database and runs the non-linear SVM analysis.
    """
    print("--- Running Analysis from Database ---")
    query = "SELECT * FROM sensor_readings LIMIT 5000"
    df_db = pd.read_sql_query(query, conn)
    
    # For demonstration, we automatically select numeric columns for our features
    num_cols = df_db.select_dtypes(include=['number']).columns.tolist()
    if len(num_cols) < 2:
        print("Not enough numeric columns for analysis.")
        return
        
    feature_cols = num_cols[:-1]
    target_col = num_cols[-1]
    
    # Convert target to binary classification if continuous
    if df_db[target_col].nunique() > 10:
        df_db[target_col] = (df_db[target_col] > df_db[target_col].mean()).astype(int)
        
    X = df_db[feature_cols].fillna(df_db[feature_cols].mean())
    y = df_db[target_col].fillna(df_db[target_col].mode()[0])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Non-Linear Kernel SVM (RBF)...")
    model = SVC(kernel='rbf')
    model.fit(X_train_scaled, y_train)
    
    preds = model.predict(X_test_scaled)
    print(f"Model Accuracy: {accuracy_score(y_test, preds):.4f}")
    
def main():
    conn = setup_database_and_load_data()
    if conn:
        analyze_data_from_db(conn)
        conn.close()

if __name__ == "__main__":
    main()
