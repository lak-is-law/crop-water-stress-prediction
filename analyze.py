import os
import sqlite3
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

DB_NAME = "crop_data.db"
DATASET_CSV_NAME = "dataset.csv"

def setup_database_and_load_data():
    """
    Connects to SQLite database and loads the CSV data into a table.
    """
    if not os.path.exists(DATASET_CSV_NAME):
        print(f"Error: {DATASET_CSV_NAME} not found in the directory.")
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
    
    num_cols = df_db.select_dtypes(include=['number']).columns.tolist()
    if len(num_cols) < 2:
        print("Not enough numeric columns for analysis.")
        return
        
    feature_cols = num_cols[:-1]
    target_col = num_cols[-1]
    
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
