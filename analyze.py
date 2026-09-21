import os
import sqlite3
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

DB_NAME = "crop_data.db"

def setup_database_and_load_data():
    """
    Connects to SQLite database and loads the CSV data into a table.
    Demonstrates DB connectivity.
    """
    print(f"Connecting to database: {DB_NAME}...")
    # 1. DB Connectivity (SQLite is used here for zero-config demonstration)
    # For PostgreSQL, you would use: psycopg2.connect(host="...", database="...", user="...", password="...")
    conn = sqlite3.connect(DB_NAME)
    
    # 2. Use the local dataset.csv
    csv_path = "dataset.csv"
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' not found.")
        return None
        
    print(f"Loading data from {csv_path} into database...")
    
    # 3. Read CSV and push to Database
    df = pd.read_csv(csv_path)
    
    # Write the data to a SQLite table named 'sensor_readings'
    df.to_sql('sensor_readings', conn, if_exists='replace', index=False)
    print("Data successfully loaded into the 'sensor_readings' table.\n")
    
    return conn

def analyze_data_from_db(conn):
    """
    Queries the database and runs the analysis / modeling.
    """
    print("--- Running Analysis from Database ---")
    
    # Execute a SQL query to fetch data
    query = "SELECT * FROM sensor_readings LIMIT 1000"  # Limit for demonstration, remove to load all
    df_db = pd.read_sql_query(query, conn)
    
    print("Preview of data fetched from DB:")
    print(df_db.head())
    
    # Attempt to find features for our Kernel model
    print("\nPreparing for Kernel-Based Analysis...")
    
    # For demonstration, we automatically select numeric columns
    num_cols = df_db.select_dtypes(include=['number']).columns.tolist()
    
    if len(num_cols) < 2:
        print("Not enough numeric columns for analysis.")
        return
        
    feature_cols = num_cols[:-1]
    target_col = num_cols[-1]
    
    # Convert target to binary classification if it's continuous
    if df_db[target_col].nunique() > 10:
        print(f"Converting continuous target '{target_col}' into binary classes for classification.")
        df_db[target_col] = (df_db[target_col] > df_db[target_col].mean()).astype(int)
        
    print(f"Features used: {feature_cols}")
    print(f"Target variable: {target_col}")
    
    X = df_db[feature_cols].fillna(df_db[feature_cols].mean())
    y = df_db[target_col].fillna(df_db[target_col].mode()[0])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining Non-Linear Kernel SVM (RBF) on DB Data...")
    model = SVC(kernel='rbf')
    model.fit(X_train_scaled, y_train)
    
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy on testing set: {acc:.4f}")
    
def main():
    conn = setup_database_and_load_data()
    if conn:
        analyze_data_from_db(conn)
        conn.close()
        print("\nDatabase connection closed.")

if __name__ == "__main__":
    main()
