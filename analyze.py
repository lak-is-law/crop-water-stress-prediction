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

def print_terminal_graph(moist, temp, hum, veg):
    """Generates an ASCII bar chart in the terminal based on user inputs."""
    def draw_bar(label, value, max_value, unit=""):
        bar_length = 30
        # Cap the value at max_value for the visual bar
        visual_val = min(value, max_value)
        visual_val = max(visual_val, 0)
        filled_len = int(bar_length * (visual_val / max_value))
        bar = '█' * filled_len + '░' * (bar_length - filled_len)
        print(f"{label:<18} |{bar}| {value:.1f}{unit}")
        
    print("\n📊 TERMINAL INPUT GRAPH:")
    draw_bar("Soil Moisture", moist, 100, "%")
    draw_bar("Temperature", temp, 50, "°C")
    draw_bar("Humidity", hum, 100, "%")
    draw_bar("Vegetation Index", veg, 1.0, "")
    print("-" * 55)

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
    
    # Custom Input Loop
    print("\n" + "="*40)
    print("🔮 Custom Water Stress Prediction 🔮")
    print("="*40)
    while True:
        choice = input("\nWould you like to enter custom inputs for a prediction? (y/n): ").strip().lower()
        if choice != 'y':
            print("Exiting custom prediction.")
            break
            
        try:
            moist = float(input("Enter Soil Moisture (e.g., 10 to 60): "))
            temp = float(input("Enter Temperature in °C (e.g., 15 to 45): "))
            hum = float(input("Enter Humidity % (e.g., 20 to 90): "))
            veg = float(input("Enter Vegetation Index (e.g., 0.2 to 0.9): "))
            
            # Draw the terminal graph
            print_terminal_graph(moist, temp, hum, veg)
            
            # Create a dataframe for the custom input to match training feature names
            custom_df = pd.DataFrame([[moist, temp, hum, veg]], columns=feature_cols)
            
            # Scale the input using the same scaler used for training
            custom_scaled = scaler.transform(custom_df)
            
            # Predict
            prediction = model.predict(custom_scaled)[0]
            
            if prediction == 1:
                print("⚠️  PREDICTION: The crop is likely UNDER WATER STRESS. Needs irrigation!")
            else:
                print("✅ PREDICTION: The crop is HEALTHY (No water stress).")
                
        except ValueError:
            print("Invalid input. Please enter numerical values only.")
    
def main():
    conn = setup_database_and_load_data()
    if conn:
        analyze_data_from_db(conn)
        conn.close()

if __name__ == "__main__":
    main()
