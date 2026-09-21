import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
    
    df.to_sql('sensor_readings', conn, if_exists='replace', index=False)
    print("Data successfully loaded into the 'sensor_readings' table.\n")
    
    return conn

def print_terminal_graph(moist, temp, hum, veg):
    """Generates an ASCII bar chart in the terminal based on user inputs."""
    def draw_bar(label, value, max_value, unit=""):
        bar_length = 30
        visual_val = min(value, max_value)
        visual_val = max(visual_val, 0)
        filled_len = int(bar_length * (visual_val / max_value))
        bar = '#' * filled_len + '-' * (bar_length - filled_len)
        print(f"{label:<18} |{bar}| {value:.1f}{unit}")
        
    print("\n[ TERMINAL INPUT GRAPH ]")
    draw_bar("Soil Moisture", moist, 100, "%")
    draw_bar("Temperature", temp, 50, "C")
    draw_bar("Humidity", hum, 100, "%")
    draw_bar("Vegetation Index", veg, 1.0, "")
    print("-" * 55)

def generate_visualizations(df, feature_cols, target_col, X_scaled):
    """
    Generates a Pie Chart for class distribution and a K-Means Cluster Map.
    """
    print("Generating Graphical Plots (Close the window to continue)...")
    
    # Set up the figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.canvas.manager.set_window_title('Crop Water Stress Analysis')
    
    # 1. Pie Chart for Water Stress Distribution
    stress_counts = df[target_col].value_counts()
    labels = ['Healthy (0)', 'Under Stress (1)'] if 0 in stress_counts.index and 1 in stress_counts.index else stress_counts.index
    ax1.pie(stress_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
    ax1.set_title('Proportion of Crop Water Stress')
    
    # 2. K-Means Cluster Map (Reduced to 2D via PCA)
    # Perform K-Means clustering (K=2, since we suspect 2 main states: Healthy vs Stressed)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Use PCA to reduce 4D data to 2D for plotting
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Scatter plot of the clusters
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', ax=ax2, s=50, alpha=0.6)
    
    # Plot cluster centers
    centers_pca = pca.transform(kmeans.cluster_centers_)
    ax2.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, marker='X', label='Centroids')
    
    ax2.set_title('K-Means Cluster Map (PCA Reduced)')
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

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
    X_scaled_full = scaler.transform(X) # For clustering full dataset
    
    # Generate the requested plots (Pie chart & K-Means Cluster Map)
    generate_visualizations(df_db, feature_cols, target_col, X_scaled_full)
    
    print("Training Non-Linear Kernel SVM (RBF)...")
    model = SVC(kernel='rbf')
    model.fit(X_train_scaled, y_train)
    
    preds = model.predict(X_test_scaled)
    print(f"Model Accuracy: {accuracy_score(y_test, preds):.4f}")
    
    # Custom Input Loop
    print("\n" + "="*40)
    print("Custom Water Stress Prediction")
    print("="*40)
    while True:
        choice = input("\nWould you like to enter custom inputs for a prediction? (y/n): ").strip().lower()
        if choice != 'y':
            print("Exiting custom prediction.")
            break
            
        try:
            moist = float(input("Enter Soil Moisture (e.g., 10 to 60): "))
            temp = float(input("Enter Temperature in C (e.g., 15 to 45): "))
            hum = float(input("Enter Humidity % (e.g., 20 to 90): "))
            veg = float(input("Enter Vegetation Index (e.g., 0.2 to 0.9): "))
            
            print_terminal_graph(moist, temp, hum, veg)
            
            custom_df = pd.DataFrame([[moist, temp, hum, veg]], columns=feature_cols)
            custom_scaled = scaler.transform(custom_df)
            prediction = model.predict(custom_scaled)[0]
            
            if prediction == 1:
                print("PREDICTION: The crop is likely UNDER WATER STRESS. Needs irrigation!")
            else:
                print("PREDICTION: The crop is HEALTHY (No water stress).")
                
        except ValueError:
            print("Invalid input. Please enter numerical values only.")

def main():
    conn = setup_database_and_load_data()
    if conn:
        analyze_data_from_db(conn)
        conn.close()

if __name__ == "__main__":
    main()
