import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split 

class TelcoChurnPreprocessor:
    def __init__(self, raw_path, output_folder='processed_data'): 
        self.raw_path = raw_path
        self.output_folder = output_folder
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_data(self):
        """Loads data from the raw Excel file."""
        try:
            df = pd.read_excel(self.raw_path)
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {self.raw_path}")
            return None
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            return None

    def preprocess(self):
        """Performs preprocessing steps consistent with the notebook."""
        df = self.load_data()
        if df is None:
            return None, None, None, None # Return None if data loading failed

        # Drop kolom tidak relevan 
        drop_cols = ['CustomerID','Count','Country','State','City','Zip Code','Lat Long',
                     'Latitude','Longitude','Churn Label','Churn Score','Churn Reason']
        df.drop(columns=drop_cols, inplace=True)

        # Handle Total Charges
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
        # Temukan median sebelum mengisi
        median_total_charges = df['Total Charges'].median()
        df['Total Charges'] = df['Total Charges'].fillna(median_total_charges)


        # Encode kategori 
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # Scaling fitur numerik 
        numerical_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV']
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])


        # Split X dan y 
        X = df.drop(columns=['Churn Value'])
        y = df['Churn Value']

        # Train-test split 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Saves the processed split data to CSV files ."""
        if X_train is None or X_test is None or y_train is None or y_test is None:
            print("Cannot save data: preprocessing failed.")
            return

        os.makedirs(self.output_folder, exist_ok=True)

        try:
            X_train.to_csv(os.path.join(self.output_folder, 'X_train_processed.csv'), index=False)
            X_test.to_csv(os.path.join(self.output_folder, 'X_test_processed.csv'), index=False)
            y_train.to_csv(os.path.join(self.output_folder, 'y_train_processed.csv'), index=False)
            y_test.to_csv(os.path.join(self.output_folder, 'y_test_processed.csv'), index=False)
            print(f"Processed data saved to folder '{self.output_folder}'.")
        except Exception as e:
            print(f"An error occurred while saving data: {e}")


if __name__ == "__main__":
    raw_path_file = "telco_churn_raw/Telco_customer_churn.xlsx"
    output_data_folder = "processed_data"

    preprocessor = TelcoChurnPreprocessor(raw_path_file, output_data_folder)
    X_train_processed, X_test_processed, y_train_processed, y_test_processed = preprocessor.preprocess()

    if X_train_processed is not None:
        preprocessor.save_processed_data(X_train_processed, X_test_processed, y_train_processed, y_test_processed)
        print("\nPreprocessing dan penyimpanan data selesai.")
        print("Contoh X_train_processed head:")
        print(X_train_processed.head())
    else:
        print("\nPreprocessing gagal.")