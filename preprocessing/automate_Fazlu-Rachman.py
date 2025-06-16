import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

class TelcoChurnPreprocessor:
    def __init__(self, raw_path, save_path=None):
        self.raw_path = raw_path
        self.save_path = save_path
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_data(self):
        df = pd.read_excel(self.raw_path)
        return df

    def preprocess(self):
        df = self.load_data()

        # Drop kolom tidak relevan
        drop_cols = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Lat Long', 'Churn Reason']
        df = df.drop(columns=drop_cols)

        # Handle Total Charges (konversi ke numeric + isi NaN dengan 0)
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
        df['Total Charges'] = df['Total Charges'].fillna(0)

        # Encode kategori
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # Scaling fitur numerik
        numerical_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV']
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])

        # Simpan hasil jika diminta
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            df.to_excel(self.save_path, index=False)

        return df

if __name__ == "__main__":
    raw_path = "telco_churn_raw/Telco_customer_churn.xlsx"
    save_path = "preprocessing/telco_churn_preprocessing/telco_churn_preprocessed.xlsx"

    preprocessor = TelcoChurnPreprocessor(raw_path, save_path)
    df_processed = preprocessor.preprocess()

    print("Preprocessing selesai!")
    print(df_processed.head())
