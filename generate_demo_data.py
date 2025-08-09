#!/usr/bin/env python3
"""
Simple data generation script for MLOps demo.
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

def generate_customer_churn_data(n_samples=5000, random_state=42):
    """Generate synthetic customer churn data."""
    np.random.seed(random_state)
    
    print(f"Generating {n_samples} customer churn records...")
    
    # Customer demographics
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_samples + 1)]
    
    # Age distribution (18-80)
    ages = np.random.normal(45, 15, n_samples).astype(int)
    ages = np.clip(ages, 18, 80)
    
    # Gender
    genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48])
    
    # Tenure in months (0-72 months, 6 years)
    tenure = np.random.exponential(24, n_samples).astype(int)
    tenure = np.clip(tenure, 0, 72)
    
    # Contract type
    contract_types = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'], 
        n_samples, 
        p=[0.55, 0.25, 0.20]
    )
    
    # Payment method
    payment_methods = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        n_samples,
        p=[0.35, 0.15, 0.25, 0.25]
    )
    
    # Services
    phone_service = np.random.choice([0, 1], n_samples, p=[0.1, 0.9])
    multiple_lines = np.where(
        phone_service == 1,
        np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        0
    )
    
    internet_service = np.random.choice(['No', 'DSL', 'Fiber optic'], n_samples, p=[0.2, 0.4, 0.4])
    
    # Additional services (dependent on internet service)
    online_security = np.where(
        internet_service != 'No',
        np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        0
    )
    
    online_backup = np.where(
        internet_service != 'No',
        np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        0
    )
    
    device_protection = np.where(
        internet_service != 'No',
        np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        0
    )
    
    tech_support = np.where(
        internet_service != 'No',
        np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        0
    )
    
    streaming_tv = np.where(
        internet_service != 'No',
        np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        0
    )
    
    streaming_movies = np.where(
        internet_service != 'No',
        np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        0
    )
    
    # Paperless billing
    paperless_billing = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    
    # Monthly charges (influenced by services)
    base_charge = 20
    service_charges = (
        phone_service * 5 +
        multiple_lines * 5 +
        (internet_service == 'DSL') * 25 +
        (internet_service == 'Fiber optic') * 45 +
        online_security * 5 +
        online_backup * 5 +
        device_protection * 5 +
        tech_support * 5 +
        streaming_tv * 10 +
        streaming_movies * 10
    )
    
    monthly_charges = base_charge + service_charges + np.random.normal(0, 5, n_samples)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)
    
    # Total charges (tenure * monthly charges with some variation)
    total_charges = tenure * monthly_charges * np.random.normal(1, 0.1, n_samples)
    total_charges = np.clip(total_charges, 0, None)
    
    # Churn probability calculation (this creates realistic patterns)
    churn_prob = 0.1  # Base probability
    
    # Factors that increase churn probability
    churn_prob += (contract_types == 'Month-to-month') * 0.3
    churn_prob += (payment_methods == 'Electronic check') * 0.2
    churn_prob += (tenure < 12) * 0.25
    churn_prob += (monthly_charges > 70) * 0.15
    churn_prob += (online_security == 0) * 0.1
    churn_prob += (tech_support == 0) * 0.1
    churn_prob += (ages > 65) * 0.05
    
    # Factors that decrease churn probability
    churn_prob -= (contract_types == 'Two year') * 0.2
    churn_prob -= (tenure > 24) * 0.15
    churn_prob -= (multiple_lines == 1) * 0.05
    churn_prob -= (streaming_tv == 1) * 0.05
    churn_prob -= (streaming_movies == 1) * 0.05
    
    # Ensure probabilities are between 0 and 1
    churn_prob = np.clip(churn_prob, 0, 1)
    
    # Generate churn based on probabilities
    churn = np.random.binomial(1, churn_prob, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'gender': genders,
        'age': ages,
        'tenure': tenure,
        'phone_service': phone_service,
        'multiple_lines': multiple_lines,
        'internet_service': internet_service,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'contract': contract_types,
        'paperless_billing': paperless_billing,
        'payment_method': payment_methods,
        'monthly_charges': np.round(monthly_charges, 2),
        'total_charges': np.round(total_charges, 2),
        'churn': churn
    })
    
    print(f"Generated data with churn rate: {churn.mean():.2%}")
    return df

def main():
    """Generate sample data for the MLOps demo."""
    # Ensure data directories exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/features', exist_ok=True)
    os.makedirs('data/external', exist_ok=True)
    
    # Generate main dataset
    df = generate_customer_churn_data(n_samples=5000)
    
    # Save main dataset
    output_path = 'data/raw/customer_churn.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ Main dataset saved to {output_path}")
    
    # Generate summary
    summary = {
        'generation_date': datetime.now().isoformat(),
        'rows': len(df),
        'columns': len(df.columns),
        'churn_rate': df['churn'].mean(),
        'features': df.columns.tolist()
    }
    
    print(f"📊 Dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"🎯 Churn rate: {df['churn'].mean():.2%}")
    print("✅ Sample data generation completed!")

if __name__ == "__main__":
    main()
