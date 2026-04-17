"""
Generate sample datasets for testing the AI Consultant.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sales_data(n_rows=1000, output_path='sales_data.csv'):
    """
    Generate realistic sales data with trends and segments.
    """
    np.random.seed(42)

    # Date range
    dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='D')

    # Regions with different performance levels
    regions = np.random.choice(['North', 'South', 'East', 'West'], n_rows,
                               p=[0.25, 0.25, 0.25, 0.25])

    # Products
    products = np.random.choice(['Standard', 'Premium', 'Enterprise'], n_rows,
                              p=[0.5, 0.35, 0.15])

    # Sales reps
    sales_reps = [f"Rep_{i}" for i in range(1, 11)]
    reps = np.random.choice(sales_reps, n_rows)

    # Generate sales amount with patterns
    base_amount = 1000
    sales = []

    for i in range(n_rows):
        amount = base_amount

        # Region multiplier (West underperforming)
        if regions[i] == 'West':
            amount *= np.random.uniform(0.6, 0.9)
        elif regions[i] == 'East':
            amount *= np.random.uniform(1.1, 1.4)

        # Product multiplier
        if products[i] == 'Premium':
            amount *= 2.5
        elif products[i] == 'Enterprise':
            amount *= 5

        # Time trend (slight increase over time)
        day_factor = 1 + (i / n_rows) * 0.1
        amount *= day_factor

        # Add noise
        amount *= np.random.uniform(0.8, 1.2)

        sales.append(round(amount, 2))

    # Customer tenure (correlated with higher sales)
    tenure_months = np.random.exponential(12, n_rows).clip(1, 60).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'region': regions,
        'product': products,
        'sales_rep': reps,
        'sales_amount': sales,
        'customer_tenure_months': tenure_months
    })

    # Create correlation: longer tenure = higher sales
    df['sales_amount'] = df['sales_amount'] * (1 + df['customer_tenure_months'] / 100)

    df.to_csv(output_path, index=False)
    print(f"Generated {n_rows} rows of sales data to {output_path}")
    return df


def generate_customer_data(n_rows=500, output_path='customer_data.csv'):
    """
    Generate customer data for RFM analysis.
    """
    np.random.seed(43)

    customer_ids = range(1000, 1000 + n_rows)

    # Generate random dates over last year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    data = []
    for cust_id in customer_ids:
        n_transactions = np.random.poisson(5) + 1

        for _ in range(n_transactions):
            transaction_date = start_date + timedelta(days=np.random.randint(0, 365))
            amount = np.random.lognormal(4, 0.5)  # Skewed distribution

            data.append({
                'customer_id': cust_id,
                'transaction_date': transaction_date.strftime('%Y-%m-%d'),
                'amount': round(amount, 2),
                'channel': np.random.choice(['Online', 'Store', 'Phone'], p=[0.5, 0.4, 0.1])
            })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions for {n_rows} customers to {output_path}")
    return df


def generate_employee_data(n_rows=200, output_path='employee_data.csv'):
    """
    Generate HR data for analysis.
    """
    np.random.seed(44)

    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Operations']
    levels = ['Junior', 'Mid', 'Senior', 'Lead', 'Director']

    data = []
    for i in range(n_rows):
        dept = np.random.choice(departments)
        level = np.random.choice(levels, p=[0.3, 0.35, 0.2, 0.1, 0.05])

        # Salary based on level
        base_salary = {'Junior': 50000, 'Mid': 75000, 'Senior': 100000,
                       'Lead': 130000, 'Director': 180000}[level]

        # Department adjustment
        if dept == 'Engineering':
            base_salary *= 1.2
        elif dept == 'Sales':
            base_salary *= 0.9

        # Performance score (0-100)
        performance = np.random.beta(7, 2) * 100

        # Satisfaction score
        satisfaction = np.random.beta(6, 3) * 100

        data.append({
            'employee_id': i + 1,
            'department': dept,
            'level': level,
            'salary': round(base_salary, -3),
            'performance_score': round(performance, 1),
            'satisfaction_score': round(satisfaction, 1),
            'tenure_years': np.random.exponential(3) + 0.5,
            'promoted_last_year': np.random.choice([0, 1], p=[0.85, 0.15])
        })

    df = pd.DataFrame(data)

    # Create correlation: higher performance <-> higher satisfaction
    df['satisfaction_score'] = df['satisfaction_score'] * 0.7 + df['performance_score'] * 0.3

    df.to_csv(output_path, index=False)
    print(f"Generated {n_rows} employee records to {output_path}")
    return df


if __name__ == "__main__":
    import os

    # Create sample_data directory if needed
    os.makedirs('sample_data', exist_ok=True)

    # Generate all sample datasets
    generate_sales_data(1000, 'sample_data/sales_data.csv')
    generate_customer_data(500, 'sample_data/customer_data.csv')
    generate_employee_data(200, 'sample_data/employee_data.csv')

    print("\n✅ Sample datasets generated successfully!")
    print("\nTo test the AI Consultant:")
    print("1. cd mckinsey_consultant")
    print("2. streamlit run app.py")
    print("3. Upload one of the sample CSV files")
