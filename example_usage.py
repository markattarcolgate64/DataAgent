"""
Example usage of DataAgent Pro
Demonstrates various data manipulation and analysis capabilities
"""

from data_agent import DataAgent
import pandas as pd
import numpy as np

def create_sample_data():
    """Create sample datasets for demonstration"""
    
    # Create sales data
    np.random.seed(42)
    sales_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=365, freq='D'),
        'product': np.random.choice(['A', 'B', 'C', 'D'], 365),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
        'quantity': np.random.randint(10, 100, 365),
        'price': np.random.uniform(10, 100, 365).round(2),
        'discount': np.random.uniform(0, 0.3, 365).round(2)
    })
    sales_data['revenue'] = (sales_data['quantity'] * sales_data['price'] * (1 - sales_data['discount'])).round(2)
    
    # Add some missing values
    sales_data.loc[np.random.choice(sales_data.index, 20), 'discount'] = np.nan
    
    # Create customer data
    customer_data = pd.DataFrame({
        'customer_id': range(1, 101),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'age': np.random.randint(18, 70, 100),
        'loyalty_years': np.random.randint(0, 10, 100),
        'total_purchases': np.random.randint(1, 50, 100)
    })
    
    # Save sample data
    sales_data.to_csv('sample_sales.csv', index=False)
    customer_data.to_csv('sample_customers.csv', index=False)
    print("✅ Sample data files created: sample_sales.csv, sample_customers.csv")

def main():
    # Create sample data
    create_sample_data()
    
    # Initialize DataAgent
    print("\n🚀 Initializing DataAgent...")
    agent = DataAgent()
    
    # Example 1: Data Discovery
    print("\n📁 Example 1: Data Discovery")
    print("-" * 50)
    response = agent.send_message("List all CSV files in the current directory")
    print(response)
    
    # Example 2: Load Data
    print("\n📊 Example 2: Loading Data")
    print("-" * 50)
    response = agent.send_message("Load the file sample_sales.csv and call it 'sales'")
    print(response)
    
    # Example 3: Data Preview
    print("\n👀 Example 3: Data Preview")
    print("-" * 50)
    response = agent.send_message("Show me the first 5 rows of the sales data")
    print(response)
    
    # Example 4: Data Info
    print("\n📋 Example 4: Data Information")
    print("-" * 50)
    response = agent.send_message("Give me detailed information about the sales dataset")
    print(response)
    
    # Example 5: Basic Statistics
    print("\n📈 Example 5: Basic Statistics")
    print("-" * 50)
    response = agent.send_message("Calculate basic statistics for the numeric columns in sales")
    print(response)
    
    # Example 6: Data Cleaning - Missing Values
    print("\n🧹 Example 6: Handle Missing Values")
    print("-" * 50)
    response = agent.send_message("Check for missing values in sales and fill them with the mean")
    print(response)
    
    # Example 7: Data Filtering
    print("\n🔍 Example 7: Filter Data")
    print("-" * 50)
    response = agent.send_message("Filter the sales data to show only records where revenue > 2000 and save it as 'high_revenue'")
    print(response)
    
    # Example 8: Group By Analysis
    print("\n🔢 Example 8: Group By Analysis")
    print("-" * 50)
    response = agent.send_message("Group the sales data by product and calculate the total revenue and average price for each product")
    print(response)
    
    # Example 9: Load Second Dataset
    print("\n📊 Example 9: Load Second Dataset")
    print("-" * 50)
    response = agent.send_message("Load sample_customers.csv as 'customers'")
    print(response)
    
    # Example 10: SQL Query
    print("\n💾 Example 10: SQL Query")
    print("-" * 50)
    response = agent.send_message("""
    Run this SQL query:
    SELECT product, region, SUM(revenue) as total_revenue 
    FROM sales 
    WHERE quantity > 50 
    GROUP BY product, region 
    ORDER BY total_revenue DESC 
    LIMIT 10
    """)
    print(response)
    
    # Example 11: Correlation Analysis
    print("\n📊 Example 11: Correlation Analysis")
    print("-" * 50)
    response = agent.send_message("Find correlations between numeric columns in the sales data")
    print(response)
    
    # Example 12: Time Series Analysis
    print("\n⏰ Example 12: Time Series Analysis")
    print("-" * 50)
    response = agent.send_message("Perform time series analysis on the sales data using the date column and revenue as the value")
    print(response)
    
    # Example 13: Data Export
    print("\n💾 Example 13: Export Data")
    print("-" * 50)
    response = agent.send_message("Save the high_revenue dataset as an Excel file called 'high_revenue_sales.xlsx'")
    print(response)
    
    # Example 14: Advanced - Clustering
    print("\n🤖 Example 14: Clustering Analysis")
    print("-" * 50)
    response = agent.send_message("Run clustering analysis on the customers data using age and total_purchases features with 3 clusters")
    print(response)
    
    print("\n✅ Demo completed! Check the generated files in your directory.")
    print("\n💡 Tip: You can start an interactive session with: agent.interactive_chat()")

if __name__ == "__main__":
    main() 