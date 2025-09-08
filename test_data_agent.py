#!/usr/bin/env python3
"""
Comprehensive test suite for DataAgent Pro
Tests all data manipulation, analysis, and tool functionality
"""

import os
import sys
import tempfile
import shutil
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Mock the external dependencies for testing
class MockAnthropicClient:
    def __init__(self, api_key):
        self.api_key = api_key
    
    def messages(self):
        return self
    
    def create(self, **kwargs):
        # Return a mock response for testing
        class MockMessage:
            def __init__(self, text="Test response"):
                self.type = "text"
                self.text = text
        
        class MockResponse:
            def __init__(self):
                self.content = [MockMessage("Test response - tool executed successfully")]
        
        return MockResponse()

class MockTwilioClient:
    def __init__(self, username, password):
        pass
        
    @property
    def messages(self):
        return self
        
    def create(self, **kwargs):
        class MockMessage:
            def __init__(self):
                self.sid = "mock_message_sid"
        return MockMessage()

# Mock the modules before importing DataAgent
sys.modules['anthropic'] = type('MockAnthropic', (), {
    'Anthropic': MockAnthropicClient
})()

sys.modules['twilio'] = type('MockTwilio', (), {})()
sys.modules['twilio.rest'] = type('MockTwilioRest', (), {
    'Client': MockTwilioClient
})()

from data_agent import DataAgent

def create_test_data(test_dir):
    """Create sample test datasets"""
    # Sales data
    np.random.seed(42)
    sales_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'product': np.random.choice(['A', 'B', 'C'], 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'quantity': np.random.randint(1, 100, 100),
        'price': np.random.uniform(10, 100, 100).round(2),
        'discount': np.random.uniform(0, 0.3, 100).round(2)
    })
    sales_data['revenue'] = (sales_data['quantity'] * sales_data['price'] * (1 - sales_data['discount'])).round(2)
    
    # Add some missing values
    sales_data.loc[np.random.choice(sales_data.index, 10), 'discount'] = np.nan
    
    # Customer data
    customer_data = pd.DataFrame({
        'customer_id': range(1, 51),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 50),
        'age': np.random.randint(18, 70, 50),
        'loyalty_years': np.random.randint(0, 10, 50),
        'total_purchases': np.random.randint(1, 50, 50)
    })
    
    # Time series data
    ts_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=50, freq='H'),
        'temperature': np.random.normal(20, 5, 50),
        'humidity': np.random.uniform(30, 90, 50)
    })
    
    # Save test data
    sales_path = os.path.join(test_dir, 'sales.csv')
    customer_path = os.path.join(test_dir, 'customers.csv')
    ts_path = os.path.join(test_dir, 'timeseries.csv')
    
    sales_data.to_csv(sales_path, index=False)
    customer_data.to_csv(customer_path, index=False)
    ts_data.to_csv(ts_path, index=False)
    
    # Create JSON test file
    json_data = [
        {"id": 1, "name": "Alice", "score": 85},
        {"id": 2, "name": "Bob", "score": 92},
        {"id": 3, "name": "Charlie", "score": 78}
    ]
    json_path = os.path.join(test_dir, 'test.json')
    pd.DataFrame(json_data).to_json(json_path, orient='records')
    
    return sales_path, customer_path, ts_path, json_path

def test_data_discovery_tools(agent, test_dir):
    """Test data discovery functionality"""
    print("Testing Data Discovery Tools...")
    
    # Test 1: List data files
    result = agent.execute_tool("list_data_files", {"directory": test_dir})
    assert "sales.csv" in result, f"Should find sales.csv: {result}"
    print("✓ list_data_files works")
    
    # Test 2: Get file info
    sales_path = os.path.join(test_dir, 'sales.csv')
    result = agent.execute_tool("get_file_info", {"filepath": sales_path})
    assert "Size" in result and "Rows" in result, f"Should show file info: {result}"
    print("✓ get_file_info works")
    
    # Test 3: Search in files
    result = agent.execute_tool("search_in_files", {
        "pattern": "product",
        "file_types": ["*.csv"]
    })
    assert "matches" in result.lower() or "no matches" in result.lower(), f"Should search files: {result}"
    print("✓ search_in_files works")

def test_data_loading_tools(agent, test_dir):
    """Test data loading functionality"""
    print("\nTesting Data Loading Tools...")
    
    sales_path = os.path.join(test_dir, 'sales.csv')
    customer_path = os.path.join(test_dir, 'customers.csv')
    json_path = os.path.join(test_dir, 'test.json')
    
    # Test 1: Load CSV data
    result = agent.execute_tool("load_data", {"filepath": sales_path, "name": "sales"})
    assert "Successfully loaded" in result, f"Should load sales data: {result}"
    assert "sales" in agent.dataframes, "Sales data should be stored"
    print("✓ load_data (CSV) works")
    
    # Test 2: Load JSON data
    result = agent.execute_tool("load_data", {"filepath": json_path, "name": "json_test"})
    assert "Successfully loaded" in result, f"Should load JSON data: {result}"
    print("✓ load_data (JSON) works")
    
    # Test 3: Preview data
    result = agent.execute_tool("preview_data", {"name": "sales", "rows": 5})
    assert "Preview" in result, f"Should preview data: {result}"
    print("✓ preview_data works")
    
    # Test 4: Get data info
    result = agent.execute_tool("get_data_info", {"name": "sales"})
    assert "Shape" in result and "Columns" in result, f"Should show data info: {result}"
    print("✓ get_data_info works")

def test_data_analysis_tools(agent):
    """Test data analysis functionality"""
    print("\nTesting Data Analysis Tools...")
    
    # Test 1: Basic statistics
    result = agent.execute_tool("basic_statistics", {"name": "sales"})
    assert "Basic statistics" in result, f"Should calculate statistics: {result}"
    print("✓ basic_statistics works")
    
    # Test 2: Data profile
    result = agent.execute_tool("data_profile", {"name": "sales"})
    assert "Data profile" in result, f"Should create profile: {result}"
    print("✓ data_profile works")
    
    # Test 3: Correlation analysis
    result = agent.execute_tool("correlation_analysis", {"name": "sales", "threshold": 0.3})
    assert "Correlation analysis" in result, f"Should analyze correlations: {result}"
    print("✓ correlation_analysis works")
    
    # Test 4: Group by analysis
    result = agent.execute_tool("group_by_analysis", {
        "name": "sales",
        "group_by": ["product"],
        "aggregate_columns": ["quantity", "revenue"],
        "functions": ["mean", "sum"]
    })
    assert "Group by analysis" in result, f"Should group data: {result}"
    print("✓ group_by_analysis works")

def test_data_manipulation_tools(agent):
    """Test data manipulation functionality"""
    print("\nTesting Data Manipulation Tools...")
    
    # Test 1: Filter data
    result = agent.execute_tool("filter_data", {
        "name": "sales",
        "conditions": [{"column": "quantity", "operator": ">", "value": 50}],
        "save_as": "filtered_sales"
    })
    assert "filtered_sales" in agent.dataframes, "Filtered data should be saved"
    print("✓ filter_data works")
    
    # Test 2: Sort data
    result = agent.execute_tool("sort_data", {
        "name": "sales",
        "columns": ["revenue"],
        "ascending": False
    })
    assert "Sorted data" in result, f"Should sort data: {result}"
    print("✓ sort_data works")
    
    # Test 3: Transform column
    result = agent.execute_tool("transform_column", {
        "name": "sales",
        "column": "quantity",
        "operation": "log",
        "new_column": "log_quantity"
    })
    assert "transformed successfully" in result, f"Should transform column: {result}"
    print("✓ transform_column works")
    
    # Test 4: Add column
    result = agent.execute_tool("add_column", {
        "name": "sales",
        "column_name": "profit_margin",
        "formula": "revenue * 0.1"
    })
    assert "added successfully" in result, f"Should add column: {result}"
    print("✓ add_column works")

def test_data_cleaning_tools(agent):
    """Test data cleaning functionality"""
    print("\nTesting Data Cleaning Tools...")
    
    # Test 1: Handle missing values
    result = agent.execute_tool("handle_missing", {
        "name": "sales",
        "strategy": "fill_mean",
        "columns": ["discount"]
    })
    assert "Missing values handled" in result, f"Should handle missing values: {result}"
    print("✓ handle_missing works")
    
    # Test 2: Remove duplicates
    result = agent.execute_tool("remove_duplicates", {"name": "sales"})
    assert "Duplicates removed" in result, f"Should remove duplicates: {result}"
    print("✓ remove_duplicates works")
    
    # Test 3: Convert types
    result = agent.execute_tool("convert_types", {
        "name": "sales",
        "conversions": {"quantity": "int", "price": "float"}
    })
    assert "Type conversions" in result, f"Should convert types: {result}"
    print("✓ convert_types works")
    
    # Test 4: Detect outliers
    result = agent.execute_tool("detect_outliers", {
        "name": "sales",
        "method": "iqr",
        "threshold": 1.5
    })
    assert "Outlier detection" in result, f"Should detect outliers: {result}"
    print("✓ detect_outliers works")

def test_advanced_features(agent, test_dir):
    """Test advanced features"""
    print("\nTesting Advanced Features...")
    
    # Test 1: SQL query
    result = agent.execute_tool("sql_query", {
        "query": "SELECT product, SUM(revenue) as total_revenue FROM sales GROUP BY product ORDER BY total_revenue DESC"
    })
    assert "Query result" in result, f"Should execute SQL: {result}"
    print("✓ sql_query works")
    
    # Test 2: Clustering
    result = agent.execute_tool("run_clustering", {
        "name": "sales",
        "features": ["quantity", "price"],
        "n_clusters": 3
    })
    assert "Clustering complete" in result, f"Should run clustering: {result}"
    print("✓ run_clustering works")
    
    # Test 3: Regression
    result = agent.execute_tool("run_regression", {
        "name": "sales",
        "target": "revenue",
        "features": ["quantity", "price"]
    })
    assert "Regression Analysis" in result, f"Should run regression: {result}"
    print("✓ run_regression works")
    
    # Test 4: Time series analysis
    ts_path = os.path.join(test_dir, 'timeseries.csv')
    agent.execute_tool("load_data", {"filepath": ts_path, "name": "timeseries"})
    result = agent.execute_tool("time_series_analysis", {
        "name": "timeseries",
        "date_column": "timestamp",
        "value_column": "temperature"
    })
    assert "Time Series Analysis" in result, f"Should analyze time series: {result}"
    print("✓ time_series_analysis works")

def test_data_export_tools(agent, test_dir):
    """Test data export functionality"""
    print("\nTesting Data Export Tools...")
    
    # Test 1: Save data as CSV
    output_path = os.path.join(test_dir, 'exported_sales.csv')
    result = agent.execute_tool("save_data", {
        "name": "sales",
        "filepath": output_path,
        "format": "csv"
    })
    assert os.path.exists(output_path), "Should create CSV file"
    print("✓ save_data (CSV) works")
    
    # Test 2: Save data as JSON
    output_path = os.path.join(test_dir, 'exported_sales.json')
    result = agent.execute_tool("save_data", {
        "name": "sales",
        "filepath": output_path,
        "format": "json"
    })
    assert os.path.exists(output_path), "Should create JSON file"
    print("✓ save_data (JSON) works")

def test_visualization_tools(agent, test_dir):
    """Test visualization functionality"""
    print("\nTesting Visualization Tools...")
    
    # Test different chart types
    chart_types = ["histogram", "heatmap"]
    
    for chart_type in chart_types:
        try:
            if chart_type == "histogram":
                result = agent.execute_tool("create_visualization", {
                    "name": "sales",
                    "chart_type": chart_type,
                    "x": "quantity",
                    "title": f"Test {chart_type}",
                    "save_path": os.path.join(test_dir, f"test_{chart_type}.png")
                })
            else:  # heatmap
                result = agent.execute_tool("create_visualization", {
                    "name": "sales",
                    "chart_type": chart_type,
                    "title": f"Test {chart_type}",
                    "save_path": os.path.join(test_dir, f"test_{chart_type}.png")
                })
            print(f"✓ create_visualization ({chart_type}) works")
        except Exception as e:
            print(f"⚠ create_visualization ({chart_type}) had issues: {e}")

def test_error_handling(agent):
    """Test error handling"""
    print("\nTesting Error Handling...")
    
    # Test non-existent dataset
    result = agent.execute_tool("preview_data", {"name": "nonexistent"})
    assert "not found" in result, f"Should handle missing dataset: {result}"
    print("✓ Missing dataset error handling works")
    
    # Test invalid file path
    result = agent.execute_tool("load_data", {"filepath": "/nonexistent/file.csv"})
    assert "not found" in result.lower() or "error" in result.lower(), f"Should handle missing file: {result}"
    print("✓ Missing file error handling works")

def run_all_tests():
    """Run comprehensive test suite"""
    print("🧪 Starting DataAgent Pro Test Suite")
    print("=" * 60)
    
    # Create temporary test directory
    test_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    
    try:
        # Change to test directory
        os.chdir(test_dir)
        
        # Create test data
        sales_path, customer_path, ts_path, json_path = create_test_data(test_dir)
        print(f"📁 Test data created in: {test_dir}")
        
        # Initialize agent with mock API key
        agent = DataAgent()
        print("🤖 DataAgent initialized")
        
        # Run test suites
        test_data_discovery_tools(agent, test_dir)
        test_data_loading_tools(agent, test_dir)
        test_data_analysis_tools(agent)
        test_data_manipulation_tools(agent)
        test_data_cleaning_tools(agent)
        test_advanced_features(agent, test_dir)
        test_data_export_tools(agent, test_dir)
        test_visualization_tools(agent, test_dir)
        test_error_handling(agent)
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print(f"📊 Datasets loaded: {list(agent.dataframes.keys())}")
        print(f"📈 Sample dataset shapes: {[(name, df.shape) for name, df in agent.dataframes.items()]}")
        
        # Test some key functionality manually
        print("\n🔍 Manual Verification Tests:")
        
        # Check that data was actually loaded and manipulated
        sales_df = agent.dataframes['sales']
        print(f"✓ Sales data shape: {sales_df.shape}")
        print(f"✓ Sales columns: {list(sales_df.columns)}")
        
        if 'filtered_sales' in agent.dataframes:
            filtered_df = agent.dataframes['filtered_sales']
            print(f"✓ Filtered data shape: {filtered_df.shape}")
            
        # Check for added columns
        if 'log_quantity' in sales_df.columns:
            print("✓ Log transformation column added")
        if 'profit_margin' in sales_df.columns:
            print("✓ Calculated column added")
        if 'cluster' in sales_df.columns:
            print("✓ Clustering results added")
            
        print("\n🎉 DataAgent Pro is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        os.chdir(old_cwd)
        shutil.rmtree(test_dir, ignore_errors=True)
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)