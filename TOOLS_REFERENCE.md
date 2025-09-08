# DataAgent Tools Quick Reference

#### Data Loading Workflow
```
User: "Load sales.csv"
Agent automatically executes:
1. load_data → Load the file
2. preview_data → Show first rows
3. get_data_info → Display structure
4. basic_statistics → Calculate stats
```

#### Data Cleaning Workflow
```
User: "Clean the data"
Agent automatically executes:
1. detect_outliers → Find anomalies
2. handle_missing → Fill/drop nulls
3. remove_duplicates → Remove dupes
4. save_data → Save cleaned version
```

#### Analysis Workflow
```
User: "Analyze this dataset"
Agent automatically executes:
1. data_profile → Comprehensive profiling
2. correlation_analysis → Find relationships
3. group_by_analysis → Aggregate insights
4. create_visualization → Generate charts
```

#### Complete EDA Workflow
```
User: "Show me what's interesting"
Agent automatically executes:
1. basic_statistics → Summary stats
2. correlation_analysis → Relationships
3. detect_outliers → Anomalies
4. group_by_analysis → Patterns
5. create_visualization → Visual insights
```

## Data Discovery & Search

### list_data_files
Find all data files in workspace
```python
params = {
    "patterns": ["*.csv", "*.xlsx"],  # File patterns to search
    "directory": "."                   # Directory to search in
}
```

### search_in_files
Search for patterns in data files
```python
params = {
    "pattern": "search_term",          # Pattern to search for
    "file_types": ["*.csv", "*.txt"]   # File types to search in
}
```

### get_file_info
Get detailed file information
```python
params = {
    "filepath": "data.csv"             # Path to file
}
```

## Data Loading & Preview

### load_data
Load data from various formats
```python
params = {
    "filepath": "data.csv",            # Path to file
    "name": "my_data",                 # Name to store as
    "sheet": 0                         # For Excel files
}
```

### preview_data
Preview dataset rows
```python
params = {
    "name": "my_data",                 # Dataset name
    "rows": 10,                        # Number of rows
    "position": "head"                 # "head" or "tail"
}
```

### get_data_info
Get comprehensive dataset information
```python
params = {
    "name": "my_data"                  # Dataset name
}
```

## Data Analysis

### basic_statistics
Calculate descriptive statistics
```python
params = {
    "name": "my_data",                 # Dataset name
    "columns": ["col1", "col2"]        # Specific columns (optional)
}
```

### data_profile
Generate comprehensive data profile
```python
params = {
    "name": "my_data"                  # Dataset name
}
```

### correlation_analysis
Find correlations between columns
```python
params = {
    "name": "my_data",                 # Dataset name
    "method": "pearson",               # "pearson", "spearman", "kendall"
    "threshold": 0.5                   # Minimum correlation to report
}
```

### group_by_analysis
Group and aggregate data
```python
params = {
    "name": "my_data",                 # Dataset name
    "group_by": ["category"],          # Columns to group by
    "aggregate_columns": ["value"],    # Columns to aggregate
    "functions": ["mean", "sum"]       # Aggregation functions
}
```

### pivot_table
Create pivot table
```python
params = {
    "name": "my_data",                 # Dataset name
    "index": "category",               # Row index
    "columns": "month",                # Column index
    "values": "sales",                 # Values to aggregate
    "aggfunc": "mean"                  # Aggregation function
}
```

## Data Manipulation

### filter_data
Filter data with conditions
```python
params = {
    "name": "my_data",                 # Dataset name
    "conditions": [
        {
            "column": "price",
            "operator": ">",           # ==, !=, >, <, >=, <=, contains, in, between
            "value": 100
        }
    ],
    "save_as": "filtered_data"         # Save filtered data as
}
```

### sort_data
Sort by columns
```python
params = {
    "name": "my_data",                 # Dataset name
    "columns": ["date", "value"],      # Columns to sort by
    "ascending": True,                 # Sort order
    "save_as": "sorted_data"           # Save sorted data as
}
```

### merge_data
Join datasets
```python
params = {
    "left_dataset": "sales",           # Left dataset
    "right_dataset": "customers",      # Right dataset
    "on": "customer_id",               # Join column(s)
    "how": "inner",                    # "inner", "left", "right", "outer"
    "save_as": "merged_data"           # Save result as
}
```

### transform_column
Transform column values
```python
params = {
    "name": "my_data",                 # Dataset name
    "column": "price",                 # Column to transform
    "operation": "log",                # log, sqrt, square, normalize, scale, rank, pct_change, cumsum, diff
    "new_column": "log_price"          # New column name
}
```

### add_column
Add new column
```python
params = {
    "name": "my_data",                 # Dataset name
    "column_name": "total",            # New column name
    "formula": "price * quantity",     # Formula using existing columns
    "value": 0                         # Or constant value
}
```

### rename_columns
Rename columns
```python
params = {
    "name": "my_data",                 # Dataset name
    "mappings": {
        "old_name": "new_name",
        "old_name2": "new_name2"
    }
}
```

## Data Cleaning

### handle_missing
Handle missing values
```python
params = {
    "name": "my_data",                 # Dataset name
    "strategy": "fill_mean",           # drop, fill_value, fill_mean, fill_median, fill_mode, forward_fill, backward_fill, interpolate
    "columns": ["col1", "col2"],       # Specific columns (optional)
    "value": 0                         # For fill_value strategy
}
```

### remove_duplicates
Remove duplicate rows
```python
params = {
    "name": "my_data",                 # Dataset name
    "subset": ["col1", "col2"],        # Columns to check for duplicates
    "keep": "first"                    # "first", "last", or false
}
```

### convert_types
Convert data types
```python
params = {
    "name": "my_data",                 # Dataset name
    "conversions": {
        "column1": "int",              # int, float, string, datetime, category, bool
        "column2": "datetime"
    }
}
```

### detect_outliers
Find outliers
```python
params = {
    "name": "my_data",                 # Dataset name
    "method": "iqr",                   # "iqr" or "zscore"
    "columns": ["price", "quantity"],  # Specific columns (optional)
    "threshold": 1.5                   # IQR multiplier or z-score threshold
}
```

## Data Export

### save_data
Save dataset to file
```python
params = {
    "name": "my_data",                 # Dataset name
    "filepath": "output.csv",          # Output path
    "format": "csv"                    # "csv", "excel", "json", "parquet"
}
```

### export_filtered
Export filtered data
```python
params = {
    "name": "my_data",                 # Dataset name
    "conditions": [...],               # Filter conditions
    "filepath": "filtered.csv",        # Output path
    "format": "csv"                    # "csv" or "excel"
}
```

## Advanced Features

### sql_query
Execute SQL on datasets
```python
params = {
    "query": "SELECT * FROM my_data WHERE price > 100",
    "save_as": "query_result"          # Save result as (optional)
}
```

### create_visualization
Generate charts
```python
params = {
    "name": "my_data",                 # Dataset name
    "chart_type": "scatter",           # scatter, line, bar, histogram, heatmap, box
    "x": "date",                       # X-axis column
    "y": "price",                      # Y-axis column
    "title": "Price over Time",        # Chart title
    "save_path": "chart.png"           # Save path (optional)
}
```

### run_clustering
K-means clustering
```python
params = {
    "name": "my_data",                 # Dataset name
    "features": ["age", "income"],     # Feature columns
    "n_clusters": 3,                   # Number of clusters
    "save_column": "cluster"           # Column to save cluster labels
}
```

### run_regression
Linear regression
```python
params = {
    "name": "my_data",                 # Dataset name
    "target": "price",                 # Target column
    "features": ["size", "rooms"]      # Feature columns
}
```

### time_series_analysis
Analyze time series
```python
params = {
    "name": "my_data",                 # Dataset name
    "date_column": "date",             # Date column
    "value_column": "sales",           # Value column
    "frequency": "D"                   # Frequency: D (daily), W (weekly), M (monthly)
}
```

## Communication

### send_sms
Send SMS via Twilio
```python
params = {
    "to_number": "+1234567890",        # Recipient phone (E.164 format)
    "message": "Your analysis is ready!" # Message content
}
``` 