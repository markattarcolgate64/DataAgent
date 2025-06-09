# DataAgent Pro - Advanced Data Analysis Assistant

An AI-powered data analysis assistant built with Anthropic's Claude API, featuring comprehensive data manipulation, analysis, and visualization capabilities.

## Key Enhancement: Multi-Tool Workflows 🚀

The DataAgent now **automatically uses multiple tools** in a single response to provide comprehensive analyses. No more one-tool-at-a-time interactions!

### Examples of Multi-Tool Workflows:
- **"Load data.csv"** → Automatically loads, previews, shows info, and calculates statistics
- **"Clean this data"** → Detects outliers, handles missing values, removes duplicates, and saves
- **"Analyze the data"** → Runs profiling, correlations, outlier detection, and visualizations
- **"Show me what's interesting"** → Executes complete exploratory data analysis workflow

## Features

### Data Discovery & Search
- **List Data Files**: Discover all available data files in your workspace
- **Search in Files**: Find specific patterns across multiple data files
- **Get File Info**: Retrieve detailed metadata about data files

### Data Loading & Preview
- **Multi-Format Support**: Load CSV, Excel, JSON, and Parquet files
- **Smart Preview**: View head/tail of datasets with formatted output
- **Data Info**: Get comprehensive information about loaded datasets

### Data Analysis
- **Basic Statistics**: Calculate mean, median, std dev, and more
- **Data Profiling**: Generate comprehensive column-by-column profiles
- **Correlation Analysis**: Find relationships between numeric variables
- **Group By Analysis**: Aggregate data with multiple functions
- **Pivot Tables**: Create pivot table summaries

### Data Manipulation
- **Filtering**: Apply complex conditions to filter data
- **Sorting**: Sort by single or multiple columns
- **Merging**: Join datasets with various merge strategies
- **Aggregation**: Apply custom aggregation functions
- **Column Transformation**: Log, sqrt, normalize, scale, and more
- **Column Operations**: Add, remove, and rename columns

### Data Cleaning
- **Missing Value Handling**: Multiple strategies including drop, fill, interpolate
- **Duplicate Removal**: Remove duplicate rows with flexible options
- **Type Conversion**: Convert between data types safely
- **Outlier Detection**: Identify outliers using IQR or Z-score methods

### Data Export
- **Multi-Format Export**: Save to CSV, Excel, JSON, or Parquet
- **Filtered Export**: Export subsets of data based on conditions

### Advanced Features
- **SQL Queries**: Execute SQL queries on loaded datasets using DuckDB
- **Data Visualization**: Create various charts (scatter, line, bar, heatmap, etc.)
- **Machine Learning**: 
  - K-means clustering
  - Linear regression analysis
- **Time Series Analysis**: Analyze temporal data with rolling statistics

### Additional Features
- **SMS Notifications**: Send SMS alerts via Twilio integration
- **Interactive Chat**: Natural language interface to all features
- **Conversation Memory**: Maintains context across interactions

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Create a .env file
ANTHROPIC_API_KEY=your-api-key-here
TWILIO_ACCOUNT_SID=your-twilio-sid  # Optional
TWILIO_AUTH_TOKEN=your-twilio-token  # Optional
TWILIO_PHONE_NUMBER=your-twilio-phone  # Optional
```

## Usage

### Basic Usage

```python
from data_agent import DataAgent

# Initialize the agent
agent = DataAgent()

# Start interactive chat
agent.interactive_chat()
```

### Example Commands

In the interactive chat, you can use natural language commands like:

- "List all CSV files in the current directory"
- "Load the sales.csv file"
- "Show me basic statistics for all numeric columns"
- "Filter the data where revenue > 1000"
- "Create a scatter plot of price vs quantity"
- "Find correlations between all numeric columns"
- "Handle missing values by filling with the mean"
- "Run clustering analysis with 3 clusters"
- "Save the filtered data as filtered_sales.xlsx"

### Programmatic Usage

```python
# Load data
response = agent.send_message("Load the file data.csv")

# Analyze data
response = agent.send_message("Show me a data profile")

# Transform data
response = agent.send_message("Remove duplicates and handle missing values")

# Visualize
response = agent.send_message("Create a correlation heatmap")
```

## Available Tools

### Data Discovery
- `list_data_files`: Find all data files in workspace
- `search_in_files`: Search for patterns in files
- `get_file_info`: Get file metadata

### Data Loading
- `load_data`: Load data from various formats
- `preview_data`: Preview dataset rows
- `get_data_info`: Get dataset information

### Data Analysis
- `basic_statistics`: Calculate descriptive statistics
- `data_profile`: Generate comprehensive profile
- `correlation_analysis`: Find correlations
- `group_by_analysis`: Group and aggregate data
- `pivot_table`: Create pivot tables

### Data Manipulation
- `filter_data`: Filter with conditions
- `sort_data`: Sort by columns
- `merge_data`: Join datasets
- `aggregate_data`: Apply aggregations
- `transform_column`: Transform with functions
- `add_column`: Add new columns
- `remove_column`: Remove columns
- `rename_columns`: Rename columns

### Data Cleaning
- `handle_missing`: Handle missing values
- `remove_duplicates`: Remove duplicate rows
- `convert_types`: Convert data types
- `detect_outliers`: Find outliers

### Data Export
- `save_data`: Save to file
- `export_filtered`: Export filtered data

### Advanced Features
- `sql_query`: Execute SQL queries
- `create_visualization`: Generate charts
- `run_clustering`: K-means clustering
- `run_regression`: Linear regression
- `time_series_analysis`: Analyze time series

## Requirements

- Python 3.8+
- Anthropic API key
- Optional: Twilio credentials for SMS functionality

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.