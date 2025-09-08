# DataAgent

AI-powered data analysis assistant built with Anthropic's Claude API. Works as both a command-line tool and web interface for comprehensive data manipulation, analysis, and visualization.

## Quick Start

### Web App (Recommended)
```bash
pip install -r requirements.txt
python run_web.py
# Open http://localhost:8000
```

### Command Line Interface
```python
from data_agent import DataAgent
agent = DataAgent()
agent.interactive_chat()
```

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

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your API key:**
   Create a `.env` file:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   TWILIO_ACCOUNT_SID=your-twilio-sid  # Optional for SMS
   TWILIO_AUTH_TOKEN=your-twilio-token  # Optional for SMS  
   TWILIO_PHONE_NUMBER=your-twilio-phone  # Optional for SMS
   ```

## Usage

### Web Interface

1. **Run the web interface:**
   ```bash
   python run_web.py
   ```

2. **Open your browser:**
   Go to `http://localhost:8000`

3. **Upload & Chat:**
   - Drag and drop a data file (CSV, Excel, JSON, or Parquet)
   - Chat with your data using natural language
   - Export processed results

### Command Line Interface

```python
from data_agent import DataAgent

# Initialize the agent
agent = DataAgent()

# Start interactive chat
agent.interactive_chat()
```

### Example Commands

**Basic Analysis:**
- "Show me basic statistics"
- "What's the shape of this data?"
- "Tell me about missing values"

**Data Exploration:**
- "What are the correlations between columns?"
- "Show me outliers in the data"
- "Create a data profile"

**Filtering & Manipulation:**
- "Filter rows where sales > 1000"
- "Remove duplicate rows"
- "Fill missing values with the mean"

**Visualizations:**
- "Create a scatter plot of X vs Y"
- "Show me a correlation heatmap"
- "Make a histogram of the price column"

**Advanced Analysis:**
- "Run clustering analysis with 3 clusters"
- "Perform linear regression on this data"
- "Show time series trends"

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

## Troubleshooting

### Common Issues

**"DataAgent initialization failed"**
- Check that `ANTHROPIC_API_KEY` is set in your `.env` file
- Verify the API key is valid

**"File type not supported"**
- Only CSV, Excel, JSON, and Parquet files are supported
- Check the file extension is correct

**"Upload failed"**  
- File might be too large (50MB limit)
- Check file isn't corrupted
- Ensure sufficient disk space

**Web interface won't load**
- Check if port 5000 is already in use
- Try running with `python web_app.py` directly
- Check console for error messages

## Requirements

- Python 3.8+
- Anthropic API key
- Optional: Twilio credentials for SMS functionality

## Technical Details

- **Backend:** Flask web server with REST API
- **Frontend:** Vanilla JavaScript with modern CSS
- **Data Processing:** Pandas, NumPy, Scikit-learn, Plotly
- **Database:** DuckDB for SQL queries
- **File Storage:** Temporary uploads with session cleanup
- **Security:** Isolated user sessions, server-side API keys

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.