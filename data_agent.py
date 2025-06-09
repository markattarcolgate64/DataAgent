import os
import glob
import json
import yaml
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Core libraries
from anthropic import Anthropic
from dotenv import load_dotenv
from twilio.rest import Client

# Data manipulation libraries
import pandas as pdp
import numpy as np
import duckdb
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# File handling
import pyarrow.parquet as pq
from openpyxl import load_workbook
from tabulate import tabulate

# Load environment variables from .env file
load_dotenv()

class DataAgent:
    def __init__(self, api_key=None, twilio_sid=None, twilio_token=None, twilio_phone=None):
        self.model = "claude-sonnet-4-20250514"
        
        # Try to get API key from parameter or environment
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        
        self.client = Anthropic(api_key=self.api_key)
        self.conversation_history = []
        self.tools = []
        
        # Data storage
        self.dataframes = {}  # Store loaded dataframes
        self.current_df = None  # Current working dataframe
        self.current_df_name = None
        
        # Twilio setup
        self.twilio_sid = twilio_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_token = twilio_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_phone = twilio_phone or os.getenv("TWILIO_PHONE_NUMBER")
        
        if self.twilio_sid and self.twilio_token:
            self.twilio_client = Client(self.twilio_sid, self.twilio_token)
        else:
            self.twilio_client = None
            print("Warning: Twilio credentials not found. SMS functionality disabled.")
        
        # Initialize DuckDB connection
        self.duckdb_conn = duckdb.connect(':memory:')
        
        self.set_tools()
    
    def send_message(self, message, system_prompt=None):
        """Send a message to Claude and get a response"""
        try:
            messages = self.conversation_history + [{"role": "user", "content": message}]
            
            enhanced_system_prompt = """You are a powerful data analysis assistant with access to comprehensive data manipulation tools.

IMPORTANT INSTRUCTIONS FOR MULTI-TOOL USAGE:
1. **Be Proactive**: When a user's request implies multiple steps, use ALL necessary tools in a single response.
2. **Complete Workflows**: Don't stop at just one tool - think about the complete workflow and execute all steps.
3. **Common Multi-Tool Patterns**:
   - "Load and analyze" → Use load_data + preview_data + basic_statistics/data_profile
   - "Clean the data" → Use handle_missing + remove_duplicates + detect_outliers
   - "Full analysis" → Use multiple analysis tools (statistics, correlations, profile)
   - "Prepare for modeling" → Load + clean + transform + save
   - "Explore the data" → Load + info + preview + statistics + correlations

4. **Examples of Multi-Tool Responses**:
   - User: "Load sales.csv and show me what's in it"
     → You should: load_data, then preview_data, then get_data_info, then basic_statistics
   - User: "Clean this dataset"
     → You should: detect_outliers, handle_missing, remove_duplicates, and save_data
   - User: "Analyze the relationship between columns"
     → You should: correlation_analysis, create_visualization (scatter plots), group_by_analysis

5. **Always Think Ahead**: After each tool result, consider what additional tools would provide valuable insights.

Remember: Users expect comprehensive responses. Use multiple tools to deliver complete analyses."""
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system=system_prompt or enhanced_system_prompt,
                messages=messages,
                tools=self.tools if self.tools else None
            )
            
            assistant_message = ""

            for mssg in response.content:
                if mssg.type == "text":
                    assistant_message += mssg.text
                elif mssg.type == "tool_use":
                    tool_result = self.execute_tool(mssg.name, mssg.input)
                    assistant_message += f"\n[Tool: {mssg.name}] {tool_result}"
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
    
    def execute_tool(self, tool_name, tool_input):
        """Execute the requested tool"""
        tool_methods = {
            # Data Discovery
            "list_data_files": self.list_data_files,
            "search_in_files": self.search_in_files,
            "get_file_info": self.get_file_info,
            
            # Data Loading
            "load_data": self.load_data,
            "preview_data": self.preview_data,
            "get_data_info": self.get_data_info,
            
            # Data Analysis
            "basic_statistics": self.basic_statistics,
            "data_profile": self.data_profile,
            "correlation_analysis": self.correlation_analysis,
            "group_by_analysis": self.group_by_analysis,
            "pivot_table": self.pivot_table,
            
            # Data Manipulation
            "filter_data": self.filter_data,
            "sort_data": self.sort_data,
            "merge_data": self.merge_data,
            "aggregate_data": self.aggregate_data,
            "transform_column": self.transform_column,
            "add_column": self.add_column,
            "remove_column": self.remove_column,
            "rename_columns": self.rename_columns,
            
            # Data Cleaning
            "handle_missing": self.handle_missing,
            "remove_duplicates": self.remove_duplicates,
            "convert_types": self.convert_types,
            "detect_outliers": self.detect_outliers,
            
            # Data Export
            "save_data": self.save_data,
            "export_filtered": self.export_filtered,
            
            # Advanced Features
            "sql_query": self.sql_query,
            "create_visualization": self.create_visualization,
            "run_clustering": self.run_clustering,
            "run_regression": self.run_regression,
            "time_series_analysis": self.time_series_analysis,
            
            # Original tools
            "send_sms": self.send_sms,
        }
        
        if tool_name in tool_methods:
            return tool_methods[tool_name](tool_input)
        else:
            return f"Tool not found: {tool_name}"
    
    # Data Discovery Tools
    def list_data_files(self, params):
        """List all available data files in the workspace"""
        try:
            patterns = params.get("patterns", ["*.csv", "*.xlsx", "*.json", "*.parquet", "*.txt"])
            directory = params.get("directory", ".")
            
            all_files = []
            for pattern in patterns:
                files = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
                all_files.extend(files)
            
            if not all_files:
                return "No data files found in the specified directory."
            
            file_info = []
            for file in all_files:
                stat = os.stat(file)
                file_info.append({
                    "path": file,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
            
            return f"Found {len(file_info)} data files:\n" + tabulate(file_info, headers="keys", tablefmt="grid")
        except Exception as e:
            return f"Error listing files: {str(e)}"
    
    def search_in_files(self, params):
        """Search for specific patterns in data files"""
        try:
            pattern = params.get("pattern")
            file_types = params.get("file_types", ["*.csv", "*.txt"])
            
            if not pattern:
                return "Please provide a search pattern"
            
            results = []
            for file_type in file_types:
                files = glob.glob(f"**/{file_type}", recursive=True)
                for file in files:
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            for i, line in enumerate(lines):
                                if pattern.lower() in line.lower():
                                    results.append({
                                        "file": file,
                                        "line": i + 1,
                                        "content": line.strip()[:100] + "..." if len(line) > 100 else line.strip()
                                    })
                    except:
                        continue
            
            if not results:
                return f"No matches found for pattern: {pattern}"
            
            return f"Found {len(results)} matches:\n" + tabulate(results[:50], headers="keys", tablefmt="grid")
        except Exception as e:
            return f"Error searching files: {str(e)}"
    
    def get_file_info(self, params):
        """Get detailed information about a specific file"""
        try:
            filepath = params.get("filepath")
            if not filepath:
                return "Please provide a filepath"
            
            if not os.path.exists(filepath):
                return f"File not found: {filepath}"
            
            stat = os.stat(filepath)
            info = {
                "Path": filepath,
                "Size": f"{stat.st_size:,} bytes ({round(stat.st_size / (1024 * 1024), 2)} MB)",
                "Created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                "Modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "Extension": os.path.splitext(filepath)[1],
            }
            
            # Try to get row count for data files
            try:
                if filepath.endswith('.csv'):
                    df = pd.read_csv(filepath, nrows=0)
                    row_count = sum(1 for _ in open(filepath)) - 1
                    info["Rows"] = f"{row_count:,}"
                    info["Columns"] = len(df.columns)
                elif filepath.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(filepath, nrows=0)
                    info["Columns"] = len(df.columns)
            except:
                pass
            
            return tabulate(info.items(), headers=["Property", "Value"], tablefmt="grid")
        except Exception as e:
            return f"Error getting file info: {str(e)}"
    
    # Data Loading Tools
    def load_data(self, params):
        """Load data from various file formats"""
        try:
            filepath = params.get("filepath")
            name = params.get("name", os.path.basename(filepath).split('.')[0])
            
            if not filepath:
                return "Please provide a filepath"
            
            if not os.path.exists(filepath):
                return f"File not found: {filepath}"
            
            # Load based on file extension
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                sheet = params.get("sheet", 0)
                df = pd.read_excel(filepath, sheet_name=sheet)
            elif filepath.endswith('.json'):
                df = pd.read_json(filepath)
            elif filepath.endswith('.parquet'):
                df = pd.read_parquet(filepath)
            else:
                return f"Unsupported file format: {filepath}"
            
            # Store the dataframe
            self.dataframes[name] = df
            self.current_df = df
            self.current_df_name = name
            
            # Register with DuckDB
            self.duckdb_conn.register(name, df)
            
            return f"Successfully loaded {filepath} as '{name}':\n- Shape: {df.shape}\n- Columns: {list(df.columns)}\n- Memory: {df.memory_usage().sum() / 1024**2:.2f} MB"
        except Exception as e:
            return f"Error loading data: {str(e)}"
    
    def preview_data(self, params):
        """Preview first/last N rows of data"""
        try:
            name = params.get("name", self.current_df_name)
            rows = params.get("rows", 10)
            position = params.get("position", "head")  # head or tail
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            
            if position == "head":
                preview = df.head(rows)
            else:
                preview = df.tail(rows)
            
            return f"Preview of '{name}' ({position} {rows} rows):\n{tabulate(preview, headers='keys', tablefmt='grid', showindex=True)}"
        except Exception as e:
            return f"Error previewing data: {str(e)}"
    
    def get_data_info(self, params):
        """Get comprehensive information about loaded data"""
        try:
            name = params.get("name", self.current_df_name)
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            
            info = {
                "Dataset": name,
                "Shape": f"{df.shape[0]:,} rows × {df.shape[1]} columns",
                "Memory Usage": f"{df.memory_usage().sum() / 1024**2:.2f} MB",
                "Columns": len(df.columns),
                "Numeric Columns": len(df.select_dtypes(include=[np.number]).columns),
                "Object Columns": len(df.select_dtypes(include=['object']).columns),
                "DateTime Columns": len(df.select_dtypes(include=['datetime64']).columns),
                "Missing Values": df.isnull().sum().sum(),
                "Duplicated Rows": df.duplicated().sum()
            }
            
            # Column information
            col_info = []
            for col in df.columns:
                col_info.append({
                    "Column": col,
                    "Type": str(df[col].dtype),
                    "Non-Null": f"{df[col].notna().sum():,}",
                    "Null": f"{df[col].isna().sum():,}",
                    "Unique": f"{df[col].nunique():,}",
                    "Sample": str(df[col].iloc[0]) if len(df) > 0 else "N/A"
                })
            
            result = tabulate(info.items(), headers=["Property", "Value"], tablefmt="grid")
            result += "\n\nColumn Information:\n"
            result += tabulate(col_info, headers="keys", tablefmt="grid")
            
            return result
        except Exception as e:
            return f"Error getting data info: {str(e)}"
    
    # Data Analysis Tools
    def basic_statistics(self, params):
        """Calculate basic statistics for numeric columns"""
        try:
            name = params.get("name", self.current_df_name)
            columns = params.get("columns", None)
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            
            if columns:
                numeric_df = df[columns].select_dtypes(include=[np.number])
            else:
                numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                return "No numeric columns found in the dataset"
            
            stats = numeric_df.describe().T
            stats['missing'] = numeric_df.isnull().sum()
            stats['zeros'] = (numeric_df == 0).sum()
            
            return f"Basic statistics for '{name}':\n{tabulate(stats, headers=stats.columns, tablefmt='grid', floatfmt='.2f')}"
        except Exception as e:
            return f"Error calculating statistics: {str(e)}"
    
    def data_profile(self, params):
        """Generate comprehensive data profile"""
        try:
            name = params.get("name", self.current_df_name)
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            profile = []
            
            for col in df.columns:
                col_data = df[col]
                profile_info = {
                    "Column": col,
                    "Type": str(col_data.dtype),
                    "Count": len(col_data),
                    "Unique": col_data.nunique(),
                    "Missing": col_data.isnull().sum(),
                    "Missing %": f"{(col_data.isnull().sum() / len(col_data) * 100):.1f}%"
                }
                
                if pd.api.types.is_numeric_dtype(col_data):
                    profile_info.update({
                        "Min": col_data.min(),
                        "Max": col_data.max(),
                        "Mean": col_data.mean(),
                        "Std": col_data.std()
                    })
                else:
                    mode = col_data.mode()
                    profile_info.update({
                        "Top Value": mode[0] if len(mode) > 0 else "N/A",
                        "Top Freq": col_data.value_counts().iloc[0] if len(col_data.value_counts()) > 0 else 0
                    })
                
                profile.append(profile_info)
            
            return f"Data profile for '{name}':\n{tabulate(profile, headers='keys', tablefmt='grid', floatfmt='.2f')}"
        except Exception as e:
            return f"Error creating profile: {str(e)}"
    
    def correlation_analysis(self, params):
        """Analyze correlations between numeric columns"""
        try:
            name = params.get("name", self.current_df_name)
            method = params.get("method", "pearson")
            threshold = params.get("threshold", 0.5)
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.shape[1] < 2:
                return "Need at least 2 numeric columns for correlation analysis"
            
            corr_matrix = numeric_df.corr(method=method)
            
            # Find significant correlations
            significant_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) >= threshold:
                        significant_corr.append({
                            "Column 1": corr_matrix.columns[i],
                            "Column 2": corr_matrix.columns[j],
                            "Correlation": round(corr_value, 3),
                            "Strength": "Strong" if abs(corr_value) >= 0.7 else "Moderate"
                        })
            
            result = f"Correlation analysis for '{name}' (method={method}):\n"
            if significant_corr:
                result += f"\nSignificant correlations (|r| >= {threshold}):\n"
                result += tabulate(significant_corr, headers="keys", tablefmt="grid")
            else:
                result += f"\nNo correlations found with |r| >= {threshold}"
            
            return result
        except Exception as e:
            return f"Error in correlation analysis: {str(e)}"
    
    def group_by_analysis(self, params):
        """Perform group by analysis"""
        try:
            name = params.get("name", self.current_df_name)
            group_cols = params.get("group_by")
            agg_cols = params.get("aggregate_columns")
            functions = params.get("functions", ["mean", "sum", "count"])
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            
            if not group_cols:
                return "Please specify columns to group by"
            
            # Create aggregation dictionary
            if agg_cols:
                agg_dict = {col: functions for col in agg_cols}
            else:
                # Use all numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                agg_dict = {col: functions for col in numeric_cols}
            
            grouped = df.groupby(group_cols).agg(agg_dict).round(2)
            
            # Flatten column names
            grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
            grouped = grouped.reset_index()
            
            return f"Group by analysis:\n{tabulate(grouped.head(50), headers='keys', tablefmt='grid', showindex=False)}"
        except Exception as e:
            return f"Error in group by analysis: {str(e)}"
    
    def pivot_table(self, params):
        """Create pivot table"""
        try:
            name = params.get("name", self.current_df_name)
            index = params.get("index")
            columns = params.get("columns")
            values = params.get("values")
            aggfunc = params.get("aggfunc", "mean")
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            
            if not all([index, columns, values]):
                return "Please specify index, columns, and values for pivot table"
            
            pivot = pd.pivot_table(df, values=values, index=index, columns=columns, aggfunc=aggfunc)
            
            return f"Pivot table:\n{tabulate(pivot.round(2), headers='keys', tablefmt='grid', showindex=True)}"
        except Exception as e:
            return f"Error creating pivot table: {str(e)}"
    
    # Data Manipulation Tools
    def filter_data(self, params):
        """Filter data based on conditions"""
        try:
            name = params.get("name", self.current_df_name)
            conditions = params.get("conditions")
            save_as = params.get("save_as")
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name].copy()
            
            if not conditions:
                return "Please provide filter conditions"
            
            # Apply conditions
            for condition in conditions:
                column = condition.get("column")
                operator = condition.get("operator")
                value = condition.get("value")
                
                if operator == "==":
                    df = df[df[column] == value]
                elif operator == "!=":
                    df = df[df[column] != value]
                elif operator == ">":
                    df = df[df[column] > value]
                elif operator == ">=":
                    df = df[df[column] >= value]
                elif operator == "<":
                    df = df[df[column] < value]
                elif operator == "<=":
                    df = df[df[column] <= value]
                elif operator == "contains":
                    df = df[df[column].astype(str).str.contains(value, case=False)]
                elif operator == "in":
                    df = df[df[column].isin(value)]
                elif operator == "between":
                    df = df[(df[column] >= value[0]) & (df[column] <= value[1])]
            
            if save_as:
                self.dataframes[save_as] = df
                self.duckdb_conn.register(save_as, df)
                return f"Filtered data saved as '{save_as}'. Shape: {df.shape}"
            else:
                return f"Filtered data preview (shape: {df.shape}):\n{tabulate(df.head(20), headers='keys', tablefmt='grid', showindex=True)}"
        except Exception as e:
            return f"Error filtering data: {str(e)}"
    
    def sort_data(self, params):
        """Sort data by columns"""
        try:
            name = params.get("name", self.current_df_name)
            columns = params.get("columns")
            ascending = params.get("ascending", True)
            save_as = params.get("save_as")
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name].copy()
            
            if not columns:
                return "Please specify columns to sort by"
            
            df = df.sort_values(by=columns, ascending=ascending)
            
            if save_as:
                self.dataframes[save_as] = df
                self.duckdb_conn.register(save_as, df)
                return f"Sorted data saved as '{save_as}'"
            else:
                return f"Sorted data preview:\n{tabulate(df.head(20), headers='keys', tablefmt='grid', showindex=True)}"
        except Exception as e:
            return f"Error sorting data: {str(e)}"
    
    def merge_data(self, params):
        """Merge two datasets"""
        try:
            left = params.get("left_dataset")
            right = params.get("right_dataset")
            on = params.get("on")
            how = params.get("how", "inner")
            save_as = params.get("save_as", f"{left}_{right}_merged")
            
            if left not in self.dataframes or right not in self.dataframes:
                return f"Datasets not found. Available: {list(self.dataframes.keys())}"
            
            left_df = self.dataframes[left]
            right_df = self.dataframes[right]
            
            merged = pd.merge(left_df, right_df, on=on, how=how)
            
            self.dataframes[save_as] = merged
            self.duckdb_conn.register(save_as, merged)
            
            return f"Merged data saved as '{save_as}':\n- Shape: {merged.shape}\n- Method: {how} join on {on}"
        except Exception as e:
            return f"Error merging data: {str(e)}"
    
    def aggregate_data(self, params):
        """Aggregate data with custom functions"""
        try:
            name = params.get("name", self.current_df_name)
            aggregations = params.get("aggregations")
            save_as = params.get("save_as")
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            
            if not aggregations:
                return "Please specify aggregations"
            
            result = df.agg(aggregations)
            
            if save_as:
                self.dataframes[save_as] = pd.DataFrame(result)
                return f"Aggregated data saved as '{save_as}'"
            else:
                return f"Aggregation results:\n{tabulate(pd.DataFrame(result), headers=['Metric', 'Value'], tablefmt='grid')}"
        except Exception as e:
            return f"Error aggregating data: {str(e)}"
    
    def transform_column(self, params):
        """Transform column with custom function"""
        try:
            name = params.get("name", self.current_df_name)
            column = params.get("column")
            operation = params.get("operation")
            new_column = params.get("new_column")
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            
            if column not in df.columns:
                return f"Column '{column}' not found in dataset"
            
            # Apply transformation
            if operation == "log":
                df[new_column or f"{column}_log"] = np.log(df[column] + 1)
            elif operation == "sqrt":
                df[new_column or f"{column}_sqrt"] = np.sqrt(df[column])
            elif operation == "square":
                df[new_column or f"{column}_squared"] = df[column] ** 2
            elif operation == "normalize":
                df[new_column or f"{column}_normalized"] = (df[column] - df[column].mean()) / df[column].std()
            elif operation == "scale":
                scaler = StandardScaler()
                df[new_column or f"{column}_scaled"] = scaler.fit_transform(df[[column]])
            elif operation == "rank":
                df[new_column or f"{column}_rank"] = df[column].rank()
            elif operation == "pct_change":
                df[new_column or f"{column}_pct_change"] = df[column].pct_change()
            elif operation == "cumsum":
                df[new_column or f"{column}_cumsum"] = df[column].cumsum()
            elif operation == "diff":
                df[new_column or f"{column}_diff"] = df[column].diff()
            else:
                return f"Unknown operation: {operation}"
            
            return f"Column transformed successfully. New column: {new_column or f'{column}_{operation}'}"
        except Exception as e:
            return f"Error transforming column: {str(e)}"
    
    def add_column(self, params):
        """Add new column with formula or constant value"""
        try:
            name = params.get("name", self.current_df_name)
            column_name = params.get("column_name")
            formula = params.get("formula")
            value = params.get("value")
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            
            if formula:
                # Use eval for formula (be careful in production!)
                df[column_name] = df.eval(formula)
            elif value is not None:
                df[column_name] = value
            else:
                return "Please provide either a formula or a value"
            
            return f"Column '{column_name}' added successfully"
        except Exception as e:
            return f"Error adding column: {str(e)}"
    
    def remove_column(self, params):
        """Remove columns from dataset"""
        try:
            name = params.get("name", self.current_df_name)
            columns = params.get("columns")
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            
            if not columns:
                return "Please specify columns to remove"
            
            df.drop(columns=columns, inplace=True)
            
            return f"Removed columns: {columns}"
        except Exception as e:
            return f"Error removing columns: {str(e)}"
    
    def rename_columns(self, params):
        """Rename columns in dataset"""
        try:
            name = params.get("name", self.current_df_name)
            mappings = params.get("mappings")
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            
            if not mappings:
                return "Please provide column name mappings"
            
            df.rename(columns=mappings, inplace=True)
            
            return f"Renamed columns: {list(mappings.keys())} → {list(mappings.values())}"
        except Exception as e:
            return f"Error renaming columns: {str(e)}"
    
    # Data Cleaning Tools
    def handle_missing(self, params):
        """Handle missing values in dataset"""
        try:
            name = params.get("name", self.current_df_name)
            strategy = params.get("strategy", "drop")
            columns = params.get("columns")
            value = params.get("value")
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            before_missing = df.isnull().sum().sum()
            
            if columns:
                cols = columns
            else:
                cols = df.columns
            
            if strategy == "drop":
                df.dropna(subset=cols, inplace=True)
            elif strategy == "fill_value":
                df[cols] = df[cols].fillna(value)
            elif strategy == "fill_mean":
                for col in cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "fill_median":
                for col in cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "fill_mode":
                for col in cols:
                    mode = df[col].mode()
                    if len(mode) > 0:
                        df[col].fillna(mode[0], inplace=True)
            elif strategy == "forward_fill":
                df[cols] = df[cols].fillna(method='ffill')
            elif strategy == "backward_fill":
                df[cols] = df[cols].fillna(method='bfill')
            elif strategy == "interpolate":
                for col in cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].interpolate()
            
            after_missing = df.isnull().sum().sum()
            
            return f"Missing values handled:\n- Before: {before_missing}\n- After: {after_missing}\n- Strategy: {strategy}"
        except Exception as e:
            return f"Error handling missing values: {str(e)}"
    
    def remove_duplicates(self, params):
        """Remove duplicate rows"""
        try:
            name = params.get("name", self.current_df_name)
            subset = params.get("subset")
            keep = params.get("keep", "first")
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            before_shape = df.shape
            
            df.drop_duplicates(subset=subset, keep=keep, inplace=True)
            
            after_shape = df.shape
            removed = before_shape[0] - after_shape[0]
            
            return f"Duplicates removed:\n- Rows removed: {removed}\n- New shape: {after_shape}"
        except Exception as e:
            return f"Error removing duplicates: {str(e)}"
    
    def convert_types(self, params):
        """Convert column data types"""
        try:
            name = params.get("name", self.current_df_name)
            conversions = params.get("conversions")
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            
            if not conversions:
                return "Please provide type conversions"
            
            results = []
            for column, new_type in conversions.items():
                old_type = str(df[column].dtype)
                try:
                    if new_type == "int":
                        df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                    elif new_type == "float":
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    elif new_type == "string":
                        df[column] = df[column].astype(str)
                    elif new_type == "datetime":
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    elif new_type == "category":
                        df[column] = df[column].astype('category')
                    elif new_type == "bool":
                        df[column] = df[column].astype(bool)
                    
                    results.append(f"{column}: {old_type} → {new_type}")
                except Exception as e:
                    results.append(f"{column}: Failed - {str(e)}")
            
            return "Type conversions:\n" + "\n".join(results)
        except Exception as e:
            return f"Error converting types: {str(e)}"
    
    def detect_outliers(self, params):
        """Detect outliers in numeric columns"""
        try:
            name = params.get("name", self.current_df_name)
            method = params.get("method", "iqr")
            columns = params.get("columns")
            threshold = params.get("threshold", 1.5)
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            
            if columns:
                numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            outlier_info = []
            
            for col in numeric_cols:
                if method == "iqr":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                elif method == "zscore":
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outliers = df[col][z_scores > threshold]
                
                outlier_info.append({
                    "Column": col,
                    "Method": method,
                    "Outliers": len(outliers),
                    "Percentage": f"{(len(outliers) / len(df) * 100):.2f}%",
                    "Min": df[col].min(),
                    "Max": df[col].max()
                })
            
            return f"Outlier detection results:\n{tabulate(outlier_info, headers='keys', tablefmt='grid')}"
        except Exception as e:
            return f"Error detecting outliers: {str(e)}"
    
    # Data Export Tools
    def save_data(self, params):
        """Save dataset to file"""
        try:
            name = params.get("name", self.current_df_name)
            filepath = params.get("filepath")
            format = params.get("format", "csv")
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            
            if not filepath:
                filepath = f"{name}.{format}"
            
            if format == "csv":
                df.to_csv(filepath, index=False)
            elif format == "excel":
                df.to_excel(filepath, index=False)
            elif format == "json":
                df.to_json(filepath, orient='records')
            elif format == "parquet":
                df.to_parquet(filepath, index=False)
            else:
                return f"Unsupported format: {format}"
            
            return f"Data saved to {filepath} (format: {format}, shape: {df.shape})"
        except Exception as e:
            return f"Error saving data: {str(e)}"
    
    def export_filtered(self, params):
        """Export filtered subset of data"""
        try:
            name = params.get("name", self.current_df_name)
            conditions = params.get("conditions")
            filepath = params.get("filepath")
            format = params.get("format", "csv")
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name].copy()
            
            # Apply filters if provided
            if conditions:
                for condition in conditions:
                    column = condition.get("column")
                    operator = condition.get("operator")
                    value = condition.get("value")
                    
                    if operator == "==":
                        df = df[df[column] == value]
                    elif operator == ">":
                        df = df[df[column] > value]
                    elif operator == "<":
                        df = df[df[column] < value]
                    elif operator == "contains":
                        df = df[df[column].str.contains(value, case=False)]
            
            if not filepath:
                filepath = f"{name}_filtered.{format}"
            
            if format == "csv":
                df.to_csv(filepath, index=False)
            elif format == "excel":
                df.to_excel(filepath, index=False)
            
            return f"Filtered data exported to {filepath} (rows: {len(df)})"
        except Exception as e:
            return f"Error exporting filtered data: {str(e)}"
    
    # Advanced Features
    def sql_query(self, params):
        """Execute SQL query on loaded data"""
        try:
            query = params.get("query")
            save_as = params.get("save_as")
            
            if not query:
                return "Please provide a SQL query"
            
            # Execute query
            result = self.duckdb_conn.execute(query).df()
            
            if save_as:
                self.dataframes[save_as] = result
                self.duckdb_conn.register(save_as, result)
                return f"Query result saved as '{save_as}' (shape: {result.shape})"
            else:
                return f"Query result:\n{tabulate(result.head(50), headers='keys', tablefmt='grid', showindex=False)}"
        except Exception as e:
            return f"Error executing SQL query: {str(e)}"
    
    def create_visualization(self, params):
        """Create data visualization"""
        try:
            name = params.get("name", self.current_df_name)
            chart_type = params.get("chart_type")
            x = params.get("x")
            y = params.get("y")
            title = params.get("title", "Data Visualization")
            save_path = params.get("save_path")
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            
            # Create appropriate visualization
            plt.figure(figsize=(10, 6))
            
            if chart_type == "scatter":
                plt.scatter(df[x], df[y], alpha=0.6)
                plt.xlabel(x)
                plt.ylabel(y)
            elif chart_type == "line":
                plt.plot(df[x], df[y])
                plt.xlabel(x)
                plt.ylabel(y)
            elif chart_type == "bar":
                if y:
                    df.groupby(x)[y].mean().plot(kind='bar')
                else:
                    df[x].value_counts().plot(kind='bar')
            elif chart_type == "histogram":
                plt.hist(df[x], bins=30, alpha=0.7)
                plt.xlabel(x)
                plt.ylabel("Frequency")
            elif chart_type == "heatmap":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            elif chart_type == "box":
                if y:
                    df.boxplot(column=y, by=x)
                else:
                    df[x].plot(kind='box')
            else:
                return f"Unsupported chart type: {chart_type}"
            
            plt.title(title)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return f"Visualization saved to {save_path}"
            else:
                plt.show()
                return "Visualization displayed"
        except Exception as e:
            return f"Error creating visualization: {str(e)}"
    
    def run_clustering(self, params):
        """Run clustering analysis"""
        try:
            name = params.get("name", self.current_df_name)
            features = params.get("features")
            n_clusters = params.get("n_clusters", 3)
            save_column = params.get("save_column", "cluster")
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            
            if not features:
                features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Prepare data
            X = df[features].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Run clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to original dataframe
            df.loc[X.index, save_column] = clusters
            
            # Calculate cluster statistics
            cluster_stats = []
            for i in range(n_clusters):
                cluster_data = df[df[save_column] == i]
                cluster_stats.append({
                    "Cluster": i,
                    "Size": len(cluster_data),
                    "Percentage": f"{(len(cluster_data) / len(df) * 100):.1f}%"
                })
            
            return f"Clustering complete:\n{tabulate(cluster_stats, headers='keys', tablefmt='grid')}\nCluster labels saved to column '{save_column}'"
        except Exception as e:
            return f"Error running clustering: {str(e)}"
    
    def run_regression(self, params):
        """Run linear regression analysis"""
        try:
            name = params.get("name", self.current_df_name)
            target = params.get("target")
            features = params.get("features")
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name]
            
            if not target or not features:
                return "Please specify target and feature columns"
            
            # Prepare data
            X = df[features].dropna()
            y = df.loc[X.index, target]
            
            # Remove any remaining NaN values
            mask = ~y.isna()
            X = X[mask]
            y = y[mask]
            
            # Fit model
            model = LinearRegression()
            model.fit(X, y)
            
            # Get predictions and metrics
            y_pred = model.predict(X)
            r2 = model.score(X, y)
            
            # Create results
            results = {
                "R² Score": f"{r2:.4f}",
                "Intercept": f"{model.intercept_:.4f}",
            }
            
            # Add coefficients
            coef_info = []
            for feature, coef in zip(features, model.coef_):
                coef_info.append({
                    "Feature": feature,
                    "Coefficient": f"{coef:.4f}",
                    "Impact": "Positive" if coef > 0 else "Negative"
                })
            
            result_str = f"Regression Analysis Results:\n"
            result_str += tabulate(results.items(), headers=["Metric", "Value"], tablefmt="grid")
            result_str += f"\n\nFeature Coefficients:\n"
            result_str += tabulate(coef_info, headers="keys", tablefmt="grid")
            
            return result_str
        except Exception as e:
            return f"Error running regression: {str(e)}"
    
    def time_series_analysis(self, params):
        """Perform time series analysis"""
        try:
            name = params.get("name", self.current_df_name)
            date_column = params.get("date_column")
            value_column = params.get("value_column")
            frequency = params.get("frequency", "D")
            
            if name not in self.dataframes:
                return f"Dataset '{name}' not found. Available: {list(self.dataframes.keys())}"
            
            df = self.dataframes[name].copy()
            
            if not date_column or not value_column:
                return "Please specify date and value columns"
            
            # Convert to datetime and set as index
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column).sort_index()
            
            # Basic time series statistics
            ts = df[value_column]
            
            stats = {
                "Start Date": str(ts.index.min()),
                "End Date": str(ts.index.max()),
                "Duration": str(ts.index.max() - ts.index.min()),
                "Total Points": len(ts),
                "Mean": f"{ts.mean():.2f}",
                "Std Dev": f"{ts.std():.2f}",
                "Min": f"{ts.min():.2f}",
                "Max": f"{ts.max():.2f}",
                "Trend": "Increasing" if ts.iloc[-1] > ts.iloc[0] else "Decreasing"
            }
            
            # Calculate rolling statistics
            window = min(30, len(ts) // 4)
            df[f"{value_column}_rolling_mean"] = ts.rolling(window=window).mean()
            df[f"{value_column}_rolling_std"] = ts.rolling(window=window).std()
            
            return f"Time Series Analysis:\n{tabulate(stats.items(), headers=['Metric', 'Value'], tablefmt='grid')}\n\nRolling statistics calculated with window={window}"
        except Exception as e:
            return f"Error in time series analysis: {str(e)}"
    
    # SMS Tool (Original)
    def send_sms(self, model_input):
        """Send SMS via Twilio"""
        if not self.twilio_client:
            return "Twilio not configured. Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER"
        
        to_number = model_input.get("to_number")
        message_body = model_input.get("message")
        
        if not to_number or not message_body:
            return "Missing required parameters: to_number and message"
        
        try:
            message = self.twilio_client.messages.create(
                body=message_body,
                from_=self.twilio_phone,
                to=to_number
            )
            return f"SMS sent successfully! Message SID: {message.sid}"
        except Exception as e:
            return f"Failed to send SMS: {str(e)}"
    
    def set_tools(self):
        """Set up available tools for the agent"""
        tools = [
            # Data Discovery Tools
            {
                "name": "list_data_files",
                "description": "List all available data files in the workspace",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File patterns to search for (e.g., ['*.csv', '*.xlsx'])",
                            "default": ["*.csv", "*.xlsx", "*.json", "*.parquet"]
                        },
                        "directory": {
                            "type": "string",
                            "description": "Directory to search in",
                            "default": "."
                        }
                    }
                }
            },
            {
                "name": "search_in_files",
                "description": "Search for specific patterns in data files",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Pattern to search for"
                        },
                        "file_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File types to search in",
                            "default": ["*.csv", "*.txt"]
                        }
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "get_file_info",
                "description": "Get detailed information about a specific file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the file"
                        }
                    },
                    "required": ["filepath"]
                }
            },
            
            # Data Loading Tools
            {
                "name": "load_data",
                "description": "Load data from various file formats (CSV, Excel, JSON, Parquet)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the data file"
                        },
                        "name": {
                            "type": "string",
                            "description": "Name to store the dataset as"
                        },
                        "sheet": {
                            "type": ["string", "integer"],
                            "description": "Sheet name or index for Excel files"
                        }
                    },
                    "required": ["filepath"]
                }
            },
            {
                "name": "preview_data",
                "description": "Preview first or last N rows of loaded data",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "rows": {
                            "type": "integer",
                            "description": "Number of rows to preview",
                            "default": 10
                        },
                        "position": {
                            "type": "string",
                            "enum": ["head", "tail"],
                            "description": "Preview from head or tail",
                            "default": "head"
                        }
                    }
                }
            },
            {
                "name": "get_data_info",
                "description": "Get comprehensive information about loaded data",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        }
                    }
                }
            },
            
            # Data Analysis Tools
            {
                "name": "basic_statistics",
                "description": "Calculate basic statistics for numeric columns",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific columns to analyze"
                        }
                    }
                }
            },
            {
                "name": "data_profile",
                "description": "Generate comprehensive data profile with column statistics",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        }
                    }
                }
            },
            {
                "name": "correlation_analysis",
                "description": "Analyze correlations between numeric columns",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["pearson", "spearman", "kendall"],
                            "description": "Correlation method",
                            "default": "pearson"
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Minimum correlation threshold to report",
                            "default": 0.5
                        }
                    }
                }
            },
            {
                "name": "group_by_analysis",
                "description": "Perform group by analysis with aggregations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "group_by": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Columns to group by"
                        },
                        "aggregate_columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Columns to aggregate"
                        },
                        "functions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Aggregation functions",
                            "default": ["mean", "sum", "count"]
                        }
                    },
                    "required": ["group_by"]
                }
            },
            {
                "name": "pivot_table",
                "description": "Create pivot table analysis",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "index": {
                            "type": "string",
                            "description": "Column for pivot table index"
                        },
                        "columns": {
                            "type": "string",
                            "description": "Column for pivot table columns"
                        },
                        "values": {
                            "type": "string",
                            "description": "Column for values to aggregate"
                        },
                        "aggfunc": {
                            "type": "string",
                            "description": "Aggregation function",
                            "default": "mean"
                        }
                    },
                    "required": ["index", "columns", "values"]
                }
            },
            
            # Data Manipulation Tools
            {
                "name": "filter_data",
                "description": "Filter data based on conditions",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "conditions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column": {"type": "string"},
                                    "operator": {
                                        "type": "string",
                                        "enum": ["==", "!=", ">", ">=", "<", "<=", "contains", "in", "between"]
                                    },
                                    "value": {}
                                }
                            },
                            "description": "Filter conditions"
                        },
                        "save_as": {
                            "type": "string",
                            "description": "Name to save filtered data as"
                        }
                    },
                    "required": ["conditions"]
                }
            },
            {
                "name": "sort_data",
                "description": "Sort data by columns",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Columns to sort by"
                        },
                        "ascending": {
                            "type": "boolean",
                            "description": "Sort in ascending order",
                            "default": True
                        },
                        "save_as": {
                            "type": "string",
                            "description": "Name to save sorted data as"
                        }
                    },
                    "required": ["columns"]
                }
            },
            {
                "name": "merge_data",
                "description": "Merge two datasets",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "left_dataset": {
                            "type": "string",
                            "description": "Left dataset name"
                        },
                        "right_dataset": {
                            "type": "string",
                            "description": "Right dataset name"
                        },
                        "on": {
                            "type": ["string", "array"],
                            "description": "Column(s) to merge on"
                        },
                        "how": {
                            "type": "string",
                            "enum": ["inner", "left", "right", "outer"],
                            "description": "Type of merge",
                            "default": "inner"
                        },
                        "save_as": {
                            "type": "string",
                            "description": "Name to save merged data as"
                        }
                    },
                    "required": ["left_dataset", "right_dataset", "on"]
                }
            },
            {
                "name": "aggregate_data",
                "description": "Aggregate data with custom functions",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "aggregations": {
                            "type": "object",
                            "description": "Column to function mapping"
                        },
                        "save_as": {
                            "type": "string",
                            "description": "Name to save aggregated data as"
                        }
                    },
                    "required": ["aggregations"]
                }
            },
            {
                "name": "transform_column",
                "description": "Transform column with operations like log, sqrt, normalize, scale, etc.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "column": {
                            "type": "string",
                            "description": "Column to transform"
                        },
                        "operation": {
                            "type": "string",
                            "enum": ["log", "sqrt", "square", "normalize", "scale", "rank", "pct_change", "cumsum", "diff"],
                            "description": "Transformation operation"
                        },
                        "new_column": {
                            "type": "string",
                            "description": "Name for new transformed column"
                        }
                    },
                    "required": ["column", "operation"]
                }
            },
            {
                "name": "add_column",
                "description": "Add new column with formula or constant value",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "column_name": {
                            "type": "string",
                            "description": "New column name"
                        },
                        "formula": {
                            "type": "string",
                            "description": "Formula using existing columns (e.g., 'col1 + col2')"
                        },
                        "value": {
                            "description": "Constant value for new column"
                        }
                    },
                    "required": ["column_name"]
                }
            },
            {
                "name": "remove_column",
                "description": "Remove columns from dataset",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Columns to remove"
                        }
                    },
                    "required": ["columns"]
                }
            },
            {
                "name": "rename_columns",
                "description": "Rename columns in dataset",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "mappings": {
                            "type": "object",
                            "description": "Old name to new name mapping"
                        }
                    },
                    "required": ["mappings"]
                }
            },
            
            # Data Cleaning Tools
            {
                "name": "handle_missing",
                "description": "Handle missing values with various strategies",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "strategy": {
                            "type": "string",
                            "enum": ["drop", "fill_value", "fill_mean", "fill_median", "fill_mode", "forward_fill", "backward_fill", "interpolate"],
                            "description": "Strategy for handling missing values",
                            "default": "drop"
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific columns to handle"
                        },
                        "value": {
                            "description": "Value to fill with (for fill_value strategy)"
                        }
                    }
                }
            },
            {
                "name": "remove_duplicates",
                "description": "Remove duplicate rows from dataset",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "subset": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Columns to consider for duplicates"
                        },
                        "keep": {
                            "type": "string",
                            "enum": ["first", "last"],
                            "description": "Which duplicate to keep",
                            "default": "first"
                        }
                    }
                }
            },
            {
                "name": "convert_types",
                "description": "Convert column data types",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "conversions": {
                            "type": "object",
                            "description": "Column to new type mapping (types: int, float, string, datetime, category, bool)"
                        }
                    },
                    "required": ["conversions"]
                }
            },
            {
                "name": "detect_outliers",
                "description": "Detect outliers in numeric columns",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["iqr", "zscore"],
                            "description": "Outlier detection method",
                            "default": "iqr"
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific columns to check"
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Threshold for outlier detection",
                            "default": 1.5
                        }
                    }
                }
            },
            
            # Data Export Tools
            {
                "name": "save_data",
                "description": "Save dataset to file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "filepath": {
                            "type": "string",
                            "description": "Path to save file"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["csv", "excel", "json", "parquet"],
                            "description": "File format",
                            "default": "csv"
                        }
                    },
                    "required": ["filepath"]
                }
            },
            {
                "name": "export_filtered",
                "description": "Export filtered subset of data",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "conditions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column": {"type": "string"},
                                    "operator": {"type": "string"},
                                    "value": {}
                                }
                            },
                            "description": "Filter conditions"
                        },
                        "filepath": {
                            "type": "string",
                            "description": "Path to save file"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["csv", "excel"],
                            "description": "File format",
                            "default": "csv"
                        }
                    },
                    "required": ["filepath"]
                }
            },
            
            # Advanced Features
            {
                "name": "sql_query",
                "description": "Execute SQL query on loaded datasets",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute"
                        },
                        "save_as": {
                            "type": "string",
                            "description": "Name to save query result as"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "create_visualization",
                "description": "Create data visualization (scatter, line, bar, histogram, heatmap, box plot)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "chart_type": {
                            "type": "string",
                            "enum": ["scatter", "line", "bar", "histogram", "heatmap", "box"],
                            "description": "Type of chart"
                        },
                        "x": {
                            "type": "string",
                            "description": "X-axis column"
                        },
                        "y": {
                            "type": "string",
                            "description": "Y-axis column"
                        },
                        "title": {
                            "type": "string",
                            "description": "Chart title",
                            "default": "Data Visualization"
                        },
                        "save_path": {
                            "type": "string",
                            "description": "Path to save visualization"
                        }
                    },
                    "required": ["chart_type"]
                }
            },
            {
                "name": "run_clustering",
                "description": "Run K-means clustering analysis",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "features": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Feature columns for clustering"
                        },
                        "n_clusters": {
                            "type": "integer",
                            "description": "Number of clusters",
                            "default": 3
                        },
                        "save_column": {
                            "type": "string",
                            "description": "Column name to save cluster labels",
                            "default": "cluster"
                        }
                    }
                }
            },
            {
                "name": "run_regression",
                "description": "Run linear regression analysis",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "target": {
                            "type": "string",
                            "description": "Target column"
                        },
                        "features": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Feature columns"
                        }
                    },
                    "required": ["target", "features"]
                }
            },
            {
                "name": "time_series_analysis",
                "description": "Perform time series analysis",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Dataset name"
                        },
                        "date_column": {
                            "type": "string",
                            "description": "Date/time column"
                        },
                        "value_column": {
                            "type": "string",
                            "description": "Value column to analyze"
                        },
                        "frequency": {
                            "type": "string",
                            "description": "Time series frequency (D, W, M, etc.)",
                            "default": "D"
                        }
                    },
                    "required": ["date_column", "value_column"]
                }
            }
        ]
        
        # Add SMS tool if Twilio is configured
        if self.twilio_client:
            tools.append({
                "name": "send_sms",
                "description": "Send SMS text message via Twilio",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "to_number": {
                            "type": "string",
                            "description": "Recipient phone number in E.164 format (e.g., +1234567890)"
                        },
                        "message": {
                            "type": "string",
                            "description": "Text message content to send"
                        }
                    },
                    "required": ["to_number", "message"]
                }
            })
        
        self.tools = tools

    def interactive_chat(self, system_prompt=None):
        """Start an interactive multi-turn chat session"""
        print("Data Agent Pro - Advanced Data Analysis Assistant")
        print("=" * 60)
        print("Commands:")
        print("  • 'quit' or 'exit' to end")
        print("  • 'clear' to reset conversation")
        print("  • 'datasets' to list loaded datasets")
        print("\nCapabilities:")
        print("  📁 File Discovery & Loading")
        print("  📊 Data Analysis & Statistics")
        print("  🔧 Data Manipulation & Transformation")
        print("  🧹 Data Cleaning & Quality")
        print("  📈 Visualization & Machine Learning")
        print("  💾 Export & SQL Queries")
        if self.twilio_client:
            print("  📱 SMS Notifications")
        print("=" * 60)
        
        default_prompt = """You are a powerful data analysis assistant with access to comprehensive data manipulation tools.

MULTI-TOOL WORKFLOW EXAMPLES:
• "Load data.csv" → load_data + preview_data + get_data_info + basic_statistics
• "Analyze this data" → data_profile + correlation_analysis + detect_outliers + create_visualization
• "Clean the data" → handle_missing + remove_duplicates + detect_outliers + save_data
• "Prepare for analysis" → All cleaning tools + transform_column + save_data

BE PROACTIVE: Don't wait for users to ask for each step. Anticipate their needs and execute complete workflows.

When users ask about data, execute a full exploration workflow automatically."""
        
        if system_prompt:
            print(f"System: {system_prompt}")
            print("-" * 60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    print("Conversation history cleared!")
                    continue
                elif user_input.lower() == 'datasets':
                    if self.dataframes:
                        print("Loaded datasets:")
                        for name, df in self.dataframes.items():
                            print(f"  • {name}: {df.shape[0]} rows × {df.shape[1]} columns")
                    else:
                        print("No datasets loaded yet.")
                    continue
                elif not user_input:
                    continue
                
                print("Agent: ", end="", flush=True)
                response = self.send_message(user_input, system_prompt or default_prompt)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\nChat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")

# Example usage
if __name__ == "__main__":
    try:
        agent = DataAgent()
        
        # Start interactive chat
        agent.interactive_chat()
        
    except ValueError as e:
        print(f"{e}")
        print("\nTo fix this:")
        print("1. Set environment variable: export ANTHROPIC_API_KEY='your-api-key'")
        print("2. Or pass API key directly: DataAgent(api_key='your-api-key')")
        print("3. Create a .env file with: ANTHROPIC_API_KEY=your-api-key")