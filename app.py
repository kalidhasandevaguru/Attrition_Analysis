from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
from io import StringIO
import logging
from datetime import datetime, timedelta
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for React frontend
CORS(app, origins=["*"])  # In production, specify your React app's domain

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

DEFAULT_CONFIG = {
    'age_groups': {
        'bins': [18, 25, 35, 45, 55, 65, 100],
        'labels': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    },
    'date_formats': ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d'],
    'max_file_size_mb': 50,
    'age_range': {'min': 16, 'max': 100}
}

def sanitize_csv_data(csv_data: str) -> str:
    """Basic sanitization for CSV injection attacks"""
    if not csv_data or not isinstance(csv_data, str):
        raise ValueError("Invalid CSV data provided")
    
    # Check file size (approximate)
    size_mb = len(csv_data.encode('utf-8')) / (1024 * 1024)
    if size_mb > DEFAULT_CONFIG['max_file_size_mb']:
        raise ValueError(f"CSV data too large: {size_mb:.1f}MB > {DEFAULT_CONFIG['max_file_size_mb']}MB")
    
    # Basic sanitization - remove potential CSV injection patterns
    dangerous_patterns = ['=', '+', '-', '@']
    lines = csv_data.strip().split('\n')
    
    sanitized_lines = []
    for line in lines:
        # Don't sanitize header row
        if line == lines[0]:
            sanitized_lines.append(line)
            continue
            
        # Sanitize data rows
        cells = line.split(',')
        sanitized_cells = []
        for cell in cells:
            cell = cell.strip().strip('"\'')
            if cell and cell[0] in dangerous_patterns:
                cell = "'" + cell  # Prefix with single quote to neutralize
            sanitized_cells.append(cell)
        sanitized_lines.append(','.join(sanitized_cells))
    
    return '\n'.join(sanitized_lines)

def validate_and_parse_dates(df, date_column, date_formats=None):
    """Validate and parse date columns with multiple format support"""
    if date_formats is None:
        date_formats = DEFAULT_CONFIG['date_formats']
    
    if date_column not in df.columns:
        return df
    
    # Try multiple date formats
    parsed_dates = None
    for date_format in date_formats:
        try:
            parsed_dates = pd.to_datetime(df[date_column], format=date_format, errors='coerce')
            if parsed_dates.notna().sum() > 0:
                break
        except:
            continue
    
    # Fallback to pandas auto-detection
    if parsed_dates is None or parsed_dates.notna().sum() == 0:
        parsed_dates = pd.to_datetime(df[date_column], errors='coerce')
    
    df[date_column] = parsed_dates
    
    # Validate date ranges
    today = pd.Timestamp.now()
    if date_column == 'Date_of_Birth':
        # Birth dates should be between 16-100 years ago
        min_birth_date = today - timedelta(days=100*365)
        max_birth_date = today - timedelta(days=16*365)
        df.loc[(df[date_column] < min_birth_date) | (df[date_column] > max_birth_date), date_column] = pd.NaT
    
    elif date_column == 'Date_of_Joining':
        # Joining dates shouldn't be in the future or too far in the past (50 years)
        min_join_date = today - timedelta(days=50*365)
        df.loc[(df[date_column] > today) | (df[date_column] < min_join_date), date_column] = pd.NaT
    
    elif date_column == 'Last_Working_Date':
        # Last working date shouldn't be in the future or before joining
        df.loc[df[date_column] > today, date_column] = pd.NaT
        # If both dates exist, last working date shouldn't be before joining date
        if 'Date_of_Joining' in df.columns:
            invalid_mask = (df[date_column].notna() & df['Date_of_Joining'].notna() & 
                          (df[date_column] < df['Date_of_Joining']))
            df.loc[invalid_mask, date_column] = pd.NaT
    
    return df

def handle_missing_categorical_data(df):
    """Handle null/missing values in categorical columns"""
    categorical_columns = ['Department', 'Gender']
    
    for col in categorical_columns:
        if col in df.columns:
            # Fill missing values with 'Unknown' and log the count
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                logger.warning(f"Found {missing_count} missing values in {col}, filling with 'Unknown'")
                df[col] = df[col].fillna('Unknown')
            
            # Clean up whitespace and standardize
            df[col] = df[col].astype(str).str.strip()
            
            # Handle empty strings
            df.loc[df[col] == '', col] = 'Unknown'
    
    return df

def analyze_attrition(csv_data: str, config=None):
    """
    Analyze employee attrition with robust error handling and validation
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    try:
        # Input sanitization
        csv_data = sanitize_csv_data(csv_data)
        
        # Parse CSV
        try:
            df = pd.read_csv(StringIO(csv_data))
        except pd.errors.EmptyDataError:
            return {"error": "Empty CSV data provided"}
        except Exception as e:
            return {"error": f"Failed to parse CSV: {str(e)}"}
        
        # Handle empty dataframe
        if df.empty:
            return {"error": "No data to analyze"}
        
        # Check for required columns
        required_columns = ['Department', 'Gender', 'Date_of_Joining']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {"error": f"Missing required columns: {missing_columns}"}
        
        logger.info(f"Processing {len(df)} employee records")
        
        # Handle missing categorical data
        df = handle_missing_categorical_data(df)
        
        # Validate and parse date columns
        date_columns = ['Date_of_Joining', 'Last_Working_Date', 'Date_of_Birth']
        for date_col in date_columns:
            df = validate_and_parse_dates(df, date_col, config['date_formats'])
        
        # Remove rows with missing critical data
        initial_count = len(df)
        df = df.dropna(subset=['Department', 'Gender', 'Date_of_Joining'])
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} rows with missing critical data")
        
        if df.empty:
            return {"error": "No valid data remaining after cleaning"}
        
        # Calculate total employees and attrition count
        total_employees = len(df)
        attrited_employees = df['Last_Working_Date'].notna().sum()
        attrition_rate = attrited_employees / total_employees if total_employees > 0 else 0
        
        # Attrition by Department
        department_attrition = []
        for dept in df['Department'].unique():
            dept_df = df[df['Department'] == dept]
            dept_total = len(dept_df)
            dept_attrited = dept_df['Last_Working_Date'].notna().sum()
            dept_rate = dept_attrited / dept_total if dept_total > 0 else 0
            department_attrition.append({
                "department": dept, 
                "attrition_rate": round(dept_rate, 4),
                "total_employees": dept_total,
                "attrited_employees": dept_attrited
            })
        
        # More robust age calculation and attrition by Age Group
        age_group_attrition = []
        invalid_ages = 0
        
        if 'Date_of_Birth' in df.columns and df['Date_of_Birth'].notna().any():
            df['Date_of_Birth'] = pd.to_datetime(df['Date_of_Birth'], errors='coerce')
            today = pd.Timestamp.now()
            df['Age'] = (today - df['Date_of_Birth']).dt.days / 365.25
            
            # Filter out invalid ages
            valid_ages_mask = df['Age'].between(config['age_range']['min'], 
                                              config['age_range']['max'], na=False)
            invalid_ages = (~valid_ages_mask & df['Age'].notna()).sum()
            
            if invalid_ages > 0:
                logger.warning(f"Found {invalid_ages} employees with invalid ages, excluding from age analysis")
            
            age_df = df[valid_ages_mask]
            
            if not age_df.empty:
                bins = config['age_groups']['bins']
                labels = config['age_groups']['labels']
                age_df['Age_Group'] = pd.cut(age_df['Age'], bins=bins, labels=labels, right=False)
                
                for age_group in age_df['Age_Group'].dropna().unique():
                    group_df = age_df[age_df['Age_Group'] == age_group]
                    group_total = len(group_df)
                    group_attrited = group_df['Last_Working_Date'].notna().sum()
                    group_rate = group_attrited / group_total if group_total > 0 else 0
                    age_group_attrition.append({
                        "age_group": str(age_group), 
                        "attrition_rate": round(group_rate, 4),
                        "total_employees": group_total,
                        "attrited_employees": group_attrited
                    })
        
        if not age_group_attrition:
            age_group_attrition.append({
                "age_group": "N/A", 
                "attrition_rate": 0,
                "total_employees": 0,
                "attrited_employees": 0,
                "note": "No valid age data available"
            })
        
        # Attrition by Gender
        gender_attrition = []
        for gender in df['Gender'].unique():
            gender_df = df[df['Gender'] == gender]
            gender_total = len(gender_df)
            gender_attrited = gender_df['Last_Working_Date'].notna().sum()
            gender_rate = gender_attrited / gender_total if gender_total > 0 else 0
            gender_attrition.append({
                "gender": gender, 
                "attrition_rate": round(gender_rate, 4),
                "total_employees": gender_total,
                "attrited_employees": gender_attrited
            })
        
        result = {
            "success": True,
            "summary": {
                "total_employees": total_employees,
                "attrited_employees": attrited_employees,
                "attrition_rate": round(attrition_rate, 4),
                "data_quality": {
                    "rows_processed": total_employees,
                    "rows_removed": removed_count,
                    "invalid_ages_excluded": invalid_ages
                }
            },
            "department_attrition": department_attrition,
            "age_group_attrition": age_group_attrition,
            "gender_attrition": gender_attrition
        }
        
        logger.info(f"Analysis completed successfully. Overall attrition rate: {attrition_rate:.2%}")
        return result
        
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        return {"error": f"Validation error: {str(ve)}"}
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}")
        return {"error": f"Unexpected error: {str(e)}"}

# API Routes
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "message": "Attrition Calculator API is running",
        "version": "1.0.0",
        "status": "healthy"
    })

@app.route('/analyze', methods=['POST'])
def analyze_csv():
    """Main endpoint for CSV analysis"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check file type
        if not file.filename.lower().endswith('.csv'):
            return jsonify({"error": "Please upload a CSV file"}), 400
        
        # Read file content
        try:
            csv_data = file.read().decode('utf-8')
        except UnicodeDecodeError:
            return jsonify({"error": "Invalid file encoding. Please ensure your CSV is UTF-8 encoded"}), 400
        
        # Analyze the data
        result = analyze_attrition(csv_data)
        
        # Return appropriate status code
        if "error" in result:
            return jsonify(result), 400
        else:
            return jsonify(result), 200
            
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/analyze-raw', methods=['POST'])
def analyze_raw_csv():
    """Alternative endpoint for raw CSV data"""
    try:
        # Get JSON data with CSV content
        data = request.get_json()
        
        if not data or 'csv_data' not in data:
            return jsonify({"error": "No CSV data provided"}), 400
        
        csv_data = data['csv_data']
        
        # Analyze the data
        result = analyze_attrition(csv_data)
        
        # Return appropriate status code
        if "error" in result:
            return jsonify(result), 400
        else:
            return jsonify(result), 200
            
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Error handlers
@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"error": "File too large. Maximum size is 50MB"}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
