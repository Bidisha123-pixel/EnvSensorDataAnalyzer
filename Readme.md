# Environmental Sensor Data Analyzer

A Python-based tool for loading, processing, analyzing, and visualizing environmental sensor data. This project provides comprehensive functionality for working with various environmental sensor measurements such as temperature, humidity, air quality, and more.

## Features

- **Data loading**: Load sensor data from CSV files or generate realistic sample data
- **Data cleaning**: Handle missing values and outliers automatically
- **Statistical analysis**: Calculate key statistics for each sensor
- **Anomaly detection**: Identify anomalies using various methods (Z-score, IQR, rolling windows)
- **Visualization**: Generate time series plots, distributions, and correlation matrices
- **Data resampling**: Change the frequency of time series data (e.g., hourly to daily)
- **Data export**: Save processed data and statistics to CSV files

## Installation

### Prerequisites

- Python 3.7 or higher
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scipy

### Setup

1. Clone this repository or download the source files
2. Install required packages:

```bash
pip install pandas numpy matplotlib seaborn scipy
```

## Usage

### Basic Usage

```python
from env_sensor_analyzer import EnvSensorAnalyzer

# Create the analyzer
analyzer = EnvSensorAnalyzer()

# Either generate sample data
analyzer.generate_sample_data(num_days=30)

# Or load data from a CSV file
# analyzer.load_csv_data('your_sensor_data.csv', date_column='timestamp')

# Clean the data
analyzer.clean_data()

# Calculate statistics
analyzer.calculate_statistics()

# Print a summary of the data
analyzer.print_summary()

# Create visualizations
fig_time = analyzer.plot_time_series()
fig_dist = analyzer.plot_distribution()
fig_corr = analyzer.plot_correlation_matrix()

# Export data
analyzer.export_to_csv('analyzed_data.csv', include_stats=True)
```

### Quick Explorer

For quick data exploration, use the included explorer script:

```bash
python quick_explorer.py --sample
```

Or to explore your own data:

```bash
python quick_explorer.py --file your_data.csv --date-column timestamp --clean
```

### Advanced Usage

Check out the `advanced_usage_example.py` script for more complex analysis patterns, including:

- Creating derived metrics from raw sensor data
- Advanced correlation analysis
- Custom visualization techniques
- Daily pattern analysis
- Specialized data resampling

## File Descriptions

- `env_sensor_analyzer.py`: Main analyzer class with all functionality
- `quick_explorer.py`: Command-line interface for quick data exploration
- `advanced_usage_example.py`: Advanced usage examples and techniques

## Command-Line Options for Quick Explorer

```
usage: quick_explorer.py [-h] [--file FILE] [--date-column DATE_COLUMN]
                         [--date-format DATE_FORMAT] [--resample RESAMPLE]
                         [--start-date START_DATE] [--end-date END_DATE]
                         [--sensors SENSORS [SENSORS ...]] [--output OUTPUT]
                         [--sample] [--clean]

Explore environmental sensor data

options:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  Path to CSV file with sensor data
  --date-column DATE_COLUMN, -d DATE_COLUMN
                        Name of the date column in the CSV
  --date-format DATE_FORMAT
                        Format of dates in the date column (e.g., "%Y-%m-%d %H:%M:%S")
  --resample RESAMPLE, -r RESAMPLE
                        Resample data to specified frequency (e.g., "1H", "1D")
  --start-date START_DATE
                        Start date for filtering data (YYYY-MM-DD)
  --end-date END_DATE   End date for filtering data (YYYY-MM-DD)
  --sensors SENSORS [SENSORS ...], -s SENSORS [SENSORS ...]
                        Specific sensors to analyze
  --output OUTPUT, -o OUTPUT
                        Path to save exported data (CSV)
  --sample              Use sample data instead of a file
  --clean               Clean the data before analysis
```

## Example Data Format

The analyzer works with CSV files containing environmental sensor data. The expected format is:

```
timestamp,temperature,humidity,pm25,pm10,...
2025-01-01 00:00:00,20.5,65.2,15.3,28.7,...
2025-01-01 01:00:00,19.8,68.5,14.8,27.2,...
...
```

The date column can have any name but should be specified when loading the data.
