import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from scipy import stats
import glob

class EnvSensorAnalyzer:
    """
    A class for analyzing environmental sensor data.
    This class provides functionality to load, process, analyze, and visualize
    environmental sensor data from various sources.
    """
    
    def __init__(self):
        """Initialize the EnvSensorAnalyzer with an empty dataframe."""
        self.data = None
        self.sensors = {}
        self.statistics = {}
        
    def load_csv_data(self, file_path, date_column=None, date_format=None):
        """
        Load sensor data from a CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
        date_column : str, optional
            Name of the column containing date/time information
        date_format : str, optional
            Format string for parsing dates (e.g., '%Y-%m-%d %H:%M:%S')
        
        Returns:
        --------
        bool
            True if data was loaded successfully, False otherwise
        """
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            
            # Convert date column to datetime if specified
            if date_column and date_column in self.data.columns:
                if date_format:
                    self.data[date_column] = pd.to_datetime(self.data[date_column], format=date_format)
                else:
                    self.data[date_column] = pd.to_datetime(self.data[date_column])
                self.data.set_index(date_column, inplace=True)
                print(f"Date column '{date_column}' converted to datetime and set as index.")
            
            # Identify numeric columns as potential sensor data
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.sensors = {col: {'values': self.data[col]} for col in numeric_cols}
            return True
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def load_multiple_csv_files(self, directory_path, pattern="*.csv", date_column=None, date_format=None):
        """
        Load and merge multiple CSV files from a directory.
        
        Parameters:
        -----------
        directory_path : str
            Path to the directory containing CSV files
        pattern : str, optional
            Glob pattern to match files (default: "*.csv")
        date_column : str, optional
            Name of the column containing date/time information
        date_format : str, optional
            Format string for parsing dates
            
        Returns:
        --------
        bool
            True if data was loaded successfully, False otherwise
        """
        try:
            all_files = glob.glob(os.path.join(directory_path, pattern))
            if not all_files:
                print(f"No files found matching pattern '{pattern}' in directory '{directory_path}'")
                return False
                
            print(f"Found {len(all_files)} files matching the pattern.")
            dataframes = []
            
            for file in all_files:
                df = pd.read_csv(file)
                # Add filename as a source column
                df['source_file'] = os.path.basename(file)
                dataframes.append(df)
            
            # Concatenate all dataframes
            self.data = pd.concat(dataframes, ignore_index=True)
            
            # Convert date column to datetime if specified
            if date_column and date_column in self.data.columns:
                if date_format:
                    self.data[date_column] = pd.to_datetime(self.data[date_column], format=date_format)
                else:
                    self.data[date_column] = pd.to_datetime(self.data[date_column])
                self.data.set_index(date_column, inplace=True)
            
            # Identify numeric columns as potential sensor data
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.sensors = {col: {'values': self.data[col]} for col in numeric_cols}
            
            print(f"Combined data loaded successfully. Shape: {self.data.shape}")
            return True
            
        except Exception as e:
            print(f"Error loading multiple files: {e}")
            return False
    
    def generate_sample_data(self, num_days=30, frequency='H'):
        """
        Generate sample environmental sensor data for testing.
        
        Parameters:
        -----------
        num_days : int, optional
            Number of days to generate data for (default: 30)
        frequency : str, optional
            Time series frequency (default: 'H' for hourly)
            
        Returns:
        --------
        bool
            True if sample data was generated successfully
        """
        try:
            # Create a datetime index
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=num_days)
            date_index = pd.date_range(start=start_date, end=end_date, freq=frequency)
            
            # Generate sample data with realistic patterns
            np.random.seed(42)  # For reproducibility
            
            # Temperature: Daily cycle plus random noise
            hour_of_day = np.array([d.hour for d in date_index])
            day_of_year = np.array([d.dayofyear for d in date_index])
            
            # Temperature with daily cycle and seasonal trend
            base_temp = 20 + 5 * np.sin(2 * np.pi * day_of_year / 365)  # Seasonal variation
            daily_cycle = 3 * np.sin(2 * np.pi * hour_of_day / 24)  # Daily cycle
            temperature = base_temp + daily_cycle + np.random.normal(0, 1, len(date_index))
            
            # Humidity: Inverse correlation with temperature plus noise
            humidity = 70 - 0.5 * (temperature - 20) + np.random.normal(0, 5, len(date_index))
            humidity = np.clip(humidity, 20, 100)  # Constrain to realistic values
            
            # Air quality (PM2.5): Random with occasional spikes
            air_quality = np.random.lognormal(mean=2.5, sigma=0.4, size=len(date_index))
            # Add some spikes
            spike_indices = np.random.choice(len(date_index), size=int(len(date_index)*0.05), replace=False)
            air_quality[spike_indices] *= 2.5
            
            # Light level: Correlates with time of day
            light_level = 50 + 50 * np.sin(np.pi * hour_of_day / 12 - 6)
            night_mask = (hour_of_day < 6) | (hour_of_day > 18)
            light_level[night_mask] = np.random.normal(5, 2, size=np.sum(night_mask))
            light_level = np.clip(light_level, 0, 100)
            
            # Create DataFrame
            self.data = pd.DataFrame({
                'temperature': temperature,
                'humidity': humidity,
                'air_quality_pm25': air_quality,
                'light_level': light_level
            }, index=date_index)
            
            # Set up sensors dictionary
            self.sensors = {
                'temperature': {'values': self.data['temperature'], 'unit': '°C'},
                'humidity': {'values': self.data['humidity'], 'unit': '%'},
                'air_quality_pm25': {'values': self.data['air_quality_pm25'], 'unit': 'μg/m³'},
                'light_level': {'values': self.data['light_level'], 'unit': '%'}
            }
            
            print(f"Sample data generated successfully. Shape: {self.data.shape}")
            return True
            
        except Exception as e:
            print(f"Error generating sample data: {e}")
            return False
    
    def clean_data(self, sensors=None, method='interpolate', max_gap=3):
        """
        Clean the sensor data by handling missing values and outliers.
        
        Parameters:
        -----------
        sensors : list, optional
            List of sensor columns to clean (default: all)
        method : str, optional
            Method for handling missing values: 'interpolate', 'drop', or 'fill'
        max_gap : int, optional
            Maximum size of gap to interpolate over
            
        Returns:
        --------
        bool
            True if cleaning was successful
        """
        if self.data is None:
            print("No data available to clean.")
            return False
        
        if sensors is None:
            sensors = list(self.sensors.keys())
        
        try:
            # Keep a copy of the original data
            original_data = self.data.copy()
            
            for sensor in sensors:
                if sensor not in self.data.columns:
                    print(f"Warning: '{sensor}' not found in data columns. Skipping.")
                    continue
                
                # Handle missing values
                missing_count = self.data[sensor].isna().sum()
                if missing_count > 0:
                    print(f"Found {missing_count} missing values in '{sensor}'")
                    
                    if method == 'interpolate':
                        self.data[sensor] = self.data[sensor].interpolate(method='time', limit=max_gap)
                        # Fill any remaining NaNs at the edges
                        self.data[sensor] = self.data[sensor].fillna(method='ffill').fillna(method='bfill')
                    elif method == 'drop':
                        self.data = self.data.dropna(subset=[sensor])
                    elif method == 'fill':
                        self.data[sensor] = self.data[sensor].fillna(self.data[sensor].mean())
                
                # Handle outliers using IQR method
                Q1 = self.data[sensor].quantile(0.25)
                Q3 = self.data[sensor].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((self.data[sensor] < lower_bound) | (self.data[sensor] > upper_bound))
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    print(f"Found {outlier_count} outliers in '{sensor}'")
                    # Cap the outliers at the bounds
                    self.data.loc[self.data[sensor] < lower_bound, sensor] = lower_bound
                    self.data.loc[self.data[sensor] > upper_bound, sensor] = upper_bound
                
                # Update the sensors dictionary
                self.sensors[sensor]['values'] = self.data[sensor]
                self.sensors[sensor]['cleaned'] = True
                if 'unit' not in self.sensors[sensor]:
                    self.sensors[sensor]['unit'] = 'unknown'
            
            # Calculate how many data points were modified
            modified = (original_data != self.data).any(axis=1).sum()
            print(f"Data cleaning complete. Modified {modified} data points.")
            return True
            
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return False
    
    def calculate_statistics(self, sensors=None):
        """
        Calculate basic statistics for sensor data.
        
        Parameters:
        -----------
        sensors : list, optional
            List of sensor columns to analyze (default: all)
            
        Returns:
        --------
        dict
            Dictionary containing statistics for each sensor
        """
        if self.data is None:
            print("No data available for analysis.")
            return {}
        
        if sensors is None:
            sensors = list(self.sensors.keys())
        
        try:
            for sensor in sensors:
                if sensor not in self.data.columns:
                    print(f"Warning: '{sensor}' not found in data columns. Skipping.")
                    continue
                
                values = self.data[sensor].dropna()
                
                stats_dict = {
                    'mean': values.mean(),
                    'median': values.median(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'range': values.max() - values.min(),
                    'q1': values.quantile(0.25),
                    'q3': values.quantile(0.75),
                    'iqr': values.quantile(0.75) - values.quantile(0.25),
                    'skewness': values.skew(),
                    'kurtosis': values.kurtosis(),
                    'count': len(values),
                    'missing': self.data[sensor].isna().sum(),
                    'unit': self.sensors[sensor].get('unit', 'unknown')
                }
                
                self.statistics[sensor] = stats_dict
            
            return self.statistics
            
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return {}
    
    def detect_anomalies(self, sensor, window=None, threshold=3.0, method='zscore'):
        """
        Detect anomalies in sensor data.
        
        Parameters:
        -----------
        sensor : str
            Name of the sensor column to analyze
        window : int, optional
            Window size for rolling calculations (default: None - use entire series)
        threshold : float, optional
            Threshold for anomaly detection (default: 3.0)
        method : str, optional
            Method for anomaly detection: 'zscore', 'iqr', or 'rolling'
            
        Returns:
        --------
        pandas.Series
            Boolean series indicating anomalies (True for anomalies)
        """
        if self.data is None or sensor not in self.data.columns:
            print(f"No data available for sensor '{sensor}'.")
            return None
        
        try:
            values = self.data[sensor].copy()
            anomalies = pd.Series(False, index=values.index)
            
            if method == 'zscore':
                # Z-score method
                if window is not None:
                    # Rolling z-score
                    rolling_mean = values.rolling(window=window).mean()
                    rolling_std = values.rolling(window=window).std()
                    z_scores = np.abs((values - rolling_mean) / rolling_std)
                    anomalies = z_scores > threshold
                else:
                    # Global z-score
                    z_scores = np.abs((values - values.mean()) / values.std())
                    anomalies = z_scores > threshold
                
            elif method == 'iqr':
                # IQR method
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                anomalies = (values < lower_bound) | (values > upper_bound)
                
            elif method == 'rolling':
                # Rolling average deviation
                if window is None:
                    window = min(30, len(values) // 10)  # Default to either 30 or 1/10 of the data
                
                rolling_mean = values.rolling(window=window, center=True).mean()
                rolling_std = values.rolling(window=window, center=True).std()
                anomalies = np.abs(values - rolling_mean) > (threshold * rolling_std)
            
            # Count and report anomalies
            anomaly_count = anomalies.sum()
            print(f"Detected {anomaly_count} anomalies in '{sensor}' using {method} method.")
            
            # Store anomalies in the sensor info
            self.sensors[sensor]['anomalies'] = anomalies
            
            return anomalies
            
        except Exception as e:
            print(f"Error detecting anomalies: {e}")
            return None
    
    def find_sensor_correlations(self, sensors=None, method='pearson'):
        """
        Calculate correlations between different sensors.
        
        Parameters:
        -----------
        sensors : list, optional
            List of sensor columns to analyze (default: all)
        method : str, optional
            Correlation method: 'pearson', 'kendall', or 'spearman'
            
        Returns:
        --------
        pandas.DataFrame
            Correlation matrix
        """
        if self.data is None:
            print("No data available for correlation analysis.")
            return None
        
        if sensors is None:
            sensors = list(self.sensors.keys())
        
        # Filter to only include sensors that are in the data
        valid_sensors = [s for s in sensors if s in self.data.columns]
        
        if len(valid_sensors) < 2:
            print("At least two valid sensors are needed for correlation analysis.")
            return None
        
        try:
            correlation_matrix = self.data[valid_sensors].corr(method=method)
            return correlation_matrix
            
        except Exception as e:
            print(f"Error calculating correlations: {e}")
            return None
    
    def plot_time_series(self, sensors=None, start_date=None, end_date=None, figsize=(12, 6)):
        """
        Plot time series data for selected sensors.
        
        Parameters:
        -----------
        sensors : list, optional
            List of sensor columns to plot (default: all)
        start_date : str or datetime, optional
            Start date for the plot
        end_date : str or datetime, optional
            End date for the plot
        figsize : tuple, optional
            Figure size (width, height) in inches
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure object
        """
        if self.data is None:
            print("No data available for plotting.")
            return None
        
        if sensors is None:
            sensors = list(self.sensors.keys())
        
        # Filter to only include sensors that are in the data
        valid_sensors = [s for s in sensors if s in self.data.columns]
        
        if not valid_sensors:
            print("No valid sensors found for plotting.")
            return None
        
        try:
            # Filter data by date range if specified
            plot_data = self.data.copy()
            if start_date is not None or end_date is not None:
                plot_data = plot_data.loc[start_date:end_date]
            
            # Create the plot
            fig, axes = plt.subplots(len(valid_sensors), 1, figsize=figsize, sharex=True)
            if len(valid_sensors) == 1:
                axes = [axes]  # Make it iterable if there's only one sensor
            
            for i, sensor in enumerate(valid_sensors):
                ax = axes[i]
                plot_data[sensor].plot(ax=ax, label=sensor)
                
                # Plot anomalies if they exist
                if sensor in self.sensors and 'anomalies' in self.sensors[sensor]:
                    anomalies = self.sensors[sensor]['anomalies']
                    if not anomalies.empty:
                        # Filter anomalies to the plot date range
                        plot_anomalies = anomalies.loc[plot_data.index]
                        if plot_anomalies.any():
                            ax.scatter(
                                plot_data.index[plot_anomalies], 
                                plot_data.loc[plot_anomalies.index, sensor],
                                color='red', marker='o', s=30, label='Anomalies'
                            )
                
                unit = self.sensors[sensor].get('unit', '')
                ax.set_ylabel(f"{sensor} ({unit})" if unit else sensor)
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error plotting time series: {e}")
            return None
    
    def plot_distribution(self, sensors=None, bins=30, figsize=(12, 8)):
        """
        Plot distribution histograms for selected sensors.
        
        Parameters:
        -----------
        sensors : list, optional
            List of sensor columns to plot (default: all)
        bins : int, optional
            Number of histogram bins
        figsize : tuple, optional
            Figure size (width, height) in inches
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure object
        """
        if self.data is None:
            print("No data available for plotting.")
            return None
        
        if sensors is None:
            sensors = list(self.sensors.keys())
        
        # Filter to only include sensors that are in the data
        valid_sensors = [s for s in sensors if s in self.data.columns]
        
        if not valid_sensors:
            print("No valid sensors found for plotting.")
            return None
        
        try:
            # Determine grid layout
            n_sensors = len(valid_sensors)
            n_cols = min(3, n_sensors)
            n_rows = (n_sensors + n_cols - 1) // n_cols  # Ceiling division
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = np.array(axes).flatten()  # Convert to 1D array for easy indexing
            
            for i, sensor in enumerate(valid_sensors):
                ax = axes[i]
                
                # Plot histogram with KDE
                sns.histplot(self.data[sensor].dropna(), kde=True, bins=bins, ax=ax)
                
                # Add statistical annotations
                if sensor in self.statistics:
                    stats = self.statistics[sensor]
                    mean_val = stats['mean']
                    median_val = stats['median']
                    std_val = stats['std']
                    
                    # Add vertical lines for mean and median
                    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                    ax.axvline(median_val, color='green', linestyle='-.', alpha=0.8, label=f'Median: {median_val:.2f}')
                    
                    # Add text annotation for std dev
                    ax.text(0.05, 0.95, f'Std Dev: {std_val:.2f}', 
                            transform=ax.transAxes, fontsize=10,
                            verticalalignment='top')
                
                unit = self.sensors[sensor].get('unit', '')
                ax.set_xlabel(f"{sensor} ({unit})" if unit else sensor)
                ax.set_ylabel('Frequency')
                ax.legend(fontsize='small')
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error plotting distributions: {e}")
            return None
    
    def plot_correlation_matrix(self, sensors=None, method='pearson', figsize=(10, 8)):
        """
        Plot a correlation matrix heatmap for the sensors.
        
        Parameters:
        -----------
        sensors : list, optional
            List of sensor columns to include (default: all)
        method : str, optional
            Correlation method: 'pearson', 'kendall', or 'spearman'
        figsize : tuple, optional
            Figure size (width, height) in inches
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure object
        """
        corr_matrix = self.find_sensor_correlations(sensors, method)
        
        if corr_matrix is None or corr_matrix.empty:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask for upper triangle
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                        annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .8},
                        ax=ax)
            
            ax.set_title(f'Sensor Correlation Matrix ({method.capitalize()})')
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            print(f"Error plotting correlation matrix: {e}")
            return None
    
    def resample_data(self, rule='1H', sensors=None, agg_func='mean'):
        """
        Resample time series data to a different frequency.
        
        Parameters:
        -----------
        rule : str, optional
            Resampling frequency (e.g., '1H', '1D', '1W')
        sensors : list, optional
            List of sensor columns to resample (default: all)
        agg_func : str or dict, optional
            Aggregation function(s) to use ('mean', 'median', 'sum', etc.)
            Can be a dictionary mapping sensors to specific functions
            
        Returns:
        --------
        pandas.DataFrame
            Resampled dataframe
        """
        if self.data is None:
            print("No data available for resampling.")
            return None
        
        if not isinstance(self.data.index, pd.DatetimeIndex):
            print("Data index is not a DatetimeIndex. Cannot resample.")
            return None
        
        if sensors is None:
            sensors = list(self.sensors.keys())
        
        # Filter to only include sensors that are in the data
        valid_sensors = [s for s in sensors if s in self.data.columns]
        
        if not valid_sensors:
            print("No valid sensors found for resampling.")
            return None
        
        try:
            # Handle different types of aggregation functions
            if isinstance(agg_func, dict):
                # Only keep valid sensor columns in the agg_func dict
                agg_func = {k: v for k, v in agg_func.items() if k in valid_sensors}
            else:
                # Use the same agg_func for all sensors
                agg_func = {sensor: agg_func for sensor in valid_sensors}
            
            # Perform resampling
            resampled_data = self.data[valid_sensors].resample(rule).agg(agg_func)
            
            print(f"Data resampled to '{rule}' frequency using {agg_func}.")
            return resampled_data
            
        except Exception as e:
            print(f"Error resampling data: {e}")
            return None
    
    def export_to_csv(self, file_path, sensors=None, include_stats=False):
        """
        Export data and optionally statistics to a CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to save the CSV file
        sensors : list, optional
            List of sensor columns to export (default: all)
        include_stats : bool, optional
            Whether to include statistics in the export
            
        Returns:
        --------
        bool
            True if export was successful
        """
        if self.data is None:
            print("No data available for export.")
            return False
        
        if sensors is None:
            sensors = list(self.sensors.keys())
        
        # Filter to only include sensors that are in the data
        valid_sensors = [s for s in sensors if s in self.data.columns]
        
        if not valid_sensors:
            print("No valid sensors found for export.")
            return False
        
        try:
            # Prepare data for export
            export_data = self.data[valid_sensors].copy()
            
            # Reset index if it's a DatetimeIndex
            if isinstance(export_data.index, pd.DatetimeIndex):
                export_data = export_data.reset_index()
            
            # Export the data
            export_data.to_csv(file_path, index=False)
            print(f"Data exported successfully to {file_path}")
            
            # Export statistics if requested
            if include_stats and self.statistics:
                stats_file = os.path.splitext(file_path)[0] + "_stats.csv"
                
                # Convert statistics dict to dataframe
                stats_data = []
                for sensor, stats in self.statistics.items():
                    if sensor in valid_sensors:
                        stats_row = {'sensor': sensor}
                        stats_row.update(stats)
                        stats_data.append(stats_row)
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_csv(stats_file, index=False)
                    print(f"Statistics exported successfully to {stats_file}")
            
            return True
            
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False
    
    def print_summary(self, sensors=None):
        """
        Print a summary of the sensor data and statistics.
        
        Parameters:
        -----------
        sensors : list, optional
            List of sensor columns to include in the summary (default: all)
        """
        if self.data is None:
            print("No data available for summary.")
            return
        
        if sensors is None:
            sensors = list(self.sensors.keys())
        
        # Filter to only include sensors that are in the data
        valid_sensors = [s for s in sensors if s in self.data.columns]
        
        if not valid_sensors:
            print("No valid sensors found for summary.")
            return
        
        # Print dataset overview
        print("\n" + "="*50)
        print("ENVIRONMENTAL SENSOR DATA SUMMARY")
        print("="*50)
        
        print(f"\nDataset Shape: {self.data.shape}")
        
        if isinstance(self.data.index, pd.DatetimeIndex):
            print(f"Date Range: {self.data.index.min()} to {self.data.index.max()}")
            print(f"Time Period: {(self.data.index.max() - self.data.index.min())}")
        
        # Print sensor statistics
        print("\nSENSOR STATISTICS:")
        print("-"*50)
        
        for sensor in valid_sensors:
            unit = self.sensors[sensor].get('unit', '')
            unit_str = f" ({unit})" if unit else ""
            
            print(f"\n• {sensor}{unit_str}:")
            
            if sensor in self.statistics:
                stats = self.statistics[sensor]
                print(f"  - Range: {stats['min']:.2f} to {stats['max']:.2f}{unit_str}")
                print(f"  - Mean: {stats['mean']:.2f}{unit_str}")
                print(f"  - Median: {stats['median']:.2f}{unit_str}")
                print(f"  - Std Dev: {stats['std']:.2f}{unit_str}")
                print(f"  - Data Points: {stats['count']}")
                if stats['missing'] > 0:
                    print(f"  - Missing Values: {stats['missing']}")
            else:
                values = self.data[sensor].dropna()
                print(f"  - Range: {values.min():.2f} to {values.max():.2f}{unit_str}")
                print(f"  - Mean: {values.mean():.2f}{unit_str}")
                print(f"  - Data Points: {len(values)}")
            
            # Print anomaly info if available
            if sensor in self.sensors and 'anomalies' in self.sensors[sensor]:
                anomalies = self.sensors[sensor]['anomalies']
                if not anomalies.empty:
                    anomaly_count = anomalies.sum()
                    print(f"  - Anomalies Detected: {anomaly_count} ({anomaly_count/len(anomalies)*100:.1f}%)")
        
        # Print correlation highlights if available
        if len(valid_sensors) > 1:
            print("\nSENSOR CORRELATIONS:")
            print("-"*50)
            
            corr_matrix = self.find_sensor_correlations(valid_sensors)
            if corr_matrix is not None:
                # Find highest correlations (excluding self-correlation)
                correlations = []
                for i in range(len(valid_sensors)):
                    for j in range(i+1, len(valid_sensors)):
                        sensor1 = valid_sensors[i]
                        sensor2 = valid_sensors[j]
                        corr_value = corr_matrix.loc[sensor1, sensor2]
                        correlations.append((sensor1, sensor2, corr_value))
                
                # Sort by absolute correlation value
                correlations.sort(key=lambda x: abs(x[2]), reverse=True)
                
                # Print top correlations
                for sensor1, sensor2, corr_value in correlations[:5]:  # Top 5
                    direction = "positive" if corr_value > 0 else "negative"
                    strength = "strong" if abs(corr_value) > 0.7 else "moderate" if abs(corr_value) > 0.4 else "weak"
                    print(f"  - {sensor1} and {sensor2}: {strength} {direction} correlation ({corr_value:.2f})")
        
        print("\n" + "="*50)


# Example usage
if __name__ == "__main__":
    # Create the analyzer
    analyzer = EnvSensorAnalyzer()
    
    # Generate sample data
    print("Generating sample environmental sensor data...")
    analyzer.generate_sample_data(num_days=60, frequency='30min')
    
    # Clean the data
    print("\nCleaning the data...")
    analyzer.clean_data()
    
    # Calculate statistics
    print("\nCalculating sensor statistics...")
    analyzer.calculate_statistics()
    
    # Detect anomalies in temperature data
    print("\nDetecting anomalies in temperature data...")
    analyzer.detect_anomalies('temperature', method='zscore', threshold=3.0)
    
    # Print a summary
    analyzer.print_summary()
    
    # Create and display plots
    print("\nCreating plots...")
    
    # Time series plot
    fig_time = analyzer.plot_time_series(figsize=(12, 10))
    if fig_time:
        plt.figure(fig_time.number)
        plt.suptitle('Environmental Sensor Time Series Data', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig('sensor_time_series.png')
        print("Time series plot saved as 'sensor_time_series.png'")
    
    # Distribution plot
    fig_dist = analyzer.plot_distribution()
    if fig_dist:
        plt.figure(fig_dist.number)
        plt.suptitle('Sensor Data Distributions', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig('sensor_distributions.png')
        print("Distribution plot saved as 'sensor_distributions.png'")
    
    # Correlation matrix
    fig_corr = analyzer.plot_correlation_matrix()
    if fig_corr:
        plt.savefig('sensor_correlations.png')
        print("Correlation matrix saved as 'sensor_correlations.png'")
    
    # Resample data to daily averages
    print("\nResampling data to daily averages...")
    daily_data = analyzer.resample_data(rule='1D')
    if daily_data is not None:
        print(daily_data.head())
    
    # Export the data
    print("\nExporting data to CSV...")
    analyzer.export_to_csv('environmental_sensor_data.csv', include_stats=True)
    
    print("\nAnalysis complete!")