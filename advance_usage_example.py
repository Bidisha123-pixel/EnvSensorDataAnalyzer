"""
Environmental Sensor Data Analysis - Advanced Examples
=====================================================

This script demonstrates more advanced usage of the EnvSensorAnalyzer class
for analyzing environmental sensor data from CSV files and performing
more complex analyses.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from env_sensor_analyzer import EnvSensorAnalyzer

def create_sample_csv():
    """Create a sample CSV file with simulated environmental data"""
    print("Creating a sample CSV file...")
    
    # Set random seed for reproducibility
    np.random.seed(123)
    
    # Create date range
    dates = pd.date_range(start='2025-01-01', end='2025-04-30', freq='H')
    
    # Create temperature data with seasonal and daily patterns
    hour_of_day = np.array([d.hour for d in dates])
    day_of_year = np.array([d.dayofyear for d in dates])
    
    # Base temperature pattern (seasonal)
    base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 15) / 365)
    
    # Daily cycle
    daily_cycle = 5 * np.sin(2 * np.pi * (hour_of_day - 10) / 24)
    
    # Combine patterns and add noise
    temperature = base_temp + daily_cycle + np.random.normal(0, 1.5, len(dates))
    
    # Create humidity data (inversely related to temperature with some noise)
    humidity = 80 - 0.8 * (temperature - 15) + np.random.normal(0, 8, len(dates))
    humidity = np.clip(humidity, 10, 100)  # Clip to valid range
    
    # Create air quality data (PM2.5, PM10)
    # Base pattern with weekday-weekend differences and random events
    is_weekend = np.array([1 if d.weekday() >= 5 else 0 for d in dates])
    
    # PM2.5 pattern
    pm25_base = 15 + 8 * np.sin(2 * np.pi * hour_of_day / 24)  # Daily cycle
    pm25_base -= 5 * is_weekend  # Lower on weekends
    
    # PM10 pattern (correlated with PM2.5 but higher values)
    pm10_base = 25 + 15 * np.sin(2 * np.pi * hour_of_day / 24)
    pm10_base -= 8 * is_weekend
    
    # Add some pollution events (high values in random periods)
    n_events = 10
    event_indices = np.random.choice(len(dates) - 48, n_events, replace=False)
    
    pm25 = pm25_base.copy()
    pm10 = pm10_base.copy()
    
    for idx in event_indices:
        # Events last between 4-12 hours
        event_duration = np.random.randint(4, 13)
        event_magnitude = np.random.uniform(2.0, 4.0)
        
        # Create a spike in pollution values
        pm25[idx:idx+event_duration] *= event_magnitude
        pm10[idx:idx+event_duration] *= event_magnitude
    
    # Add random noise
    pm25 += np.random.lognormal(0, 0.3, len(dates))
    pm10 += np.random.lognormal(0, 0.4, len(dates))
    
    # Create wind speed and direction
    wind_speed = 5 + 3 * np.sin(2 * np.pi * day_of_year / 365) + np.random.gamma(2, 1.5, len(dates))
    wind_direction = np.random.uniform(0, 360, len(dates))
    
    # Create light level (correlated with hour of day)
    light_base = np.zeros(len(dates))
    daytime = (hour_of_day >= 6) & (hour_of_day <= 18)
    light_base[daytime] = 100 * np.sin(np.pi * (hour_of_day[daytime] - 6) / 12)
    light_level = light_base + np.random.normal(0, 5, len(dates))
    light_level = np.clip(light_level, 0, 100)
    
    # Add some missing values randomly
    for col in [temperature, humidity, pm25, pm10, wind_speed, wind_direction, light_level]:
        missing_idx = np.random.choice(len(col), size=int(len(col)*0.02), replace=False)
        col[missing_idx] = np.nan
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'temperature': temperature,
        'humidity': humidity,
        'pm25': pm25,
        'pm10': pm10,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
        'light_level': light_level
    })
    
    # Save to CSV
    data.to_csv('environmental_sample_data.csv', index=False)
    print(f"Sample CSV created with {len(data)} records spanning from {dates[0].date()} to {dates[-1].date()}")
    return 'environmental_sample_data.csv'

def main():
    # Create a sample CSV file
    csv_file = create_sample_csv()
    
    # Create the analyzer
    analyzer = EnvSensorAnalyzer()
    
    # Load the CSV data
    print("\nLoading CSV data...")
    analyzer.load_csv_data(csv_file, date_column='timestamp')
    
    # Set sensor units
    print("Setting sensor units...")
    analyzer.sensors['temperature']['unit'] = '°C'
    analyzer.sensors['humidity']['unit'] = '%'
    analyzer.sensors['pm25']['unit'] = 'μg/m³'
    analyzer.sensors['pm10']['unit'] = 'μg/m³'
    analyzer.sensors['wind_speed']['unit'] = 'm/s'
    analyzer.sensors['wind_direction']['unit'] = '°'
    analyzer.sensors['light_level']['unit'] = '%'
    
    # Clean the data
    print("\nCleaning sensor data...")
    analyzer.clean_data(method='interpolate')
    
    # Calculate statistics for each sensor
    print("\nCalculating statistics...")
    statistics = analyzer.calculate_statistics()
    
    # Detect anomalies in PM2.5 data using rolling window method
    print("\nDetecting anomalies in air quality data...")
    pm25_anomalies = analyzer.detect_anomalies('pm25', window=24, threshold=3.5, method='rolling')
    
    # Advanced analysis: Create a derived air quality index
    print("\nCreating derived air quality index...")
    
    # Simple air quality index formula (for demonstration)
    def calculate_aqi(row):
        pm25_weight = 0.7
        pm10_weight = 0.3
        pm25_val = min(row['pm25'] / 35.0, 1.0) * 100  # Normalize to 0-100
        pm10_val = min(row['pm10'] / 150.0, 1.0) * 100  # Normalize to 0-100
        return pm25_weight * pm25_val + pm10_weight * pm10_val
    
    # Add AQI to the data
    analyzer.data['air_quality_index'] = analyzer.data.apply(calculate_aqi, axis=1)
    analyzer.sensors['air_quality_index'] = {'values': analyzer.data['air_quality_index'], 'unit': 'index'}
    analyzer.calculate_statistics(['air_quality_index'])
    
    # Advanced correlation analysis
    print("\nAnalyzing correlations between weather parameters and air quality...")
    corr_matrix = analyzer.find_sensor_correlations(method='pearson')
    
    print("\nTop correlations with air quality:")
    # Sort correlations with air quality index
    aqi_correlations = [(col, corr_matrix.loc['air_quality_index', col]) 
                         for col in corr_matrix.columns 
                         if col != 'air_quality_index']
    aqi_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for param, corr in aqi_correlations:
        print(f"  - {param}: {corr:.3f}")
    
    # Resample data to daily values with different aggregation methods per sensor
    print("\nResampling data to daily values with specific aggregation methods...")
    agg_funcs = {
        'temperature': ['mean', 'min', 'max'],
        'humidity': 'mean',
        'pm25': 'mean',
        'pm10': 'mean',
        'wind_speed': 'mean',
        'wind_direction': lambda x: np.rad2deg(np.arctan2(
            np.sum(np.sin(np.deg2rad(x))), 
            np.sum(np.cos(np.deg2rad(x)))
        )) % 360,  # Circular mean for directions
        'light_level': 'max',
        'air_quality_index': ['mean', 'max']
    }
    daily_data = analyzer.resample_data(rule='1D', agg_func=agg_funcs)
    print("\nDaily resampled data:")
    print(daily_data.head())
    
    # Advanced visualizations
    print("\nCreating advanced visualizations...")
    
    # 1. Time series with PM2.5 anomalies highlighted
    plt.figure(figsize=(14, 6))
    plt.plot(analyzer.data.index, analyzer.data['pm25'], label='PM2.5', color='blue', alpha=0.7)
    
    # Highlight anomalies
    anomaly_points = analyzer.data.index[pm25_anomalies]
    anomaly_values = analyzer.data.loc[pm25_anomalies, 'pm25']
    plt.scatter(anomaly_points, anomaly_values, color='red', s=30, label='Anomalies')
    
    plt.title('PM2.5 Levels with Detected Anomalies', fontsize=14)
    plt.ylabel('PM2.5 (μg/m³)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pm25_anomalies.png')
    print("PM2.5 anomalies plot saved as 'pm25_anomalies.png'")
    
    # 2. Scatter plot showing relationship between temperature and humidity
    plt.figure(figsize=(10, 8))
    plt.scatter(analyzer.data['temperature'], analyzer.data['humidity'], 
                alpha=0.5, c=analyzer.data['air_quality_index'], cmap='viridis')
    plt.colorbar(label='Air Quality Index')
    plt.title('Temperature vs. Humidity, Colored by Air Quality Index', fontsize=14)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Humidity (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('temp_humidity_aqi.png')
    print("Temperature vs. humidity scatter plot saved as 'temp_humidity_aqi.png'")
    
    # 3. Daily patterns - aggregating by hour of day
    print("\nAnalyzing daily patterns...")
    analyzer.data['hour'] = analyzer.data.index.hour
    
    hourly_patterns = analyzer.data.groupby('hour').agg({
        'temperature': 'mean',
        'humidity': 'mean', 
        'pm25': 'mean',
        'air_quality_index': 'mean'
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    
    sensors = ['temperature', 'humidity', 'pm25', 'air_quality_index']
    titles = ['Temperature', 'Humidity', 'PM2.5', 'Air Quality Index']
    units = ['°C', '%', 'μg/m³', 'index']
    
    for i, (sensor, title, unit) in enumerate(zip(sensors, titles, units)):
        axes[i].plot(hourly_patterns.index, hourly_patterns[sensor], marker='o', markersize=4)
        axes[i].set_title(f'Average {title} by Hour of Day', fontsize=12)
        axes[i].set_ylabel(f'{title} ({unit})')
        axes[i].grid(True, alpha=0.3)
        
        # Add min/max text annotations
        min_hour = hourly_patterns[sensor].idxmin()
        max_hour = hourly_patterns[sensor].idxmax()
        min_val = hourly_patterns[sensor].min()
        max_val = hourly_patterns[sensor].max()
        
        axes[i].annotate(f'Min: {min_val:.1f}', xy=(min_hour, min_val),
                        xytext=(min_hour, min_val-hourly_patterns[sensor].std()),
                        arrowprops=dict(arrowstyle='->'))
        
        axes[i].annotate(f'Max: {max_val:.1f}', xy=(max_hour, max_val),
                        xytext=(max_hour, max_val+hourly_patterns[sensor].std()),
                        arrowprops=dict(arrowstyle='->'))
    
    for ax in axes:
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel('Hour of Day')
    
    plt.tight_layout()
    plt.savefig('daily_patterns.png')
    print("Daily patterns plot saved as 'daily_patterns.png'")
    
    # Export the analyzed data and results
    print("\nExporting analyzed data with derived metrics...")
    analyzer.export_to_csv('analyzed_environmental_data.csv', include_stats=True)
    
    # Print a comprehensive summary
    analyzer.print_summary()
    
    print("\nAdvanced analysis complete!")

if __name__ == "__main__":
    main()