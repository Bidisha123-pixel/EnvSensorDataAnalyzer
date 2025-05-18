"""
Quick Environmental Data Exploration

This script provides a simple interface for quick exploration of environmental
sensor data using the EnvSensorAnalyzer class.
"""

import argparse
import matplotlib.pyplot as plt
from env_sensor_analyzer import EnvSensorAnalyzer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Explore environmental sensor data')
    parser.add_argument('--file', '-f', help='Path to CSV file with sensor data')
    parser.add_argument('--date-column', '-d', default=None, help='Name of the date column in the CSV')
    parser.add_argument('--date-format', default=None, help='Format of dates in the date column (e.g., "%%Y-%%m-%%d %%H:%%M:%%S")')
    parser.add_argument('--resample', '-r', default=None, help='Resample data to specified frequency (e.g., "1H", "1D")')
    parser.add_argument('--start-date', default=None, help='Start date for filtering data (YYYY-MM-DD)')
    parser.add_argument('--end-date', default=None, help='End date for filtering data (YYYY-MM-DD)')
    parser.add_argument('--sensors', '-s', nargs='+', help='Specific sensors to analyze')
    parser.add_argument('--output', '-o', default=None, help='Path to save exported data (CSV)')
    parser.add_argument('--sample', action='store_true', help='Use sample data instead of a file')
    parser.add_argument('--clean', action='store_true', help='Clean the data before analysis')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    analyzer = EnvSensorAnalyzer()
    
    # Either load from file or generate sample data
    if args.sample:
        print("Generating sample environmental data...")
        analyzer.generate_sample_data(num_days=30)
    elif args.file:
        print(f"Loading data from {args.file}...")
        analyzer.load_csv_data(args.file, date_column=args.date_column, date_format=args.date_format)
    else:
        print("No data source specified. Use --file or --sample.")
        return
    
    # Clean the data if requested
    if args.clean:
        print("Cleaning data...")
        analyzer.clean_data()
    
    # Calculate statistics for all sensors or specified ones
    print("Calculating statistics...")
    statistics = analyzer.calculate_statistics(sensors=args.sensors)
    
    # Print a summary
    analyzer.print_summary(sensors=args.sensors)
    
    # If resampling is requested
    if args.resample:
        print(f"Resampling data to {args.resample} frequency...")
        resampled_data = analyzer.resample_data(rule=args.resample, sensors=args.sensors)
        if resampled_data is not None:
            print("\nResampled data preview:")
            print(resampled_data.head())
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Time series plot
    fig_time = analyzer.plot_time_series(
        sensors=args.sensors, 
        start_date=args.start_date, 
        end_date=args.end_date,
        figsize=(12, 8)
    )
    
    if fig_time:
        plt.figure(fig_time.number)
        plt.suptitle('Sensor Time Series Data', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)
    
    # Distribution plot
    fig_dist = analyzer.plot_distribution(sensors=args.sensors)
    if fig_dist:
        plt.figure(fig_dist.number)
        plt.suptitle('Sensor Data Distributions', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)
    
    # Correlation matrix (only if multiple sensors)
    if not args.sensors or len(args.sensors) > 1:
        fig_corr = analyzer.plot_correlation_matrix(sensors=args.sensors)
        if fig_corr:
            plt.show(block=False)
    
    # Export if requested
    if args.output:
        print(f"\nExporting data to {args.output}...")
        analyzer.export_to_csv(args.output, sensors=args.sensors, include_stats=True)
    
    print("\nExploration complete! Close plot windows to exit.")
    plt.show()  # Block until all windows are closed

if __name__ == "__main__":
    main()