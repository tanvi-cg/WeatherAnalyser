import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date, timedelta

DATE_COL = "04-12-2025"
TEMP_COL = "19"
RAIN_COL = "0%"
HUMIDITY_COL = "57%"
CLEANED_CSV_PATH = (r"/Users/tanvichoudhary/Documents/Assignment4/cleaned_weather.csv","a")

def generate_mock_data(start_date: str = '2024-01-01', days: int = 365) -> pd.DataFrame:
    print(f"Generating mock data for {days} days...")
    
    dates = [date.fromisoformat(start_date) + timedelta(days=i) for i in range(days)]
    
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    temp_base = 20 + 10 * np.sin(2 * np.pi * day_of_year / 365)
    temperatures = np.round(temp_base + np.random.normal(0, 3, days), 1)
    
    humidity_base = 65 - 10 * np.sin(2 * np.pi * day_of_year / 365)
    humidities = np.clip(np.round(humidity_base + np.random.normal(0, 5, days)), 40, 95).astype(int)
    
    rainfall = np.random.choice([0.0, 0.0, 0.0, 0.1, 0.5, 2.0, 5.0, 10.0], days, p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01])
    
    temperatures[50:55] = np.nan
    rainfall[200] = np.nan
    
    data = pd.DataFrame({
        DATE_COL: dates,
        TEMP_COL: temperatures,
        RAIN_COL: rainfall,
        HUMIDITY_COL: humidities
    })
    
    data.to_csv('mock_raw_data.csv', index=False)
    print("Mock data saved to 'mock_raw_data.csv'.")
    return data

if __name__ == "__main__":
    
    raw_df = generate_mock_data(days=365)
    
    if raw_df.empty:
        print("\nCould not proceed due to missing or empty data.")
    else:
        
        print("\n--- Data Inspection ---")
        print("\nDataFrame Head:")
        print(raw_df.head())
        
        print("\n--- Data Cleaning and Processing ---")
        
        df = raw_df.copy()
        
        if DATE_COL in df.columns:
            df[DATE_COL] = pd.to_datetime(df[DATE_COL])
            df = df.set_index(DATE_COL).sort_index()
            print(f"Set '{DATE_COL}' as datetime index.")
        
        required_cols = [TEMP_COL, RAIN_COL, HUMIDITY_COL]
        df = df[required_cols]
        
        print(f"Missing values before cleaning:\n{df.isnull().sum()}")
        
        df[TEMP_COL].fillna(df[TEMP_COL].rolling(window=7, min_periods=1, center=True).mean(), inplace=True)
        
        df.dropna(inplace=True)
        
        print(f"Missing values after cleaning:\n{df.isnull().sum()}")
        print(f"Cleaned data shape: {df.shape}")
        
        if df.empty:
            print("\nCleaned DataFrame is empty. Cannot proceed with analysis.")
        else:
            
            print("\n--- Statistical Analysis ---")
            
            print("\n--- Summary of Daily Data (NumPy) ---")
            temp_series = df[TEMP_COL].to_numpy()
            print(f"Max Temperature: {np.max(temp_series):.1f}°C")
            print(f"Min Temperature: {np.min(temp_series):.1f}°C")
            
            monthly_data = df.groupby(df.index.to_period('M')).agg({
                TEMP_COL: ['mean', 'min', 'max'],
                RAIN_COL: 'sum',
                HUMIDITY_COL: 'mean'
            }).reset_index()
            monthly_data['Month'] = monthly_data[DATE_COL].astype(str)
            
            print("\n--- Monthly Aggregate Statistics ---")
            print(monthly_data.to_string(index=False))
            
            def get_season(month):
                if month in [12, 1, 2]: return 'Winter'
                elif month in [3, 4, 5]: return 'Spring'
                elif month in [6, 7, 8]: return 'Summer'
                else: return 'Autumn'
            
            df['SEASON'] = df.index.month.map(get_season)
            
            seasonal_summary = df.groupby('SEASON').agg({
                TEMP_COL: ['mean', 'max'],
                RAIN_COL: 'sum',
                HUMIDITY_COL: 'mean'
            }).reindex(['Spring', 'Summer', 'Autumn', 'Winter'])

            print("\n--- Seasonal Weather Summary ---")
            print(seasonal_summary.to_string())
            
            print("\n--- Visualization ---")
            
            os.makedirs('plots', exist_ok=True)

            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df[TEMP_COL], label='Daily Temperature', color='#E63946', linewidth=1.5)
            plt.title(f'Daily Temperature Trend ({df.index.min().year})', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel(f'Temperature ({TEMP_COL.split("_")[0]})', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig('plots/1_daily_temperature_line_chart.png')
            print("Saved '1_daily_temperature_line_chart.png'")
            plt.close()

            plt.figure(figsize=(10, 6))
            monthly_rainfall = monthly_data[RAIN_COL, 'sum']
            months = monthly_data['Month']
            plt.bar(months, monthly_rainfall, color='#457B9D')
            plt.title('Monthly Rainfall Totals', fontsize=16)
            plt.xlabel('Month', fontsize=12)
            plt.ylabel(f'Total Rainfall ({RAIN_COL.split("_")[0]})', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig('plots/2_monthly_rainfall_bar_chart.png')
            print("Saved '2_monthly_rainfall_bar_chart.png'")
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.scatter(df[TEMP_COL], df[HUMIDITY_COL], alpha=0.6, color='#1D3557', edgecolors='w', s=50)
            plt.title('Relationship Between Temperature and Humidity', fontsize=16)
            plt.xlabel(f'Temperature ({TEMP_COL.split("_")[0]})', fontsize=12)
            plt.ylabel(f'Humidity ({HUMIDITY_COL.split("_")[0]})', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig('plots/3_temp_humidity_scatter.png')
            print("Saved '3_temp_humidity_scatter.png'")
            plt.close()
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            color1 = '#E63946'
            ax1.set_xlabel('Date')
            ax1.set_ylabel(f'Temperature ({TEMP_COL.split("_")[0]})', color=color1)
            ax1.plot(df.index, df[TEMP_COL], color=color1, linewidth=1.5)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(axis='y', linestyle=':', alpha=0.4)
            
            ax2 = ax1.twinx()
            color2 = '#457B9D'
            ax2.set_ylabel(f'Rainfall ({RAIN_COL.split("_")[0]})', color=color2)
            ax2.bar(df.index, df[RAIN_COL], color=color2, alpha=0.3, width=1.0)
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_ylim(0, df[RAIN_COL].max() * 1.2)
            
            fig.suptitle('Daily Temperature and Rainfall (Combined View)', fontsize=16)
            fig.tight_layout()
            plt.savefig('plots/4_combined_temp_rainfall.png')
            print("Saved '4_combined_temp_rainfall.png'")
            plt.close()
            
            print("\n--- Exporting Data ---")
            df.to_csv(r"/Users/tanvichoudhary/Documents/Assignment4/cleaned_weather.csv")
            print(f"Cleaned data exported to '/Users/tanvichoudhary/Documents/Assignment4/cleaned_weather.csv'.")
            
            report_content = f"""
# Weather Data Analysis Report

## Summary of Findings

This report summarizes the analysis of weather data from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}.

### Key Statistical Overview
- **Overall Mean Temperature:** {df[TEMP_COL].mean():.1f}°C
- **Hottest Day:** {df[TEMP_COL].max():.1f}°C on {df[TEMP_COL].idxmax().strftime('%Y-%m-%d')}
- **Total Rainfall:** {df[RAIN_COL].sum():.1f} mm

### Seasonal Trends
{seasonal_summary.to_string()}

**Summer:** Characterized by the highest average temperatures.
**Winter:** The driest and coldest season.
"""
            with open('summary_report_auto.md', 'w') as f:
                f.write(report_content)
                
            print("Auto-generated summary report saved to 'summary_report_auto.md'.")
            
            print("\n--- EXECUTION COMPLETE ---")
            print("Check the 'plots/' directory for PNGs.")