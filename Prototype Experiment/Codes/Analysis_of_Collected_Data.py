import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

def calculate_statistics(csv_file):
    # Load the dataset
    df = pd.read_csv(csv_file)
    # İlk 2 satırdaki hatalı/0 verileri yoksaymak için
    df = df.iloc[2:]

    # Mapping of the image's attributes to the CSV columns
    attribute_mapping = {
        'microphone': 'Sound_Level',
        'gas': 'Gas_Value',
        'wind_direction': 'Wind_Direction',
        'line_detect_status': 'Line_Sensor',
        'vibration': 'Vibration_Value',
        'wind_speed': 'Wind_Speed_kmh',
        'ambient_temperature': 'Temperature_1',
        'gearbox_temperature': 'Temperature_2',
        'nacelle_temperature': 'Temperature_3',
        'voltage': 'Turbine_Voltage',
        'current (mA)': 'Turbine_Current',
        'power (mW)': 'Turbine_Power'
    }

    # Create a list to store the statistics
    stats = []

    for attr, col in attribute_mapping.items():
        if col in df.columns:
            data = df[col]
            
            avg = data.mean()
            std = data.std()
            coef_var = std / avg if avg != 0 else np.nan
            min_val = data.min()
            max_val = data.max()
            val_range = max_val - min_val
            iqr = data.quantile(0.75) - data.quantile(0.25)
            
            # Calculate MAPE
            # In descriptive statistics without a reference value, MAPE is often calculated 
            # as the Mean Absolute Percentage Deviation from the mean.
            if avg != 0:
                mape = np.mean(np.abs((data - avg) / avg)) * 100
            else:
                mape = np.nan
                
            stats.append({
                'Attributes': attr,
                'Average': round(avg, 2),
                'STD': round(std, 2),
                'Coef. Variance': round(coef_var, 2),
                'Min': round(min_val, 2),
                'Max': round(max_val, 2),
                'Range': round(val_range, 2),
                'IQR': round(iqr, 2),
                'MAPE': round(mape, 2)
            })

    # Create a DataFrame for the results
    stats_df = pd.DataFrame(stats)

    # Print the results in a formatted table
    print("Table 1. Basic Statistics of Collected Data from the Prototype.")
    print("-" * 85)
    print(stats_df.to_string(index=False))
    print("-" * 85)

    # Save to a new Excel file
    output_filename = 'basic_statistics_output1.xlsx'
    try:
        stats_df.to_excel(output_filename, index=False)
        print(f"\nİstatistikler '{output_filename}' dosyasına fotoğraftaki sırayla kaydedildi.")
    except ModuleNotFoundError:
        print(f"\nExcel'e kaydetmek için 'openpyxl' kütüphanesi eksik. Lütfen 'pip install openpyxl' komutunu çalıştırın.")

if __name__ == "__main__":
    calculate_statistics('combined_sensor_data_1.csv')
