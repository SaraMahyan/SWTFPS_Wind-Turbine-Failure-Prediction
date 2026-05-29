import serial
import time
import csv
import threading
from datetime import datetime

# --- USER SETTINGS ---
ESP1_PORT = 'COM7'  # Port where Bitirme1.ino is connected
ESP2_PORT = 'COM3'  # Port where Bitirme2.ino is connected
BAUD_RATE = 9600
CSV_FILE_NAME = 'combined_sensor_data.csv'

# Global dictionaries to hold sensor data
esp1_data = {
    'millis': 0, 'wind_dir': 0, 'wind_speed': 0.0, 't1': 0.0, 't2': 0.0, 't3': 0.0,
    'voltage': 0.0, 'current': 0.0, 'power': 0.0, 'gas': 0, 'line': 0
}

esp2_data = {
    'millis': 0, 'sound': 0, 'vibration': 0
}

# Data reading function (Will run in a separate background thread for each port)
def read_serial_port(port_name, esp_number):
    try:
        ser = serial.Serial(port_name, BAUD_RATE, timeout=1)
        print(f"[SUCCESS] Connected to port {port_name} (ESP {esp_number}).")
    except Exception as e:
        print(f"[ERROR] Could not connect to port {port_name}. Please check the port: {e}")
        return

    while True:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # Only extract lines starting with the word "CSV,"
                if line.startswith("CSV,"):
                    parts = line.split(',')
                    
                    if esp_number == 1 and len(parts) >= 12:
                        # Data from ESP1: CSV,millis,dir,speed,t1,t2,t3,v,i,p,gas,line
                        esp1_data['millis'] = parts[1]
                        esp1_data['wind_dir'] = parts[2]
                        esp1_data['wind_speed'] = parts[3]
                        esp1_data['t1'] = parts[4]
                        esp1_data['t2'] = parts[5]
                        esp1_data['t3'] = parts[6]
                        esp1_data['voltage'] = parts[7]
                        esp1_data['current'] = parts[8]
                        esp1_data['power'] = parts[9]
                        esp1_data['gas'] = parts[10]
                        esp1_data['line'] = parts[11]
                        
                    elif esp_number == 2 and len(parts) >= 4:
                        # Data from ESP2: CSV,millis,sound,vib
                        esp2_data['millis'] = parts[1]
                        esp2_data['sound'] = parts[2]
                        esp2_data['vibration'] = parts[3]
                        
        except Exception as e:
            time.sleep(1)

def main():
    # 1. Create the CSV file and write the header row (column names)
    headers = [
        "PC_Datetime", "ESP1_Millis", "ESP2_Millis",
        "Wind_Direction", "Wind_Speed_kmh", "Temperature_1", "Temperature_2", "Temperature_3",
        "Turbine_Voltage", "Turbine_Current", "Turbine_Power", "Gas_Value", "Line_Sensor",
        "Sound_Level", "Vibration_Value"
    ]
    
    with open(CSV_FILE_NAME, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        print(f"[{CSV_FILE_NAME}] file created successfully.\n")

    # 2. Start the threads (background tasks) reading the ports
    t1 = threading.Thread(target=read_serial_port, args=(ESP1_PORT, 1), daemon=True)
    t2 = threading.Thread(target=read_serial_port, args=(ESP2_PORT, 2), daemon=True)
    t1.start()
    t2.start()

    print("\nData collection started! You can press CTRL+C to stop.\n")

    # 3. Combine and save the latest incoming data every 1 second
    try:
        while True:
            time.sleep(1.0) # Saves every second
            
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            row = [
                current_time,
                esp1_data['millis'],
                esp2_data['millis'],
                esp1_data['wind_dir'],
                esp1_data['wind_speed'],
                esp1_data['t1'],
                esp1_data['t2'],
                esp1_data['t3'],
                esp1_data['voltage'],
                esp1_data['current'],
                esp1_data['power'],
                esp1_data['gas'],
                esp1_data['line'],
                esp2_data['sound'],
                esp2_data['vibration']
            ]

            with open(CSV_FILE_NAME, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
            print(f"[{current_time}] Data merged and saved to CSV.")

    except KeyboardInterrupt:
        print("\nProgram stopped by user. CSV file is ready.")

if __name__ == '__main__':
    main()
