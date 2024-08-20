import os
import pandas as pd

# Definiere die Verzeichnisse, die durchsucht werden sollen
directories = ['training', 'test', 'schlecht', 'neu']

# Funktion zum Berechnen der Dauer einer Datenaufnahme
def calculate_duration(file_path):
    try:
        data = pd.read_csv(file_path)
        start_time = data['gyroTimestamp_sinceReboot(s)'].iloc[0]
        end_time = data['gyroTimestamp_sinceReboot(s)'].iloc[-1]
        
        duration = end_time - start_time
        return duration
    except Exception as e:
        print(f"Fehler beim Verarbeiten der Datei {file_path}: {e}")
        return 0

# Gesamtdauer initialisieren
total_duration = 0
total_len = 0
count = 0
# Durch die Verzeichnisse iterieren
for directory in directories:
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                duration= calculate_duration(file_path)
                count += 1
                total_duration += duration
                print(f"Dauer der Datei {file_path}: {duration:.2f} Sekunden.")

print(f"Gesamtdauer aller Datenaufnahmen: {total_duration:.2f} Sekunden, Count: {count}")
