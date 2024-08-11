"""

import numpy as np
import pandas as pd
import os

def process_file(file_path):
    # Daten initialisieren
    data = pd.read_csv(file_path)
    
    # Initiale Werte
    initial_heading = data['locationTrueHeading(°)'].iloc[0]
    data["time"] = data['gyroTimestamp_sinceReboot(s)'] - data['gyroTimestamp_sinceReboot(s)'].iloc[0]
    data["timedelta"] = data["time"].diff().fillna(0)
    data["period_angle"] = data['gyroRotationZ(rad/s)'] * data["timedelta"]
    data["cumulated_angle_rad"] = np.cumsum(data["period_angle"])
    data["angle_degree"] = data["cumulated_angle_rad"] / np.pi * 180
    data["current_angle"] = initial_heading - data["angle_degree"]

    data["future_time"] = data["time"] + 2
    data["future_angle"] = np.nan

    data["angle_change_per_sec"] = data["angle_degree"].diff().fillna(0) / data["timedelta"]
    significant_changes = data[np.abs(data["angle_change_per_sec"]) > 80]
    significant_times = significant_changes["time"].tolist()

    filtered_times = []
    threshold = 1  # 1 Sekunde Unterschied

    for i in range(1, len(significant_times)):
        if significant_times[i] - significant_times[i-1] > threshold:
            filtered_times.append(significant_times[i-1])

    # Hinzufügen des letzten Elements
    if significant_times:
        filtered_times.append(significant_times[-1])

    # Aktualisierung der zukünftigen Winkel
    for index, row in data.iterrows():
        future_time = row["future_time"]
        future_row = data[data["time"] >= future_time].iloc[0] if not data[data["time"] >= future_time].empty else None
        if future_row is not None:
            data.at[index, "future_angle"] = future_row["current_angle"]

    def calculate_future_angles(data, filtered_times):
        future_angles = []
        delta_times = []

        # Initialer Zeitpunkt 0 hinzufügen
        filtered_times.insert(0, 0)

        # Endzeitpunkt hinzufügen
        filtered_times.append(data["time"].max())

        for time in filtered_times:
            # Holen des zukünftigen Winkels aus der Spalte "future_angle" zur gegebenen Zeit
            future_angle = data.loc[data["time"] == time, "future_angle"].values
            if len(future_angle) > 0:
                future_angle = future_angle[0]
            else:
                future_angle = np.nan
            future_angles.append(future_angle)

        # Berechnen der delta_times als Unterschied zwischen aufeinanderfolgenden filtered_times
        delta_times = [0] + [filtered_times[i] - filtered_times[i - 1] for i in range(1, len(filtered_times))]

        result_df = pd.DataFrame({
            "filtered_time": filtered_times,
            "future_angle": future_angles,
            "delta_time": delta_times
        })

     
        return result_df
    
    

    # Berechnung von result2
    results2 = calculate_future_angles(data, filtered_times)

    # Ausgabe von result2
    print(f"Results for {file_path}:")
    print(results2)

# Ordnerpfad setzen
folder_path = "training"

# Durch alle CSV-Dateien im Ordner gehen
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        process_file(file_path)
"""
import numpy as np
import pandas as pd
import os
from geopy.distance import geodesic
from scipy.signal import find_peaks, butter, filtfilt

# Funktionen zum Filtern und Berechnen der Schrittlänge
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def find_step_length(filtered_times, step_frequency):
    if filtered_times[0] > 12:
        x = (11.16 + 7.58) / (step_frequency * filtered_times[0])
    else:
        x = 11.16 / (step_frequency * filtered_times[0])
    return x

def calculate_future_angles(data, filtered_times):
    future_angles = []
    delta_times = []
    filtered_times.insert(0, 0)
    filtered_times.append(data["time"].max())
    for time in filtered_times:
        future_angle = data.loc[data["time"] == time, "future_angle"].values
        if len(future_angle) > 0:
            future_angle = future_angle[0]
        else:
            future_angle = np.nan
        future_angles.append(future_angle)
    delta_times = [0] + [filtered_times[i] - filtered_times[i - 1] for i in range(1, len(filtered_times))]
    result_df = pd.DataFrame({"filtered_time": filtered_times, "future_angle": future_angles, "delta_time": delta_times})
    return result_df

def find_nearest_point(lat, lon, allLat, allLong):
    min_distance = float('inf')
    nearest_point = None
    for i in range(len(allLat)):
        point = (allLat[i], allLong[i])
        distance = geodesic((lat, lon), point).meters
        if distance < min_distance:
            min_distance = distance
            nearest_point = point
    return nearest_point

# Eckpunkte des Parcours
allLat = [53.2295966, 53.2295037, 53.2294835, 53.2294804, 53.2296053, 53.2298517, 53.2298470, 53.2297627, 53.2295982, 53.2294744, 53.2293866, 53.2293154, 53.2293119, 53.2294001, 53.2294155, 53.2294238, 53.2293211, 53.2296167, 53.2297696]
allLong = [10.4053239, 10.4053357, 10.4050846, 10.4048623, 10.4048372, 10.4047826, 10.4046903, 10.4046940, 10.4047287, 10.4047508, 10.4047560, 10.4048099, 10.4048889, 10.4048764, 10.4051001, 10.4053372, 10.4050829, 10.4050307, 10.4046194]

# Verzeichnis der Trainingsdateien
training_dir = "Training"

# Zähler für korrekte und falsche Klassifizierungen
correct_classifications = 0
incorrect_classifications = 0

# Alle CSV-Dateien im Verzeichnis durchlaufen
for file_name in os.listdir(training_dir):
    if file_name.endswith(".csv"):
        file_path = os.path.join(training_dir, file_name)
        data = pd.read_csv(file_path)

        # Initiale GPS-Positionen
        initial_position = (data["locationLatitude(WGS84)"].iloc[0], data["locationLongitude(WGS84)"].iloc[0])
        final_position = (data["locationLatitude(WGS84)"].iloc[-1], data["locationLongitude(WGS84)"].iloc[-1])
        initial_heading = data['locationTrueHeading(°)'].iloc[0]

        # Winkelberechnung
        data["time"] = data['gyroTimestamp_sinceReboot(s)'] - data['gyroTimestamp_sinceReboot(s)'].iloc[0]
        data["timedelta"] = data["time"].diff().fillna(0)
        data["period_angle"] = data['gyroRotationZ(rad/s)'] * data["timedelta"]
        data["cumulated_angle_rad"] = np.cumsum(data["period_angle"])
        data["angle_degree"] = data["cumulated_angle_rad"] / np.pi * 180
        data["current_angle"] = initial_heading - data["angle_degree"]

        data["future_time"] = data["time"] + 2
        data["future_angle"] = np.nan
        data["angle_change_per_sec"] = data["angle_degree"].diff().fillna(0) / data["timedelta"]

        significant_changes = data[np.abs(data["angle_change_per_sec"]) > 80]
        significant_times = significant_changes["time"].tolist()
        filtered_times = []
        threshold = 1  # 1 Sekunde Unterschied

        for i in range(1, len(significant_times)):
            if significant_times[i] - significant_times[i-1] > threshold:
                filtered_times.append(significant_times[i-1])

        if significant_times:
            filtered_times.append(significant_times[-1])

        # Filtern der Beschleunigungsdaten
        fs = 1 / data["accelerometerTimestamp_sinceReboot(s)"].diff().mean()  # Abtastrate berechnet aus Zeitdifferenzen
        cutoff = 5.0  # Cutoff-Frequenz in Hz
        data["accelX_filtered"] = lowpass_filter(data['accelerometerAccelerationX(G)'], cutoff, fs)
        data["accelY_filtered"] = lowpass_filter(data['accelerometerAccelerationY(G)'], cutoff, fs)
        data["accelZ_filtered"] = lowpass_filter(data['accelerometerAccelerationZ(G)'], cutoff, fs)

        data["accel_magnitude"] = np.sqrt(data["accelX_filtered"]**2 + data["accelY_filtered"]**2 + data["accelZ_filtered"]**2)

        peaks, _ = find_peaks(data["accel_magnitude"], height=1.1, distance=fs/2)
        num_steps = len(peaks)
        total_time = data["accelerometerTimestamp_sinceReboot(s)"].iloc[-1] - data["accelerometerTimestamp_sinceReboot(s)"].iloc[0]
        step_frequency = num_steps / total_time

        average_stride_length = find_step_length(filtered_times, step_frequency)
        fixed_speed = step_frequency * average_stride_length

        for index, row in data.iterrows():
            future_time = row["future_time"]
            future_row = data[data["time"] >= future_time].iloc[0] if not data[data["time"] >= future_time].empty else None
            if future_row is not None:
                data.at[index, "future_angle"] = future_row["current_angle"]

        positions = [initial_position]
        current_position = initial_position
        current_angle = initial_heading
        angles = []

       
        results2 = calculate_future_angles(data, filtered_times)

        positions = [(allLat[0], allLong[0])]
        for i in range(1, len(results2)):
            delta_time = results2['delta_time'][i]
            future_direction = results2['future_angle'][i-1]
            if not np.isnan(future_direction) and np.isfinite(future_direction):
                distance = fixed_speed * delta_time
                direction_rad = np.deg2rad(future_direction)
                new_position = geodesic(meters=distance).destination(positions[-1], np.rad2deg(direction_rad))
                corrected_pos = find_nearest_point(new_position.latitude, new_position.longitude, allLat, allLong)
                positions.append(corrected_pos)

        geschätzt_endposition_berechnet = positions[-1]
        echte_endposition_korrigiert = find_nearest_point(final_position[0], final_position[1], allLat, allLong)

        if echte_endposition_korrigiert == geschätzt_endposition_berechnet:
            correct_classifications += 1
        else:
            incorrect_classifications += 1
            print(f"Falsch klassifiziert: {file_name}")

# Ausgabe der Ergebnisse
print(f"Korrekte Klassifizierungen: {correct_classifications}")
print(f"Falsche Klassifizierungen: {incorrect_classifications}")
