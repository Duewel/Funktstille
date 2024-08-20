# Wurde benötigt um anfängliches Problem ausfindig. Das Programm schien zum anfänglichen
# Zeitpunkt sehr rechenintensiv und lieferte kein Ergbenis, 
# So konnte die rechenintensive Funktion ausfindig gemacht werden und optimiert werden. 

import numpy as np
import pandas as pd
import os
from geopy.distance import geodesic
from scipy.signal import find_peaks, butter, filtfilt
import time

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
    result_df = winkel_angleichen(result_df)
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

def winkel_angleichen(result_df):
    future_angle = result_df["future_angle"]
    for i, angle in enumerate(future_angle):
        if pd.notna(angle):  # Überprüfen, ob der Winkel nicht NaN ist
            rest = angle % 90
            if rest > 30:
                result_df.at[i, "future_angle"] = (angle // 90 + 1) * 90
            else: 
                result_df.at[i, "future_angle"] =  (angle // 90 ) * 90
    return result_df

# Eckpunkte des Parcours
allLat = [53.2295966, 53.2295037, 53.2294835, 53.2294804, 53.2296053, 53.2298517, 53.2298470, 53.2297627, 53.2295982, 53.2294744, 53.2293866, 53.2293154, 53.2293119, 53.2294001, 53.2294155, 53.2294238, 53.2293211, 53.2296167, 53.2297696]
allLong = [10.4053239, 10.4053357, 10.4050846, 10.4048623, 10.4048372, 10.4047826, 10.4046903, 10.4046940, 10.4047287, 10.4047508, 10.4047560, 10.4048099, 10.4048889, 10.4048764, 10.4051001, 10.4053372, 10.4050829, 10.4050307, 10.4046194]

# Verzeichnis der Trainingsdateien
training_dir = "Training"

# Zähler für korrekte und falsche Klassifizierungen
correct_classifications = 0
incorrect_classifications = 0

# Zeitmessung für verschiedene Abschnitte
timing_data = {
    "read_csv": 0,
    "initial_gps": 0,
    "angle_computation": 0,
    "significant_changes": 0,
    "lowpass_filtering": 0,
    "step_detection": 0,
    "future_angles": 0,
    "geodesic_computation": 0,
    "nearest_point": 0
}

# Alle CSV-Dateien im Verzeichnis durchlaufen
for file_name in os.listdir(training_dir):
    if file_name.endswith(".csv"):
        start_time = time.time()
        file_path = os.path.join(training_dir, file_name)
        data = pd.read_csv(file_path)
        timing_data["read_csv"] += time.time() - start_time

        start_time = time.time()
        # Initiale GPS-Positionen
        initial_position = (data["locationLatitude(WGS84)"].iloc[0], data["locationLongitude(WGS84)"].iloc[0])
        final_position = (data["locationLatitude(WGS84)"].iloc[-1], data["locationLongitude(WGS84)"].iloc[-1])
        initial_heading = data['locationTrueHeading(°)'].iloc[0]
        timing_data["initial_gps"] += time.time() - start_time

        start_time = time.time()
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
        timing_data["angle_computation"] += time.time() - start_time

        start_time = time.time()
        significant_changes = data[np.abs(data["angle_change_per_sec"]) > 80]
        significant_times = significant_changes["time"].tolist()
        filtered_times = []
        threshold = 1  # 1 Sekunde Unterschied

        for i in range(1, len(significant_times)):
            if significant_times[i] - significant_times[i-1] > threshold:
                filtered_times.append(significant_times[i-1])

        if significant_times:
            filtered_times.append(significant_times[-1])
        timing_data["significant_changes"] += time.time() - start_time

        start_time = time.time()
        # Filtern der Beschleunigungsdaten
        fs = 1 / data["accelerometerTimestamp_sinceReboot(s)"].diff().mean()  # Abtastrate berechnet aus Zeitdifferenzen
        cutoff = 5.0  # Cutoff-Frequenz in Hz
        data["accelX_filtered"] = lowpass_filter(data['accelerometerAccelerationX(G)'], cutoff, fs)
        data["accelY_filtered"] = lowpass_filter(data['accelerometerAccelerationY(G)'], cutoff, fs)
        data["accelZ_filtered"] = lowpass_filter(data['accelerometerAccelerationZ(G)'], cutoff, fs)
        timing_data["lowpass_filtering"] += time.time() - start_time

        start_time = time.time()
        data["accel_magnitude"] = np.sqrt(data["accelX_filtered"]**2 + data["accelY_filtered"]**2 + data["accelZ_filtered"]**2)

        peaks, _ = find_peaks(data["accel_magnitude"], height=1.1, distance=fs/2)
        num_steps = len(peaks)
        total_time = data["accelerometerTimestamp_sinceReboot(s)"].iloc[-1] - data["accelerometerTimestamp_sinceReboot(s)"].iloc[0]
        step_frequency = num_steps / total_time
        timing_data["step_detection"] += time.time() - start_time

      
        average_stride_length = find_step_length(filtered_times, step_frequency)
        fixed_speed = step_frequency * average_stride_length
        
        # Annahme: Sie möchten um etwa 2 Sekunden verschieben
        time_shift_seconds = 2

        # Zeitindex und Abtastfrequenz (falls vorhanden)
        time_index = data["time"]  # Zeitindex
        sampling_rate = time_index.diff().mean()  # Annahme: Abtastfrequenz aus Zeitdifferenzen berechnen

        # Berechnung der Anzahl von Zeilen für den Zeitversatz
        rows_to_shift = int(time_shift_seconds / sampling_rate)

        # Verschieben der Daten um die berechnete Anzahl von Zeilen
        data["future_angle"] = data["current_angle"].shift(-rows_to_shift).fillna(np.nan)
            

        start_time = time.time()
        results2 = calculate_future_angles(data, filtered_times)
        timing_data["geodesic_computation"] += time.time() - start_time

        start_time = time.time()
        positions = [(allLat[0], allLong[0])]
        for i in range(1, len(results2)):
            delta_time = results2['delta_time'][i]
            future_direction = results2['future_angle'][i-1]
        
            if not np.isnan(future_direction) and np.isfinite(future_direction):
                distance = fixed_speed * delta_time
                direction_rad = np.deg2rad(future_direction)
                new_position = geodesic(meters=distance).destination(positions[-1], np.rad2deg(direction_rad))
                # Debug 
                #new_positions.append(new_position)
                corrected_pos = find_nearest_point(new_position.latitude, new_position.longitude, allLat, allLong)
                positions.append(corrected_pos)
        timing_data["nearest_point"] += time.time() - start_time

        start_time = time.time()
        geschätzt_endposition_berechnet = positions[-1]
        echte_endposition_korrigiert = find_nearest_point(final_position[0], final_position[1], allLat, allLong)

        if echte_endposition_korrigiert == geschätzt_endposition_berechnet:
            correct_classifications += 1
        else:
            incorrect_classifications += 1
            print(f"Falsch klassifiziert: {file_name}")

# Ausgabe der Zeitmessungen
print("Zeitmessungen:")
total_time = sum(timing_data.values())
for key, value in timing_data.items():
    print(f"{key}: {value} Sekunden ({value/total_time * 100:.2f}% der Gesamtzeit)")

# Ausgabe der Ergebnisse
print(f"Korrekte Klassifizierungen: {correct_classifications}")
print(f"Falsche Klassifizierungen: {incorrect_classifications}")

