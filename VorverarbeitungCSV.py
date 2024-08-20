import os
import glob
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from scipy.signal import find_peaks, butter, filtfilt

allLat = [53.2295966, 53.2295037, 53.2294835, 53.2294804, 53.2296053, 53.2298517, 53.2298470, 53.2297627, 53.2295982, 53.2294744, 53.2293866, 53.2293154, 53.2293119, 53.2294001, 53.2294155, 53.2294238, 53.2293211, 53.2296167, 53.2297696]
allLong = [10.4053239, 10.4053357, 10.4050846, 10.4048623, 10.4048372, 10.4047826, 10.4046903, 10.4046940, 10.4047287, 10.4047508, 10.4047560, 10.4048099, 10.4048889, 10.4048764, 10.4051001, 10.4053372, 10.4050829, 10.4050307, 10.4046194]
name_without_ext_list =[]

# Funktionen zur Berechnung und Verarbeitung
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
    if(filtered_times[0] > 12):
        x = (11.16 + 7.58)/(step_frequency* filtered_times[0])
        return x
    else: 
        x = 11.16/(step_frequency* filtered_times[0])
        return x
    

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

def calculate_future_angles(data, filtered_times):
    future_angles = []
    delta_times = []
    gps_latitudes = []
    gps_longitudes = []
    corrected_latitudes = []
    corrected_longitudes = []

    # Initialer Zeitpunkt 0 hinzufügen
    filtered_times.insert(0, 0)

    # Endzeitpunkt hinzufügen
    filtered_times.append(data["time"].max())

    for idx, time in enumerate(filtered_times):
        # Holen des zukünftigen Winkels aus der Spalte "future_angle" zur gegebenen Zeit
        future_angle = data.loc[data["time"] == time, "future_angle"].values
        if len(future_angle) > 0:
            future_angle = future_angle[0]
        else:
            future_angle = np.nan
        future_angles.append(future_angle)

        # Holen der GPS-Daten (Latitude und Longitude) zum jeweiligen Zeitpunkt
        gps_lat = data.loc[data["time"] == time, "locationLatitude(WGS84)"].values
        gps_lon = data.loc[data["time"] == time, "locationLongitude(WGS84)"].values
        if len(gps_lat) > 0 and len(gps_lon) > 0:
            gps_latitudes.append(gps_lat[0])
            gps_longitudes.append(gps_lon[0])
        else:
            gps_latitudes.append(np.nan)
            gps_longitudes.append(np.nan)

        # Korrigieren der GPS-Daten auf den nächsten Eckpunkt
        if idx == 0:
            # Der erste Punkt wird auf initial_lat und initial_lon korrigiert
            corrected_latitudes.append(allLat[0])
            corrected_longitudes.append(allLong[0])
        else:
            if not np.isnan(gps_latitudes[-1]) and not np.isnan(gps_longitudes[-1]):
                corrected_pos = find_nearest_point(gps_latitudes[-1], gps_longitudes[-1], allLat, allLong)
                corrected_latitudes.append(corrected_pos[0])
                corrected_longitudes.append(corrected_pos[1])
            else:
                corrected_latitudes.append(np.nan)
                corrected_longitudes.append(np.nan)

    # Berechnen der delta_times als Unterschied zwischen aufeinanderfolgenden filtered_times
    delta_times = [0] + [filtered_times[i] - filtered_times[i - 1] for i in range(1, len(filtered_times))]

    result_df = pd.DataFrame({
        "filtered_time": filtered_times,
        "future_angle": future_angles,
        "delta_time": delta_times,
        "gps_latitude": gps_latitudes,
        "gps_longitude": gps_longitudes,
        "corrected_latitude": corrected_latitudes,
        "corrected_longitude": corrected_longitudes
    })
    winkel_angleichen(result_df)
    return result_df

# Pfad zum Ordner mit Trainingsdaten
training_folder = "new"

# Liste der CSV-Dateien im Ordner "training"
csv_files = glob.glob(os.path.join(training_folder, "*.csv"))


# Liste zum Speichern der Ergebnistabellen für jede Datei
all_results = []

# Iteriere über jede CSV-Datei im Ordner
for csv_file in csv_files:
    try:
        

        # Daten aus der CSV-Datei laden
        data = pd.read_csv(csv_file)
        # Initiale GPS-Position (ersetzen Sie dies durch Ihre tatsächliche Startposition)
        initial_position = (data["locationLatitude(WGS84)"].iloc[0], data["locationLongitude(WGS84)"].iloc[0])
        final_position = (data["locationLatitude(WGS84)"].iloc[-1], data["locationLongitude(WGS84)"].iloc[-1])

        initial_heading = data['locationTrueHeading(°)'].iloc[0]

        # Berechnung Winkel
        data["time"] = data['gyroTimestamp_sinceReboot(s)'] - data['gyroTimestamp_sinceReboot(s)'].iloc[0]
        data["timedelta"] = data["time"].diff().fillna(0)
        data["period_angle"] = data['gyroRotationZ(rad/s)'] * data["timedelta"]
        data["cumulated_angle_rad"] = np.cumsum(data["period_angle"])
        data["angle_degree"] = data["cumulated_angle_rad"] / np.pi * 180
        data["current_angle"] = initial_heading - data["angle_degree"]

        # Berechnung zukünftiger Winkel
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

        fs = 1 / data["accelerometerTimestamp_sinceReboot(s)"].diff().mean()  # Abtastrate berechnet aus Zeitdifferenzen
        cutoff = 5.0  # Cutoff-Frequenz in Hz, anpassen je nach Bedarf

        # Anwendung des Low-Pass-Filters auf die Beschleunigungsdaten
        data["accelX_filtered"] = lowpass_filter(data['accelerometerAccelerationX(G)'], cutoff, fs)
        data["accelY_filtered"] = lowpass_filter(data['accelerometerAccelerationY(G)'], cutoff, fs)
        data["accelZ_filtered"] = lowpass_filter(data['accelerometerAccelerationZ(G)'], cutoff, fs)

        # Berechnung der Magnitude des Beschleunigungsvektors
        data["accel_magnitude"] = np.sqrt(data["accelX_filtered"]**2 + data["accelY_filtered"]**2 + data["accelZ_filtered"]**2)

        # Schritt-Erkennung durch Identifikation von Spitzen
        peaks, _ = find_peaks(data["accel_magnitude"], height=1.1, distance=fs/2)  # Passen Sie die Parameter height und distance an

        # Anzahl der Schritte
        num_steps = len(peaks)

        # Berechnung der Gesamtzeitdauer des Datensatzes
        total_time = data["accelerometerTimestamp_sinceReboot(s)"].iloc[-1] - data["accelerometerTimestamp_sinceReboot(s)"].iloc[0]

        # Schrittfrequenz berechnen
        step_frequency = num_steps / total_time  # Schritte pro Sekunde

        # Schrittlänge schätzen (beispielhaft, anpassen je nach Bedarf)
        average_stride_length = find_step_length(filtered_times, step_frequency) # Durchschnittliche Schrittlänge in Metern

        # Geschwindigkeit berechnen
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

        # Positionen berechnen
        positions = [initial_position]
        current_position = initial_position
        current_angle = initial_heading

        angles = []
        base_name = os.path.basename(csv_file)
        name_without_ext_list.append(os.path.splitext(base_name)[0])
        results2 = calculate_future_angles(data, filtered_times)

        all_results.append(results2)

        # Berechnung Route über einzelne Winkel 
        #positions = [initial_position] 
        positions = [(allLat[0], allLong[0])] 
        # positions = [initial_position]  # Startposition
        for i in range(1, len(results2)):
            delta_time = results2['delta_time'][i]
            future_direction = results2['future_angle'][i-1]
            
            # Berechnung der zurückgelegten Strecke in diesem Zeitintervall in Metern
            distance = fixed_speed * delta_time
            
            # Umwandlung der Richtung von Grad in Bogenmaß
            direction_rad = np.deg2rad(future_direction)
            
            # Berechnung der Verschiebung in Metern
            delta_lat = distance * np.cos(direction_rad)
            delta_lon = distance * np.sin(direction_rad)
            
            # Berechnung der neuen Position unter Verwendung der geopy-Bibliothek
            prev_lat, prev_lon = positions[-1]
            new_position = geodesic(meters=distance).destination((prev_lat, prev_lon), np.rad2deg(direction_rad))

            corrected_pos = find_nearest_point(new_position.latitude, new_position.longitude, allLat, allLong)
            positions.append(corrected_pos)

        #Vergleich echt und geschätzt
        geschätzt_endposition_berechnet = positions[-1]
        echte_endposition_korrigiert = find_nearest_point(final_position[0], final_position[1], allLat, allLong)
        
        if echte_endposition_korrigiert == positions[-1]:
            print("Klassifizierung RICHTIG")
        else: 
            print("Klassifizierung FALSCH")
            
        # Extrahieren von Breiten- und Längengraden für die Darstellung des Pfads
        path_lats, path_lons = zip(*positions)

    except Exception as e:
        print(f"Fehler beim Verarbeiten der Datei {csv_file}: {str(e)}")

# Speichern der Ergebnistabellen als CSV-Dateien
output_folder = "test"
os.makedirs(output_folder, exist_ok=True)

for idx, result_df in enumerate(all_results):
    output_file = os.path.join(output_folder, f"result_{idx}_{name_without_ext_list[idx]}.csv")
    result_df.to_csv(output_file, index=False)
    print(f"Ergebnisse für Datei {idx} gespeichert in {output_file}")
