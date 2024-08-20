
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from scipy.signal import find_peaks, butter, filtfilt
import pickle
import os 


# Beispiel-Daten initialisieren (ersetzen Sie dies durch Ihren tatsächlichen Pfad und Ihre Daten)
test_path = "uploads/2024-07-04_09_04_11_my_iOS_device.csv"   #"new/D- Park.csv"
data = pd.read_csv(test_path)


actual_lats = data["locationLatitude(WGS84)"]
actual_lons = data["locationLongitude(WGS84)"]
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

print(filtered_times)


# Funktion zum filtern --> Berehcnen Schrittlänge und GEschwindigkeit: 
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Funktion zum Anwenden des Low-Pass-Filters auf Daten
def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def find_step_length():
    if(filtered_times[0] > 12):
        x = (11.16 + 7.58)/(step_frequency* filtered_times[0])
        print(x)
        return x
    else: 
        x = 11.16/(step_frequency* filtered_times[0])
        print(x)
        return x
    

def winkel_angleichen(result_df):
    future_angle = result_df["future_angle"]
    for i, angle in enumerate(future_angle):
        if pd.notna(angle):  # Überprüfen, ob der Winkel nicht NaN ist
            rest = angle % 90
            if rest > 50:
                result_df.at[i, "future_angle"] = (angle // 90 + 1) * 90
            else: 
                result_df.at[i, "future_angle"] =  (angle // 90 ) * 90
    return result_df


# Sampling-Frequenz und Cutoff-Frequenz
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
print(f'Frequenz: {step_frequency}')

# Schrittlänge schätzen (beispielhaft, anpassen je nach Bedarf)
average_stride_length = find_step_length() # Durchschnittliche Schrittlänge in Metern

# Geschwindigkeit berechnen
fixed_speed = step_frequency * average_stride_length  
print(fixed_speed)


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

for index, row in data.iterrows():
    delta_time = row['timedelta']
    period_angle = row['angle_degree']

    current_angle = initial_heading - period_angle
    
    # Berechnung der Distanz, die zurückgelegt wurde
    distance = fixed_speed * delta_time

    # Berechnung der neuen Position basierend auf dem aktuellen Winkel und der Distanz
    current_position = geodesic(meters=distance).destination(current_position, current_angle)
    positions.append(current_position)
    angles.append(current_angle)

# Extrahieren der Latitude und Longitude für das Plotten
lats, lons = zip(*positions)

# Define the coordinates for each area
areas = {
    'additional_1': {'latitude': [53.2295966, 53.2295037], 'longitude': [10.4053239, 10.4053357]},
    'additional_2': {'latitude': [53.2295037, 53.2294835], 'longitude': [10.4053357, 10.4050846]},
    'additional_3': {'latitude': [53.2294835, 53.2294804], 'longitude': [10.4050846, 10.4048623]},
    'additional_4': {'latitude': [53.2294804, 53.2296053], 'longitude': [10.4048623, 10.4048372]},
    'additional_5': {'latitude': [53.2296053, 53.2298517], 'longitude': [10.4048372, 10.4047826]},
    'additional_6': {'latitude': [53.2298517, 53.2298470], 'longitude': [10.4047826, 10.4046903]},
    'additional_7': {'latitude': [53.2298470, 53.2297627], 'longitude': [10.4046903, 10.4046940]},
    'additional_8': {'latitude': [53.2297627, 53.2295982], 'longitude': [10.4046940, 10.4047287]},
    'additional_9': {'latitude': [53.2295982, 53.2294744], 'longitude': [10.4047287, 10.4047508]},
    'additional_10': {'latitude': [53.2294744, 53.2293866], 'longitude': [10.4047508, 10.4047560]},
    'additional_11': {'latitude': [53.2293866, 53.2293154], 'longitude': [10.4047560, 10.4048099]},
    'additional_12': {'latitude': [53.2293154, 53.2293119], 'longitude': [10.4048099, 10.4048889]},
    'additional_13': {'latitude': [53.2293119, 53.2294001], 'longitude': [10.4048889, 10.4048764]},
    'additional_14': {'latitude': [53.2294001, 53.2294155], 'longitude': [10.4048764, 10.4051001]},
    'additional_15': {'latitude': [53.2294155, 53.2294238], 'longitude': [10.4051001, 10.4053372]},
    'keller1': {'latitude': [53.2293119, 53.2293211], 'longitude': [10.4048889, 10.4050829]},
    'keller2': {'latitude': [53.2296053, 53.2296167], 'longitude': [10.4048372, 10.4050307]},
    'gang': {'latitude': [53.2294835, 53.2294155], 'longitude': [10.4050846, 10.4051001]},
    'park': {'latitude': [53.2297627, 53.2297696], 'longitude': [10.4046940, 10.4046194]},
    'gang2': {'latitude': [53.2294804, 53.2294001], 'longitude': [10.4048623, 10.4048764]},
    'gang3': {'latitude': [53.2294001, 53.2293866], 'longitude': [10.4048764, 10.4047560]},
    'gang4': {'latitude': [53.2294804, 53.2294744], 'longitude': [10.4048623, 10.4047508]},
    'gang5': {'latitude': [53.2296053, 53.2295982], 'longitude': [10.4048372, 10.4047287]},
    'gangstart': {'latitude': [53.2295037, 53.2294238], 'longitude': [10.4053357, 10.4053372]},
}





# Funktion zum Finden der durchlaufenen Punkte und Zeiten
def find_nearest_points_with_time(data, allLat, allLong, angles):
    # Eckpunkte des Parcours
    corners = list(zip(allLat, allLong))
    
    # Spalten für die Ergebnisse
    nearest_points = []
    distances = []
    times = []
    current_angles = []
    future_angles = []

    # Durchlauf durch die GPS-Positionen und Zeitstempel
    for corner in corners:
        min_distance = float('inf')
        nearest_time = None
        nearest_index = None

        for i, (lat, lon, time, current_angle, future_angle) in enumerate(zip(data["locationLatitude(WGS84)"], data["locationLongitude(WGS84)"], data["time"], data["current_angle"], data["future_angle"])):
            distance = geodesic((lat, lon), corner).meters
            if distance < min_distance:
                min_distance = distance
                nearest_time = time
                nearest_index = i

        if min_distance < 5:  # Schwellenwert für signifikante Nähe (z.B. 5 Meter)
            nearest_points.append(corner)
            distances.append(min_distance)
            times.append(nearest_time)
            current_angles.append(data.loc[nearest_index, "current_angle"])
            future_angles.append(data.loc[nearest_index, "future_angle"])

    # Erstellen eines DataFrames mit den Ergebnissen
    results = pd.DataFrame({
        'latitude': [point[0] for point in nearest_points],
        'longitude': [point[1] for point in nearest_points],
        'distance': distances,
        'time': times,
        'current_angle': current_angles,
        'future_Richtung': future_angles
    })

    # Sortieren nach Zeit
    results = results.sort_values(by='time').reset_index(drop=True)

    # Berechnen von delta_time
    results['delta_time'] = results['time'].diff().fillna(0)

    return results

#Debug
new_positions = []
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
    result_df = winkel_angleichen(result_df)
    print(result_df)
    return result_df

results2 =calculate_future_angles(data, filtered_times)

# Eckpunkte des Parcours
allLat = [53.2295966, 53.2295037, 53.2294835, 53.2294804, 53.2296053, 53.2298517, 53.2298470, 53.2297627, 53.2295982, 53.2294744, 53.2293866, 53.2293154, 53.2293119, 53.2294001, 53.2294155, 53.2294238, 53.2293211, 53.2296167, 53.2297696]
allLong = [10.4053239, 10.4053357, 10.4050846, 10.4048623, 10.4048372, 10.4047826, 10.4046903, 10.4046940, 10.4047287, 10.4047508, 10.4047560, 10.4048099, 10.4048889, 10.4048764, 10.4051001, 10.4053372, 10.4050829, 10.4050307, 10.4046194]

# Finden der nächstgelegenen Punkte und Zeiten
results = find_nearest_points_with_time(data, allLat, allLong, angles)

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
    #Debug 
    #new_positions.append(new_position)

    corrected_pos = find_nearest_point(new_position.latitude, new_position.longitude, allLat, allLong)
    print(f'korrigiert: {corrected_pos}, New: {new_position}')
    positions.append(corrected_pos)


# MLP Test 

def predict_trajectory(test_file, model, scaler, label_encoder):
    df = test_file
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Start with the first corrected position
    current_lat = 53.2295966
    current_long = 10.4053239
    predicted_positions = [(current_lat, current_long)]  # Include the start point in the predicted positions

    # Go through each row in the file
    for i in range(len(df) - 1):
        future_angle = df['future_angle'][i]
        delta_time = df['delta_time'][i + 1]  # Take the Delta time to the next point

        # Scale the features for the model
        features = np.array([[current_lat, current_long, future_angle, delta_time]])
        features_scaled = scaler.transform(features)

        # Predict the next position
        predicted_index = np.argmax(model.predict(features_scaled), axis=1)
        predicted_lat_long = label_encoder.inverse_transform(predicted_index)[0]
        predicted_lat, predicted_long = map(float, predicted_lat_long.split('_'))
        
        # Update the current position for the next iteration
        current_lat, current_long = predicted_lat, predicted_long
        predicted_positions.append((predicted_lat, predicted_long))
        print(f"Predicted: ({predicted_lat}, {predicted_long})")
    
    
    # The actual path and the last predicted position
    
    final_predicted_position = predicted_positions[-1]
    
    return predicted_positions, final_predicted_position

# Load the model and other resources
model = load_model('final_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


# predicted postions MLP
predicted_positions, final_predicted_position = predict_trajectory(results2, model, scaler, label_encoder)

#Vergleich echt und geschätzt
geschätzt_endposition_berechnet = positions[-1]
print(f'Endposition berechnet: {geschätzt_endposition_berechnet}')
echte_endposition_korrigiert = find_nearest_point(final_position[0], final_position[1], allLat, allLong)
lat_final_corrected, lon_final_corrected =  echte_endposition_korrigiert
print(f'Endposition echt: {echte_endposition_korrigiert}')
if echte_endposition_korrigiert == positions[-1]:
    print("Klassifizierung RICHTIG")
else: 
    print("Klassifizierung FALSCH")
    

# Extrahieren von Breiten- und Längengraden für die Darstellung des Pfads
path_lats, path_lons = zip(*positions)




# Ausgabe der Ergebnisse
print(results)

"""
# Create a figure and axis
fig, ax = plt.subplots(figsize=(15, 10))

# Plot each area as a line segment and annotate points

for area, coords in areas.items():
    latitudes = coords['latitude']
    longitudes = coords['longitude']
    ax.plot(longitudes, latitudes, color='black',linewidth=3, marker='o')
    #for lat, lon in zip(latitudes, longitudes):
    #    ax.annotate(f'({lon:.7f}, {lat:.7f})', xy=(lon, lat), textcoords='offset points', xytext=(5,5), ha='right', fontsize=8, color='grey')

# Pfad jeder einzelne Punkt
ax.plot(lons, lats, linestyle=':', color='purple', label='Prediction Path - jeder Datenpunkt',linewidth=4) # marker entfernt

# Pfad nur Eckpunkte 
ax.plot(path_lons, path_lats, marker='o', linestyle=':', color='fuchsia', label='Prediction Path - nur Eckpunkte',linewidth=4)  # Pfad in Blau zeichnen

#echter Weg 
ax.plot(actual_lons, actual_lats, linestyle=':', color='pink', label='Actual Path', linewidth=4)  # Pfad in Schwarz zeichnen

#Debug
#_lats, new_positions_lons = zip(*[(pos.latitude, pos.longitude) for pos in new_positions])
#ax.scatter(new_positions_lons, new_positions_lats, color='magenta', s=100, label='New Positions', marker='x')

# Pfad MLP
predicted_lats_MLP, predicted_longs_MLP = zip(*predicted_positions)
ax.plot(predicted_longs_MLP,predicted_lats_MLP, color= "brown", linestyle = '-', marker = 'o', label = "Predicted Path - NN", linewidth = 3)

# Highlight the initial and final positions
#ax.scatter(lons[0], lats[0], color='blue', s=100, label='Calculated Start Point', marker='x')
#ax.scatter(lons[-1], lats[-1], color='blue', s=100, label='Calculated End Point')
ax.scatter(initial_position[1], initial_position[0], color='cyan', s=100, label='Actual Start Point', marker='x')
ax.scatter(final_position[1], final_position[0], color='cyan', s=100, label='Actual End Point')

# Highlight the nearest points
# ax.scatter(results['longitude'], results['latitude'], color='cyan', s=100, label='Nearest Points', marker='x')

# Add annotations for the initial and final positions
#ax.annotate('Calculated Start', xy=(lons[0], lats[0]), xytext=(10, 10), textcoords='offset points', color='green', fontsize=12)
#ax.annotate('Calculated End', xy=(lons[-1], lats[-1]), xytext=(10, 10), textcoords='offset points', color='red', fontsize=12)
#ax.annotate('Actual Start', xy=(initial_position[1], initial_position[0]), xytext=(10, 10), textcoords='offset points', color='orange', fontsize=12)
#ax.annotate('Actual End', xy=(final_position[1], final_position[0]), xytext=(10, 10), textcoords='offset points', color='purple', fontsize=12)

# Add annotations for the nearest points
#for lon, lat in zip(results['longitude'], results['latitude']):
 #   ax.annotate(f'Nearest Point', xy=(lon, lat), xytext=(10, 10), textcoords='offset points', color='cyan', fontsize=12)

# Set labels and title
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Areas in Coordinate System and Calculated Positions')

# Add legend
ax.legend()

# Show plot
plt.grid(True)
plt.show()
"""

def plot_areas(ax):
    for area, coords in areas.items():
        latitudes = coords['latitude']
        longitudes = coords['longitude']
        ax.plot(longitudes, latitudes, color='black', linewidth=3, marker='o')

# Funktion zur Überprüfung der Klassifizierung
def is_classification_correct(predicted_position, true_position):
    return predicted_position == true_position

# Plot each prediction path in a separate graph
fig, axes = plt.subplots(2, 2, figsize=(20, 20))

predicted_lats_MLP, predicted_longs_MLP = zip(*predicted_positions)
# Berechnung der Endpositionen
predicted_end_pos_jeder_datenpunkt = find_nearest_point(lats[-1],lons[-1],allLat,allLong)
predicted_end_pos_nur_eckpunkte = find_nearest_point( path_lats[-1],path_lons[-1],allLat,allLong)
predicted_end_pos_nn = find_nearest_point( predicted_lats_MLP[-1],predicted_longs_MLP[-1],allLat,allLong)

# Überprüfung der Klassifizierungen
classification_jeder_datenpunkt = is_classification_correct(predicted_end_pos_jeder_datenpunkt, echte_endposition_korrigiert)
classification_nur_eckpunkte = is_classification_correct(predicted_end_pos_nur_eckpunkte, echte_endposition_korrigiert)
classification_nn = is_classification_correct(predicted_end_pos_nn, echte_endposition_korrigiert)

# Plot each area as a line segment and annotate points
for i, ax in enumerate(axes.flat):
    plot_areas(ax)
    if i == 0:
        ax.plot(actual_lons, actual_lats, linestyle=':', color='pink', label='Actual Path', linewidth=4)
        
        ax.scatter(lon_final_corrected, lat_final_corrected,  color='darkblue', s=100, label='Actual Corrected End Point', marker='x')
        ax.set_title('Actual Path')
    elif i == 1:
        ax.plot(lons, lats, linestyle=':', color='purple', label='Prediction Path - jeder Datenpunkt', linewidth=4)
        lat_predicted_end_pos_jeder_datenpunkt, lon_predicted_end_pos_jeder_datenpunkt = predicted_end_pos_jeder_datenpunkt
        ax.scatter(lon_predicted_end_pos_jeder_datenpunkt, lat_predicted_end_pos_jeder_datenpunkt, color='darkblue', s=100, label='Predicted Corrected End Point', marker='x')
        title_suffix = " (Richtig Klassifiziert)" if classification_jeder_datenpunkt else " (Falsch Klassifiziert)"
        ax.set_title('Prediction Path - Jeder Datenpunkt' + title_suffix)
    elif i == 2:
        ax.plot(path_lons, path_lats, marker='o', linestyle=':', color='fuchsia', label='Prediction Path - nur Eckpunkte', linewidth=4)
        ax.scatter(path_lons[-1], path_lats[-1], color='darkblue', s=100, label='Predicted Corrected End Point', marker='x')
        title_suffix = " (Richtig Klassifiziert)" if classification_nur_eckpunkte else " (Falsch Klassifiziert)"
        ax.set_title('Prediction Path - Nur Eckpunkte' + title_suffix)
    elif i == 3:
        
        ax.plot(predicted_longs_MLP, predicted_lats_MLP, color="brown", linestyle='-', marker='o', label='Predicted Path - NN', linewidth=3)
        ax.scatter(predicted_longs_MLP[-1], predicted_lats_MLP[-1], color='darkblue', s=100, label='Predicted Corrected End Point', marker='x')
        title_suffix = " (Richtig Klassifiziert)" if classification_nn else " (Falsch Klassifiziert)"
        ax.set_title('Predicted Path - NN' + title_suffix)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True)

plt.show()