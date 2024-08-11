
"""
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

from flask import Flask, request, jsonify, send_file, render_template


# Beispiel-Daten initialisieren (ersetzen Sie dies durch Ihren tatsächlichen Pfad und Ihre Daten)
test_path = "new/D- Park.csv"
data = pd.read_csv(test_path)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods = ['POST'])
def upload_files():
    global data
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.csv'):
        data = pd.read_csv(file)




    
    return jsonify({'text': 'Yes'})




winkel_zeit_vorberechnung()


def winkel_zeit_vorberechnung():
    global actual_lats, actual_lons, initial_position, final_position
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

def schritt_und_weg_berechnung():
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
def route_einzelWinkel();
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
    df = pd.read_csv(os.path.join('test', test_file))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

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

test_file = "result_0_D- Park.csv"
# predicted postions MLP
predicted_positions, final_predicted_position = predict_trajectory(test_file, model, scaler, label_encoder)

#Vergleich echt und geschätzt
geschätzt_endposition_berechnet = positions[-1]
print(f'Endposition berechnet: {geschätzt_endposition_berechnet}')
echte_endposition_korrigiert = find_nearest_point(final_position[0], final_position[1], allLat, allLong)
print(f'Endposition echt: {echte_endposition_korrigiert}')
if echte_endposition_korrigiert == positions[-1]:
    print("Klassifizierung RICHTIG")
else: 
    print("Klassifizierung FALSCH")
    

# Extrahieren von Breiten- und Längengraden für die Darstellung des Pfads
path_lats, path_lons = zip(*positions)




# Ausgabe der Ergebnisse
print(results)


# Create a figure and axis
fig, ax = plt.subplots(figsize=(15, 10))

# Plot each area as a line segment and annotate points

for area, coords in areas.items():
    latitudes = coords['latitude']
    longitudes = coords['longitude']
    ax.plot(longitudes, latitudes, color='black',linewidth=3, marker='o')
    #for lat, lon in zip(latitudes, longitudes):
       # ax.annotate(f'({lon:.7f}, {lat:.7f})', xy=(lon, lat), textcoords='offset points', xytext=(5,5), ha='right', fontsize=8, color='grey')

# Pfad jeder einzelne Punkt
#ax.plot(lons, lats, linestyle=':', color='purple', label='Prediction Path - jeder Datenpunkt',linewidth=4) # marker entfernt

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




if __name__ == '__main__':
    app.run(debug=True)

"""


"""
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
from flask import Flask, request, jsonify, send_file, render_template
# Beispiel-Daten initialisieren (ersetzen Sie dies durch Ihren tatsächlichen Pfad und Ihre Daten)


app = Flask(__name__)


#Definitionen 
# Eckpunkte des Parcours
allLat = [53.2295966, 53.2295037, 53.2294835, 53.2294804, 53.2296053, 53.2298517, 53.2298470, 53.2297627, 53.2295982, 53.2294744, 53.2293866, 53.2293154, 53.2293119, 53.2294001, 53.2294155, 53.2294238, 53.2293211, 53.2296167, 53.2297696]
allLong = [10.4053239, 10.4053357, 10.4050846, 10.4048623, 10.4048372, 10.4047826, 10.4046903, 10.4046940, 10.4047287, 10.4047508, 10.4047560, 10.4048099, 10.4048889, 10.4048764, 10.4051001, 10.4053372, 10.4050829, 10.4050307, 10.4046194]

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
angles = []
filtered_times = []
threshold = 1  # 1 Sekunde Unterschied

#Funktionen 

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

def find_step_length(step_frequency):
    
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
def berechnung_einzel_ueber_Winkel(results2,fixed_speed):
    global delta_lat, delta_lon, positions
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
    df.dropna(inplace=True)

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





@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods = ['POST'])
def upload_files():
    global data
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.csv'):
        data = pd.read_csv(file)
        
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

        

        for i in range(1, len(significant_times)):
            if significant_times[i] - significant_times[i-1] > threshold:
                filtered_times.append(significant_times[i-1])

        # Hinzufügen des letzten Elements
        if significant_times:
            filtered_times.append(significant_times[-1])

        print(filtered_times)


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
        average_stride_length = find_step_length(step_frequency) # Durchschnittliche Schrittlänge in Metern

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

        






        #Debug

        results2 =calculate_future_angles(data, filtered_times)

        
        # Finden der nächstgelegenen Punkte und Zeiten
        results = find_nearest_points_with_time(data, allLat, allLong, angles)





        # Load the model and other resources
        model = load_model('final_model.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        berechnung_einzel_ueber_Winkel(results2,fixed_speed)
        # predicted postions MLP
        predicted_positions, final_predicted_position = predict_trajectory(results2, model, scaler, label_encoder)

        #Vergleich echt und geschätzt
        geschätzt_endposition_berechnet = positions[-1]
        print(f'Endposition berechnet: {geschätzt_endposition_berechnet}')
        echte_endposition_korrigiert = find_nearest_point(final_position[0], final_position[1], allLat, allLong)
        print(f'Endposition echt: {echte_endposition_korrigiert}')
        if echte_endposition_korrigiert == positions[-1]:
            print("Klassifizierung RICHTIG")
        else: 
            print("Klassifizierung FALSCH")
            

        # Extrahieren von Breiten- und Längengraden für die Darstellung des Pfads
        path_lats, path_lons = zip(*positions)




        # Ausgabe der Ergebnisse
        print(results)


        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(15, 10))

        # Plot each area as a line segment and annotate points

        for area, coords in areas.items():
            latitudes = coords['latitude']
            longitudes = coords['longitude']
            ax.plot(longitudes, latitudes, color='black',linewidth=3, marker='o')
            #for lat, lon in zip(latitudes, longitudes):
            # ax.annotate(f'({lon:.7f}, {lat:.7f})', xy=(lon, lat), textcoords='offset points', xytext=(5,5), ha='right', fontsize=8, color='grey')

        # Pfad jeder einzelne Punkt
        #ax.plot(lons, lats, linestyle=':', color='purple', label='Prediction Path - jeder Datenpunkt',linewidth=4) # marker entfernt

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




    
    return jsonify({'text': 'Yes'})


if __name__ == '__main__':
    app.run(debug=True)




from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from scipy.signal import find_peaks, butter, filtfilt
import pickle
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    global data
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.csv'):
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

        data = pd.read_csv(file)
        actual_lats, actual_lons, initial_position, final_position, initial_heading, data, filtered_times = process_data(data)
        positions, angles, fixed_speed = calculate_positions(data, filtered_times, None, initial_position, initial_heading)
        results2 = calculate_future_angles(data, filtered_times)
        allLat, allLong = define_coordinates()
        results = find_nearest_points_with_time(data, allLat, allLong, angles)
        positions = [(allLat[0], allLong[0])]
        for i in range(1, len(results2)):
            delta_time = results2['delta_time'][i]
            future_direction = results2['future_angle'][i-1]
            distance = fixed_speed * delta_time
            direction_rad = np.deg2rad(future_direction)
            prev_lat, prev_lon = positions[-1]
            new_position = geodesic(meters=distance).destination((prev_lat, prev_lon), np.rad2deg(direction_rad))
            corrected_pos = find_nearest_point(new_position.latitude, new_position.longitude, allLat, allLong)
            positions.append(corrected_pos)

        model, scaler, label_encoder = load_model_and_scalers()
        predicted_positions, final_predicted_position = predict_trajectory(results2, model, scaler, label_encoder)
        
        geschätzt_endposition_berechnet = positions[-1]
        echte_endposition_korrigiert = find_nearest_point(final_position[0], final_position[1], allLat, allLong)
        classification_result = "RICHTIG" if echte_endposition_korrigiert == positions[-1] else "FALSCH"
        print(f'Endposition berechnet: {geschätzt_endposition_berechnet}')
        print(f'Endposition echt: {echte_endposition_korrigiert}')
        print(f"Klassifizierung {classification_result}")

        plot_results(positions, actual_lats, actual_lons, predicted_positions, areas)
        
        return jsonify({'text': 'Yes'})

def process_data(data):
    actual_lats = data["locationLatitude(WGS84)"]
    actual_lons = data["locationLongitude(WGS84)"]
    initial_position = (data["locationLatitude(WGS84)"].iloc[0], data["locationLongitude(WGS84)"].iloc[0])
    final_position = (data["locationLatitude(WGS84)"].iloc[-1], data["locationLongitude(WGS84)"].iloc[-1])
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
    threshold = 1

    for i in range(1, len(significant_times)):
        if significant_times[i] - significant_times[i-1] > threshold:
            filtered_times.append(significant_times[i-1])

    if significant_times:
        filtered_times.append(significant_times[-1])

    return actual_lats, actual_lons, initial_position, final_position, initial_heading, data, filtered_times

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def find_step_length(step_frequency, filtered_times):
    if(filtered_times[0] > 12):
        x = (11.16 + 7.58) / (step_frequency * filtered_times[0])
        return x
    else: 
        x = 11.16 / (step_frequency * filtered_times[0])
        return x

def winkel_angleichen(result_df):
    future_angle = result_df["future_angle"]
    for i, angle in enumerate(future_angle):
        if pd.notna(angle):
            rest = angle % 90
            if rest > 50:
                result_df.at[i, "future_angle"] = (angle // 90 + 1) * 90
            else: 
                result_df.at[i, "future_angle"] = (angle // 90) * 90
    return result_df

def calculate_positions(data, filtered_times, step_frequency, initial_position, initial_heading):
    fs = 1 / data["accelerometerTimestamp_sinceReboot(s)"].diff().mean()
    cutoff = 5.0
    data["accelX_filtered"] = lowpass_filter(data['accelerometerAccelerationX(G)'], cutoff, fs)
    data["accelY_filtered"] = lowpass_filter(data['accelerometerAccelerationY(G)'], cutoff, fs)
    data["accelZ_filtered"] = lowpass_filter(data['accelerometerAccelerationZ(G)'], cutoff, fs)
    data["accel_magnitude"] = np.sqrt(data["accelX_filtered"]**2 + data["accelY_filtered"]**2 + data["accelZ_filtered"]**2)
    peaks, _ = find_peaks(data["accel_magnitude"], height=1.1, distance=fs/2)
    num_steps = len(peaks)
    total_time = data["accelerometerTimestamp_sinceReboot(s)"].iloc[-1] - data["accelerometerTimestamp_sinceReboot(s)"].iloc[0]
    step_frequency = num_steps / total_time
    average_stride_length = find_step_length(step_frequency, filtered_times)
    fixed_speed = step_frequency * average_stride_length

    time_shift_seconds = 2
    time_index = data["time"]
    sampling_rate = time_index.diff().mean()
    rows_to_shift = int(time_shift_seconds / sampling_rate)
    data["future_angle"] = data["current_angle"].shift(-rows_to_shift).fillna(np.nan)

    positions = [initial_position]
    current_position = initial_position
    current_angle = initial_heading
    angles = []

    for index, row in data.iterrows():
        delta_time = row['timedelta']
        period_angle = row['angle_degree']
        current_angle = initial_heading - period_angle
        distance = fixed_speed * delta_time
        current_position = geodesic(meters=distance).destination(current_position, current_angle)
        positions.append(current_position)
        angles.append(current_angle)

    return positions, angles, fixed_speed

def find_nearest_points_with_time(data, allLat, allLong, angles):
    corners = list(zip(allLat, allLong))
    nearest_points = []
    distances = []
    times = []
    current_angles = []
    future_angles = []

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

        if min_distance < 5:
            nearest_points.append(corner)
            distances.append(min_distance)
            times.append(nearest_time)
            current_angles.append(data.loc[nearest_index, "current_angle"])
            future_angles.append(data.loc[nearest_index, "future_angle"])

    results = pd.DataFrame({
        'latitude': [point[0] for point in nearest_points],
        'longitude': [point[1] for point in nearest_points],
        'distance': distances,
        'time': times,
        'current_angle': current_angles,
        'future_Richtung': future_angles
    })

    results = results.sort_values(by='time').reset_index(drop=True)
    results['delta_time'] = results['time'].diff().fillna(0)

    return results

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

    result_df = pd.DataFrame({
        "filtered_time": filtered_times,
        "future_angle": future_angles,
        "delta_time": delta_times
    })
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

def load_model_and_scalers():
    model = load_model('final_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, scaler, label_encoder

def predict_trajectory(test_file, model, scaler, label_encoder):
    df = test_file
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    current_lat = 53.2295966
    current_long = 10.4053239
    predicted_positions = [(current_lat, current_long)]

    for i in range(len(df) - 1):
        future_angle = df['future_angle'][i]
        delta_time = df['delta_time'][i + 1]
        features = np.array([[current_lat, current_long, future_angle, delta_time]])
        features_scaled = scaler.transform(features)
        predicted_index = np.argmax(model.predict(features_scaled), axis=1)
        predicted_lat_long = label_encoder.inverse_transform(predicted_index)[0]
        predicted_lat, predicted_long = map(float, predicted_lat_long.split('_'))
        current_lat, current_long = predicted_lat, predicted_long
        predicted_positions.append((predicted_lat, predicted_long))
        print(f"Predicted: ({predicted_lat}, {predicted_long})")

    final_predicted_position = predicted_positions[-1]
    return predicted_positions, final_predicted_position

def define_coordinates():
    allLat = [53.2295966, 53.2295037, 53.2294835, 53.2294804, 53.2296053, 53.2298517, 53.2298470, 53.2297627, 53.2295982, 53.2294744, 53.2293866, 53.2293154, 53.2293119, 53.2294001, 53.2294155, 53.2294238, 53.2293211, 53.2296167, 53.2297696]
    allLong = [10.4053239, 10.4053357, 10.4050846, 10.4048623, 10.4048372, 10.4047826, 10.4046903, 10.4046940, 10.4047287, 10.4047508, 10.4047560, 10.4048099, 10.4048889, 10.4048764, 10.4051001, 10.4053372, 10.4050829, 10.4050307, 10.4046194]
    return allLat, allLong

def plot_results(positions, actual_lats, actual_lons, predicted_positions, areas):
    path_lats, path_lons = zip(*positions)

    fig, ax = plt.subplots(figsize=(15, 10))

    for area, coords in areas.items():
        latitudes = coords['latitude']
        longitudes = coords['longitude']
        ax.plot(longitudes, latitudes, color='black', linewidth=3, marker='o')

    ax.plot(path_lons, path_lats, marker='o', linestyle=':', color='fuchsia', label='Prediction Path - nur Eckpunkte', linewidth=4)
    ax.plot(actual_lons, actual_lats, linestyle=':', color='pink', label='Actual Path', linewidth=4)
    predicted_lats_MLP, predicted_longs_MLP = zip(*predicted_positions)
    ax.plot(predicted_longs_MLP, predicted_lats_MLP, color="brown", linestyle='-', marker='o', label="Predicted Path - NN", linewidth=3)

    initial_position = positions[0]
    final_position = positions[-1]
    ax.scatter(initial_position[1], initial_position[0], color='cyan', s=100, label='Actual Start Point', marker='x')
    ax.scatter(final_position[1], final_position[0], color='cyan', s=100, label='Actual End Point')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Areas in Coordinate System and Calculated Positions')
    ax.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    app.run(debug=True)
"""


import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
import matplotlib.pyplot as plt
import seaborn as sns

# Funktion zum Laden der CSV-Dateien und Extrahieren der Features und Labels
def load_data_from_csv(folder_path):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            # Ensure there are no infinite or NaN values
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            # Extract features and label for every step but the last
            for i in range(len(df) - 1):
                data.append([
                    df['corrected_latitude'].iloc[i],
                    df['corrected_longitude'].iloc[i],
                    df['future_angle'].iloc[i],
                    df['delta_time'].iloc[i+1],
                    df['corrected_latitude'].iloc[i+1],
                    df['corrected_longitude'].iloc[i+1]
                ])
    columns = ['initial_lat', 'initial_long', 'future_angle', 'delta_time', 'end_lat', 'end_long']
    return pd.DataFrame(data, columns=columns)

# Load data from CSV
folder_path = 'outputcsv'
df = load_data_from_csv(folder_path)

# Combine end_lat and end_long into a single class label
df['end_position'] = df[['end_lat', 'end_long']].apply(lambda row: f"{row['end_lat']}_{row['end_long']}", axis=1)

# Encode class labels
label_encoder = LabelEncoder()
df['end_position_label'] = label_encoder.fit_transform(df['end_position'])

# Select features and labels
X = df[['initial_lat', 'initial_long', 'future_angle', 'delta_time']]
y = df['end_position_label']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler and label encoder
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Convert X_scaled and y to numpy arrays
X_scaled = np.array(X_scaled)
y = np.array(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define model builder function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32), activation='relu'))
        model.add(Dropout(rate=hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(len(np.unique(y)), activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Define Keras Tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='keras_tuner_logs',
    project_name='neural_network_tuning3'
)

# Perform hyperparameter search
tuner.search(X_train, y_train, epochs=200, validation_split=0.2)

# Get best model and evaluate on test set
best_model = tuner.get_best_models(num_models=1)[0]
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_accuracy:.2f}")

best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:")
print(best_hyperparameters.values)

# Save the best model
best_model.save('best_model.h5')

# Save the search results
tuner.results_summary()
results = tuner.oracle.get_best_trials(num_trials=10)
with open('tuning_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Load the search results
with open('tuning_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Extract hyperparameter values and corresponding accuracies
hyperparameters = []
accuracies = []
for trial in results:
    hyperparameters.append(trial.hyperparameters.values)
    accuracies.append(trial.score)

# Convert to DataFrame for easy plotting
df_results = pd.DataFrame(hyperparameters)
df_results['accuracy'] = accuracies

# Plotting function for hyperparameters
def plot_hyperparameters(df, param_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[param_name], df['accuracy'])
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs {param_name}')
    plt.show()

# Example plots for individual hyperparameters
plot_hyperparameters(df_results, 'learning_rate')
plot_hyperparameters(df_results, 'units_0')
plot_hyperparameters(df_results, 'dropout_0')

# Pairplot to see interactions
sns.pairplot(df_results)
plt.show()

# Heatmap for correlation between hyperparameters and accuracy
plt.figure(figsize=(10, 8))
sns.heatmap(df_results.corr(), annot=True, cmap='coolwarm')
plt.show()
