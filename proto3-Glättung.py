import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from shapely.geometry import LineString, Point
from scipy.signal import savgol_filter

# Funktion zum Laden von Trainingsdaten aus dem angegebenen Ordner
def load_training_data(folder_path):
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            data.append(df)
    return pd.concat(data, ignore_index=True)

# Laden der Trainingsdaten
folder_path = "training"
training_data = load_training_data(folder_path)

# Definieren von Features und Labels
features = ['accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)',
            'gyroRotationX(rad/s)', 'gyroRotationY(rad/s)', 'gyroRotationZ(rad/s)', 
            'accelerometerTimestamp_sinceReboot(s)', 'gyroTimestamp_sinceReboot(s)']
labels = ['locationLatitude(WGS84)', 'locationLongitude(WGS84)']

# Aufteilen der Daten in Features (X) und Labels (y)
X = training_data[features]
y = training_data[labels]

# Aufteilen der Daten in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Trainieren der KNN-Modelle für Latitude und Longitude
knn_latitude = KNeighborsRegressor(n_neighbors=10)
knn_longitude = KNeighborsRegressor(n_neighbors=10)

knn_latitude.fit(X_train, y_train['locationLatitude(WGS84)'])
knn_longitude.fit(X_train, y_train['locationLongitude(WGS84)'])

# Vorhersagen für die Testdaten
y_pred_latitude = knn_latitude.predict(X_test)
y_pred_longitude = knn_longitude.predict(X_test)

# Berechnung des mittleren quadratischen Fehlers
mse_latitude = mean_squared_error(y_test['locationLatitude(WGS84)'], y_pred_latitude)
mse_longitude = mean_squared_error(y_test['locationLongitude(WGS84)'], y_pred_longitude)

print(f'Mean Squared Error (Latitude): {mse_latitude}')
print(f'Mean Squared Error (Longitude): {mse_longitude}')

# Parcours-Daten
allLat = [53.2295966, 53.2295037, 53.2294835, 53.2294804, 53.2296053, 53.2298517, 53.2298470, 53.2297627, 53.2295982, 53.2294744, 53.2293866, 53.2293154, 53.2293119, 53.2294001, 53.2294155, 53.2294238, 53.2293211, 53.2296167, 53.2297696]
allLong = [10.4053239, 10.4053357, 10.4050846, 10.4048623, 10.4048372, 10.4047826, 10.4046903, 10.4046940, 10.4047287, 10.4047508, 10.4047560, 10.4048099, 10.4048889, 10.4048764, 10.4051001, 10.4053372, 10.4050829, 10.4050307, 10.4046194]

additional_latitude = [53.2295966, 53.2295037, 53.2294835, 53.2294804, 53.2296053, 53.2298517, 53.2298470, 53.2297627, 53.2295982, 53.2294744, 53.2293866, 53.2293154, 53.2293119, 53.2294001, 53.2294155, 53.2294238]
additional_longitude = [10.4053239, 10.4053357, 10.4050846, 10.4048623, 10.4048372, 10.4047826, 10.4046903, 10.4046940, 10.4047287, 10.4047508, 10.4047560, 10.4048099, 10.4048889, 10.4048764, 10.4051001, 10.4053372]

# Einzelne Linien des Parcours definieren
keller1_latitude = [53.2293119, 53.2293211]
keller1_longitude = [10.4048889, 10.4050829]

keller2_latitude = [53.2296053, 53.2296167]
keller2_longitude = [10.4048372, 10.4050307]

gang_latitude = [53.2294835, 53.2294155]
gang_longitude = [10.4050846, 10.4051001]

park_latitude = [53.2297627, 53.2297696]
park_longitude = [10.4046940, 10.4046194]

gang2_latitude = [53.2294804, 53.2294001]
gang2_longitude = [10.4048623, 10.4048764]

gang3_latitude = [53.2294001, 53.2293866]
gang3_longitude = [10.4048764, 10.4047560]

gang4_latitude = [53.2294804, 53.2294744]
gang4_longitude = [10.4048623, 10.4047508]

gang5_latitude = [53.2296053, 53.2295982]
gang5_longitude = [10.4048372, 10.4047287]

gangstart_latitude = [53.2295037, 53.2294238]
gangstart_longitude = [10.4053357, 10.4053372]

# Zeichne den Parcours
def draw_lines(latitudes, longitudes, color='red'):
    lines = []
    for i in range(len(latitudes) - 1):
        start_point = (latitudes[i], longitudes[i])
        end_point = (latitudes[i+1], longitudes[i+1])
        
        # Linear interpolierte Koordinaten zwischen den Start- und Endpunkten
        x_values = np.linspace(start_point[1], end_point[1], num=1000)  # 1000 Zwischenpunkte 
        y_values = np.linspace(start_point[0], end_point[0], num=1000)
        
        # Speichern der Linienkoordinaten
        line_coords = list(zip(y_values, x_values))
        lines.extend(line_coords)
        
        # Zeichne die Linie
        plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], color=color, linestyle='-', linewidth=2)
    return lines

# Linien für die einzelnen Bereiche des Parcours zeichnen
areas = [
    (keller1_latitude, keller1_longitude),
    (keller2_latitude, keller2_longitude),
    (gang_latitude, gang_longitude),
    (park_latitude, park_longitude),
    (gang2_latitude, gang2_longitude),
    (gang3_latitude, gang3_longitude),
    (gang4_latitude, gang4_longitude),
    (gang5_latitude, gang5_longitude),
    (gangstart_latitude, gangstart_longitude),
    (additional_latitude, additional_longitude)
]

# Sammeln aller Linienkoordinaten ohne sie zu verbinden
all_lines = []
plt.figure(figsize=(20, 12))

for latitudes, longitudes in areas:
    lines = draw_lines(latitudes, longitudes, color='blue')
    all_lines.append(LineString(lines))

# Annotiere die Koordinaten
for lat, long in zip(allLat, allLong):
    plt.annotate(f'({lat:.7f}, {long:.7f})', (long, lat), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8, color='orange')

# Funktion zum Laden eines spezifischen Testdatensatzes
def load_test_data(file_path):
    return pd.read_csv(file_path)

# Laden des spezifischen Testdatensatzes
test_file_path = "test/set11.csv"
test_data = load_test_data(test_file_path)

# Vorhersagen für den spezifischen Testdatensatz
X_test_specific = test_data[features]
y_test_specific = test_data[labels]

y_pred_latitude_specific = knn_latitude.predict(X_test_specific)
y_pred_longitude_specific = knn_longitude.predict(X_test_specific)

# Glätten der Vorhersagen
y_pred_latitude_smoothed = savgol_filter(y_pred_latitude_specific, window_length=5, polyorder=2)
y_pred_longitude_smoothed = savgol_filter(y_pred_longitude_specific, window_length=5, polyorder=2)

# Korrigierte Punkte basierend auf der nächsten Projektion auf den Parcours
corrected_points_lat = []
corrected_points_long = []

for lat, long in zip(y_pred_latitude_smoothed, y_pred_longitude_smoothed):
    point = Point(lat, long)  # Beachte die Reihenfolge: (longitude, latitude) für Point
    closest_point = None
    min_distance = float('inf')
    for line in all_lines:
        projected_point = line.interpolate(line.project(point))
        distance = point.distance(projected_point)
        if distance < min_distance:
            min_distance = distance
            closest_point = projected_point
    corrected_points_long.append(closest_point.y)
    corrected_points_lat.append(closest_point.x)

# Visualisierung der echten, vorhergesagten und korrigierten Punkte
plt.scatter(y_test_specific['locationLongitude(WGS84)'], y_test_specific['locationLatitude(WGS84)'], color='green', label='Echte Positionen')
plt.scatter(y_pred_longitude_smoothed, y_pred_latitude_smoothed, color='red', label='Vorhergesagte Positionen')
plt.scatter(corrected_points_long, corrected_points_lat, color='purple', label='Korrigierte Positionen')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.title('Parcours und Positionen')
plt.show()
