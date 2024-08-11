import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from shapely.geometry import LineString, Point

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

# Trainieren der Gradient Boosting Regressoren für Latitude und Longitude
gbr_latitude = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
gbr_longitude = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)

gbr_latitude.fit(X_train, y_train['locationLatitude(WGS84)'])
gbr_longitude.fit(X_train, y_train['locationLongitude(WGS84)'])

# Vorhersagen für die Testdaten
y_pred_latitude = gbr_latitude.predict(X_test)
y_pred_longitude = gbr_longitude.predict(X_test)

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
        x_values = np.linspace(start_point[1], end_point[1], num=20)  # 20 Zwischenpunkte
        y_values = np.linspace(start_point[0], end_point[0], num=20)
        
        # Speichern der Linienkoordinaten
        line_coords = list(zip(y_values, x_values))
        lines.extend(line_coords)
        
        # Zeichne die Linie
        plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], color=color)
    
    return lines

plt.figure(figsize=(10, 6))

# Zeichnen der Parcours-Linien
lines1 = draw_lines(keller1_latitude, keller1_longitude, color='blue')
lines2 = draw_lines(keller2_latitude, keller2_longitude, color='blue')
lines3 = draw_lines(gang_latitude, gang_longitude, color='blue')
lines4 = draw_lines(park_latitude, park_longitude, color='blue')
lines5 = draw_lines(gang2_latitude, gang2_longitude, color='blue')
lines6 = draw_lines(gang3_latitude, gang3_longitude, color='blue')
lines7 = draw_lines(gang4_latitude, gang4_longitude, color='blue')
lines8 = draw_lines(gang5_latitude, gang5_longitude, color='blue')
lines9 = draw_lines(gangstart_latitude, gangstart_longitude, color='blue')

# Laden der Testdaten
test_data_path = "test/set11.csv"
test_data = pd.read_csv(test_data_path)
X_test_new = test_data[features]

# Vorhersage der Testdaten
y_pred_latitude_new = gbr_latitude.predict(X_test_new)
y_pred_longitude_new = gbr_longitude.predict(X_test_new)

# Echte und vorhergesagte Koordinaten
actual_coords = list(zip(test_data['locationLatitude(WGS84)'], test_data['locationLongitude(WGS84)']))
predicted_coords = list(zip(y_pred_latitude_new, y_pred_longitude_new))

# Alle Linien kombinieren
all_lines = lines1 + lines2 + lines3 + lines4 + lines5 + lines6 + lines7 + lines8 + lines9
line_string = LineString(all_lines)

# Projektierte Punkte auf den Parcours
corrected_coords = []
for pred_lat, pred_long in predicted_coords:
    point = Point(pred_lat, pred_long)
    nearest_point = line_string.interpolate(line_string.project(point))
    corrected_coords.append((nearest_point.x, nearest_point.y))

# Visualisieren der Koordinaten
actual_latitudes, actual_longitudes = zip(*actual_coords)
predicted_latitudes, predicted_longitudes = zip(*predicted_coords)
corrected_latitudes, corrected_longitudes = zip(*corrected_coords)

plt.scatter(actual_longitudes, actual_latitudes, c='green', label='Actual', s=10)
plt.scatter(predicted_longitudes, predicted_latitudes, c='red', label='Predicted', s=10)
plt.scatter(corrected_longitudes, corrected_latitudes, c='purple', label='Corrected', s=10)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('GPS Position Correction')
plt.legend()
plt.show()
