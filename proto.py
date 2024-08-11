
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

# Verzeichnis mit CSV-Dateien
folder_path = "training"
file_names = os.listdir(folder_path)

# Listen für die Koordinaten
latitude_list = []
longitude_list = []

# Daten getrennt visualisieren
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        
        # Koordinaten hinzufügen
        latitude_list.extend(df['locationLatitude(WGS84)'])
        longitude_list.extend(df['locationLongitude(WGS84)'])

# Koordinaten in ein numpy Array umwandeln
coords = np.array(list(zip(latitude_list, longitude_list)))

# Standardisierung der Koordinaten
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

# DBSCAN Clustering
db = DBSCAN(eps=0.005, min_samples=70).fit(coords_scaled)
labels = db.labels_

# Filter für die Hauptclusterpunkte
mask = labels != -1
core_coords = coords[mask]

# Zusätzliche Koordinaten
additional_latitude = [53.2295966, 53.2295037, 53.2294835, 53.2294804, 53.2296053, 53.2298517, 53.2298470, 53.2297627, 53.2295982, 53.2294744, 53.2293866, 53.2293154, 53.2293119, 53.2294001, 53.2294155, 53.2294238]
additional_longitude = [10.4053239, 10.4053357, 10.4050846, 10.4048623, 10.4048372, 10.4047826, 10.4046903, 10.4046940, 10.4047287, 10.4047508, 10.4047560, 10.4048099, 10.4048889, 10.4048764, 10.4051001, 10.4053372]

allLat = [53.2295966, 53.2295037, 53.2294835, 53.2294804, 53.2296053, 53.2298517, 53.2298470, 53.2297627, 53.2295982, 53.2294744, 53.2293866, 53.2293154, 53.2293119, 53.2294001, 53.2294155, 53.2294238, 53.2293211, 53.2296167, 53.2297696]
allLong = [10.4053239, 10.4053357, 10.4050846, 10.4048623, 10.4048372, 10.4047826, 10.4046903, 10.4046940, 10.4047287, 10.4047508, 10.4047560, 10.4048099, 10.4048889, 10.4048764, 10.4051001, 10.4053372, 10.4050829, 10.4050307, 10.4046194]

# Geschätzte Koordinaten
keller1_latitude = [53.2293119, 53.2293211]
keller1_longitude = [10.4048889, 10.4050829]

keller2_latitude = [53.2296053,53.2296167]
keller2_longitude = [10.4048372, 10.4050307]

gang_latitude = [53.2294835, 53.2294155]
gang_longitude = [10.4050846, 10.4051001]

park_latitude = [53.2297627,53.2297696]
park_longitude = [10.4046940, 10.4046194]

gang2_latitude = [53.2294804, 53.2294001]
gang2_longitude = [10.4048623, 10.4048764]

gang3_latitude = [53.2294001,53.2293866]
gang3_longitude = [10.4048764,10.4047560]

gang4_latitude = [53.2294804, 53.2294744]
gang4_longitude = [10.4048623, 10.4047508]

gang5_latitude = [53.2296053, 53.2295982]
gang5_longitude = [10.4048372,10.4047287]

gangstart_latitude = [53.2295037,53.2294238]
gangstart_longitude = [10.4053357,10.4053372]

# Visualisierung der Cluster und der zusätzlichen Koordinaten
plt.figure(figsize=(20, 12))

# Zeichnen der Cluster
# sns.kdeplot(x=core_coords[:, 1], y=core_coords[:, 0], cmap='hot', fill=True, bw_adjust=0.5)
# plt.scatter(core_coords[:, 1], core_coords[:, 0], c='blue', s=10, alpha=0.5, label='Clustered Points')

# Zusätzliche Koordinaten und Route zeichnen
plt.scatter(allLong, allLat, color='green', marker='o', label='Additional Points')

# Linien zwischen den Punkten zeichnen und Linienkoordinaten speichern
all_lines = []

def draw_lines(latitudes, longitudes, color='red'):
    for i in range(len(latitudes) - 1):
        start_point = (latitudes[i], longitudes[i])
        end_point = (latitudes[i+1], longitudes[i+1])
        
        # Linear interpolierte Koordinaten zwischen den Start- und Endpunkten
        x_values = np.linspace(start_point[1], end_point[1], num=1000) # 100 Zwischenpunkte 
        y_values = np.linspace(start_point[0], end_point[0], num=1000)
        
        # Speichern der Linienkoordinaten
        line_coords = list(zip(y_values, x_values))
        all_lines.extend(line_coords)
        
        # Zeichne die Linie
        plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], color=color, linestyle='-', linewidth=2)

# Linien für die zusätzlichen Koordinaten zeichnen
draw_lines(additional_latitude, additional_longitude, color='yellow')

# Linien für alle geschätzten Bereiche zeichnen
areas = [
    (keller1_latitude, keller1_longitude),
    (keller2_latitude, keller2_longitude),
    (gang_latitude, gang_longitude),
    (park_latitude, park_longitude),
    (gang2_latitude, gang2_longitude),
    (gang3_latitude, gang3_longitude),
    (gang4_latitude, gang4_longitude),
    (gang5_latitude, gang5_longitude),
    (gangstart_latitude, gangstart_longitude)
]

for latitudes, longitudes in areas:
    draw_lines(latitudes, longitudes, color='yellow')

# Koordinaten annotieren
for lat, long in zip(allLat, allLong):
    plt.annotate(f'({lat:.7f}, {long:.7f})', (long, lat), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8, color='orange')

# Speichern der Linienkoordinaten als LineString für die Projektion
route_linestring = LineString(all_lines)


def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
# Funktion zur Projektion der vorhergesagten Positionen auf die nächste Linie
# Funktion zur Projektion der vorhergesagten Positionen auf die nächste Linie unter Verwendung der euklidischen Distanz
def project_to_route(predicted_lat, predicted_long, route_coords):
    # Starte mit der Annahme, dass der erste Punkt der nächstgelegene ist
    min_distance = euclidean_distance((predicted_lat, predicted_long), route_coords[0])
    closest_point = route_coords[0]
    
    # Durchlaufe alle Punkte entlang der Linie und aktualisiere den nächsten Punkt, falls ein näherer gefunden wird
    for point in route_coords[1:]:
        distance = euclidean_distance((predicted_lat, predicted_long), point)
        if distance < min_distance:
            min_distance = distance
            closest_point = point
    
    return closest_point

data_list = []

# Laden aller CSV-Dateien
for file_name in file_names:
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        data_list.append(df)

# Konkateniere alle Datenframes
all_data = pd.concat(data_list, ignore_index=True)

# Feature Engineering: Berechne kumulierte Geschwindigkeit und Position
all_data['accel_x_m_s2'] = all_data['accelerometerAccelerationX(G)'] * 9.81
all_data['accel_y_m_s2'] = all_data['accelerometerAccelerationY(G)'] * 9.81
all_data['accel_z_m_s2'] = all_data['accelerometerAccelerationZ(G)'] * 9.81

all_data['dt'] = all_data['accelerometerTimestamp_sinceReboot(s)'].diff().fillna(0)

all_data['velocity_x'] = (all_data['accel_x_m_s2'] * all_data['dt']).cumsum()
all_data['velocity_y'] = (all_data['accel_y_m_s2'] * all_data['dt']).cumsum()
all_data['velocity_z'] = (all_data['accel_z_m_s2'] * all_data['dt']).cumsum()

all_data['position_x'] = (all_data['velocity_x'] * all_data['dt']).cumsum()
all_data['position_y'] = (all_data['velocity_y'] * all_data['dt']).cumsum()
all_data['position_z'] = (all_data['velocity_z'] * all_data['dt']).cumsum()

# Wähle relevante Features
features = ['accel_x_m_s2', 'accel_y_m_s2', 'accel_z_m_s2', 'velocity_x', 'velocity_y', 'velocity_z', 'position_x', 'position_y', 'position_z']

# Zielvariablen sind die Endpositionen
targets = all_data[['locationLatitude(WGS84)', 'locationLongitude(WGS84)', 'locationAltitude(m)']]

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(all_data[features], targets, test_size=0.2, random_state=42)

# Standardisierung der Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lineare Regression
linear_model = MultiOutputRegressor(LinearRegression())
linear_model.fit(X_train_scaled, y_train)

# Modellbewertung der linearen Regression
linear_score = linear_model.score(X_test_scaled, y_test)
print(f"Modellgenauigkeit der linearen Regression: {linear_score}")

# k-NN Regression
knn_model = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5))
knn_model.fit(X_train_scaled, y_train)

# Modellbewertung des k-NN Regressors
knn_score = knn_model.score(X_test_scaled, y_test)
print(f"Modellgenauigkeit des k-NN Regressors: {knn_score}")

# Gradient Boosting Regression
gbr_model = MultiOutputRegressor(GradientBoostingRegressor())
param_grid = {
    'estimator__n_estimators': [50],
    'estimator__learning_rate': [0.1],
    'estimator__max_depth': [3]
}

grid_search = GridSearchCV(gbr_model, param_grid, cv=2, n_jobs=-1, scoring='r2', error_score='raise')
try:
    grid_search.fit(X_train_scaled[:10000], y_train[:10000])  # Verwenden Sie nur einen Teil der Daten
    # Bestes Modell aus GridSearch
    best_gbr_model = grid_search.best_estimator_
    gbr_score = best_gbr_model.score(X_test_scaled, y_test)
    print(f"Modellgenauigkeit des besten Gradient Boosting Regressors: {gbr_score}")
except ValueError as e:
    print(f"Grid Search Fehler: {e}")

# Vergleich der Modelle
print(f"Verbesserung durch Wechsel zu k-NN: {knn_score - linear_score}")

# Laden der set11.csv
file_path = os.path.join(folder_path, "set11.csv")
df = pd.read_csv(file_path)

# Berechnung der Features für set11.csv
df['accel_x_m_s2'] = df['accelerometerAccelerationX(G)'] * 9.81
df['accel_y_m_s2'] = df['accelerometerAccelerationY(G)'] * 9.81
df['accel_z_m_s2'] = df['accelerometerAccelerationZ(G)'] * 9.81

df['dt'] = df['accelerometerTimestamp_sinceReboot(s)'].diff().fillna(0)

df['velocity_x'] = (df['accel_x_m_s2'] * df['dt']).cumsum()
df['velocity_y'] = (df['accel_y_m_s2'] * df['dt']).cumsum()
df['velocity_z'] = (df['accel_z_m_s2'] * df['dt']).cumsum()

df['position_x'] = (df['velocity_x'] * df['dt']).cumsum()
df['position_y'] = (df['velocity_y'] * df['dt']).cumsum()
df['position_z'] = (df['velocity_z'] * df['dt']).cumsum()

# Skalieren der Features
X_set11_scaled = scaler.transform(df[features])

# Vorhersage mit linearem Modell
predicted_position_linear = linear_model.predict(X_set11_scaled[-1].reshape(1, -1))

# Vorhersage mit k-NN Modell
predicted_position_knn = knn_model.predict(X_set11_scaled[-1].reshape(1, -1))

# Vorhersage mit Gradient Boosting Modell (falls erfolgreich)
if 'best_gbr_model' in locals():
    predicted_position_gbr = best_gbr_model.predict(X_set11_scaled[-1].reshape(1, -1))
else:
    predicted_position_gbr = [None, None, None]

# Tatsächliche Endposition aus set11.csv
true_end_position = df[['locationLatitude(WGS84)', 'locationLongitude(WGS84)', 'locationAltitude(m)']].iloc[-1]

print(f"Vorhergesagte Endposition (Lineare Regression): Lat={predicted_position_linear[0][0]}, Lon={predicted_position_linear[0][1]}, Alt={predicted_position_linear[0][2]}")
print(f"Vorhergesagte Endposition (k-NN): Lat={predicted_position_knn[0][0]}, Lon={predicted_position_knn[0][1]}, Alt={predicted_position_knn[0][2]}")
if 'best_gbr_model' in locals():
    print(f"Vorhergesagte Endposition (Gradient Boosting): Lat={predicted_position_gbr[0][0]}, Lon={predicted_position_gbr[0][1]}, Alt={predicted_position_gbr[0][2]}")
print(f"Tatsächliche Endposition: Lat={true_end_position['locationLatitude(WGS84)']}, Lon={true_end_position['locationLongitude(WGS84)']}, Alt={true_end_position['locationAltitude(m)']}")


# Beispiel: Projektion einer vorhergesagten Position
#predicted_lat =  53.22950647103114  # Linear Regression 53.22952063167624 KNN 53.22946543742027

#predicted_long = 10.405101931480711 # Linear Regression 10.405008261397096 KNN 10.40516691946708
predicted_lat = predicted_position_gbr[0][0]
predicted_long = predicted_position_gbr[0][1]

corrected_lat, corrected_long= project_to_route(predicted_lat, predicted_long, all_lines)

real_lat = 53.22931533548023
real_long = 10.40482934497705


print("Koordinaten von route_linestring:")
for coord in route_linestring.coords:
    print(coord)


# Visualisierung des vorhergesagten und des korrigierten Punktes
plt.scatter([predicted_long], [predicted_lat], color='orange', marker='x', s=100, label='Predicted Point')
plt.scatter([corrected_long], [corrected_lat], color='cyan', marker='x', s=100, label='Corrected Point')
plt.scatter([real_long], [real_lat], color='yellow', marker='x', s=100, label='real Point')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clustered GPS Data with Additional Route Points and Predicted/Corrected Points')
plt.legend()
plt.grid(True)
plt.show()

print(f'Vorhergesagte Position: ({predicted_lat}, {predicted_long})')
print(f'Korrigierte Position: ({corrected_lat}, {corrected_long})')
