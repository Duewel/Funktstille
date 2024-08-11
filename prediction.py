"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Verzeichnis mit CSV-Dateien
folder_path = "training"
file_names = os.listdir(folder_path)

# Liste zum Speichern der Daten
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
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Modellbewertung
score = model.score(X_test_scaled, y_test)
print(f"Modellgenauigkeit: {score}")

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

# Vorhersage
predicted_position = model.predict(X_set11_scaled[-1].reshape(1, -1))

# Tatsächliche Endposition aus set11.csv
true_end_position = df[['locationLatitude(WGS84)', 'locationLongitude(WGS84)', 'locationAltitude(m)']].iloc[-1]

print(f"Vorhergesagte Endposition: Lat={predicted_position[0][0]}, Lon={predicted_position[0][1]}, Alt={predicted_position[0][2]}")
print(f"Tatsächliche Endposition: Lat={true_end_position['locationLatitude(WGS84)']}, Lon={true_end_position['locationLongitude(WGS84)']}, Alt={true_end_position['locationAltitude(m)']}")
"""


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

# Verzeichnis mit CSV-Dateien
folder_path = "training"
file_names = os.listdir(folder_path)

# Liste zum Speichern der Daten
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
