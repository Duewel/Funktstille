"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic

# Beispiel-Daten initialisieren (ersetzen Sie dies durch Ihren tatsächlichen Pfad und Ihre Daten)
test_path = "test/set11.csv"
data = pd.read_csv(test_path)

# Initiale GPS-Position (ersetzen Sie dies durch Ihre tatsächliche Startposition)
initial_position = (data["locationLatitude(WGS84)"].iloc[0], data["locationLongitude(WGS84)"].iloc[0])
final_position = (data["locationLatitude(WGS84)"].iloc[-1], data["locationLongitude(WGS84)"].iloc[-1])

initial_heading = data['locationTrueHeading(°)'].iloc[0]
fixed_speed = 0.9

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

# Aktualisierung der zukünftigen Winkel
for index, row in data.iterrows():
    future_time = row["future_time"]
    future_row = data[data["time"] >= future_time].iloc[0] if not data[data["time"] >= future_time].empty else None
    if future_row is not None:
        data.at[index, "future_angle"] = future_row["current_angle"]

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

        if min_distance < 5:  # Schwellenwert für signifikante Nähe (z.B. 10 Meter)
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
        'future_angle': future_angles
    })

    # Sortieren nach Zeit
    results = results.sort_values(by='time').reset_index(drop=True)

    # Berechnen von delta_time
    results['delta_time'] = results['time'].diff().fillna(0)

    return results

# Eckpunkte des Parcours
allLat = [53.2295966, 53.2295037, 53.2294835, 53.2294804, 53.2296053, 53.2298517, 53.2298470, 53.2297627, 53.2295982, 53.2294744, 53.2293866, 53.2293154, 53.2293119, 53.2294001, 53.2294155, 53.2294238, 53.2293211, 53.2296167, 53.2297696]
allLong = [10.4053239, 10.4053357, 10.4050846, 10.4048623, 10.4048372, 10.4047826, 10.4046903, 10.4046940, 10.4047287, 10.4047508, 10.4047560, 10.4048099, 10.4048889, 10.4048764, 10.4051001, 10.4053372, 10.4050829, 10.4050307, 10.4046194]

# Finden der nächstgelegenen Punkte und Zeiten
results = find_nearest_points_with_time(data, allLat, allLong, angles)

# Ausgabe der Ergebnisse
print(results)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(15, 10))

# Plot each area as a line segment and annotate points
for area, coords in areas.items():
    latitudes = coords['latitude']
    longitudes = coords['longitude']
    ax.plot(longitudes, latitudes, label=area)
    for lat, lon in zip(latitudes, longitudes):
        ax.annotate(f'({lon:.6f}, {lat:.7f})', xy=(lon, lat), textcoords='offset points', xytext=(5,5), ha='right', fontsize=8, color='red')

# Plot the calculated positions
ax.plot(lons, lats, marker='o', linestyle='-', color='blue', label='Calculated Positions')

# Highlight the initial and final positions
ax.scatter(lons[0], lats[0], color='green', s=100, label='Calculated Start Point')
ax.scatter(lons[-1], lats[-1], color='red', s=100, label='Calculated End Point')
ax.scatter(initial_position[1], initial_position[0], color='orange', s=100, label='Actual Start Point', marker='x')
ax.scatter(final_position[1], final_position[0], color='purple', s=100, label='Actual End Point', marker='x')

# Highlight the nearest points
ax.scatter(results['longitude'], results['latitude'], color='cyan', s=100, label='Nearest Points', marker='x')

# Add annotations for the initial and final positions
ax.annotate('Calculated Start', xy=(lons[0], lats[0]), xytext=(10, 10), textcoords='offset points', color='green', fontsize=12)
ax.annotate('Calculated End', xy=(lons[-1], lats[-1]), xytext=(10, 10), textcoords='offset points', color='red', fontsize=12)
ax.annotate('Actual Start', xy=(initial_position[1], initial_position[0]), xytext=(10, 10), textcoords='offset points', color='orange', fontsize=12)
ax.annotate('Actual End', xy=(final_position[1], final_position[0]), xytext=(10, 10), textcoords='offset points', color='purple', fontsize=12)

# Add annotations for the nearest points
for lon, lat in zip(results['longitude'], results['latitude']):
    ax.annotate(f'Nearest Point', xy=(lon, lat), xytext=(10, 10), textcoords='offset points', color='cyan', fontsize=12)

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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic

# Beispiel-Daten initialisieren (ersetzen Sie dies durch Ihren tatsächlichen Pfad und Ihre Daten)
test_path = "test/set11.csv"
data = pd.read_csv(test_path)

# Initiale GPS-Position (ersetzen Sie dies durch Ihre tatsächliche Startposition)
initial_position = (data["locationLatitude(WGS84)"].iloc[0], data["locationLongitude(WGS84)"].iloc[0])
final_position = (data["locationLatitude(WGS84)"].iloc[-1], data["locationLongitude(WGS84)"].iloc[-1])

initial_heading = data['locationTrueHeading(°)'].iloc[0]
fixed_speed = 0.95

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

# Aktualisierung der zukünftigen Winkel
for index, row in data.iterrows():
    future_time = row["future_time"]
    future_row = data[data["time"] >= future_time].iloc[0] if not data[data["time"] >= future_time].empty else None
    if future_row is not None:
        data.at[index, "future_angle"] = future_row["current_angle"]

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
        'future_angle': future_angles
    })

    # Sortieren nach Zeit
    results = results.sort_values(by='time').reset_index(drop=True)

    # Berechnen von delta_time
    results['delta_time'] = results['time'].diff().fillna(0)

    return results

# Eckpunkte des Parcours
allLat = [53.2295966, 53.2295037, 53.2294835, 53.2294804, 53.2296053, 53.2298517, 53.2298470, 53.2297627, 53.2295982, 53.2294744, 53.2293866, 53.2293154, 53.2293119, 53.2294001, 53.2294155, 53.2294238, 53.2293211, 53.2296167, 53.2297696]
allLong = [10.4053239, 10.4053357, 10.4050846, 10.4048623, 10.4048372, 10.4047826, 10.4046903, 10.4046940, 10.4047287, 10.4047508, 10.4047560, 10.4048099, 10.4048889, 10.4048764, 10.4051001, 10.4053372, 10.4050829, 10.4050307, 10.4046194]

# Finden der nächstgelegenen Punkte und Zeiten
results = find_nearest_points_with_time(data, allLat, allLong, angles)

# Ausgabe der Ergebnisse
print(results)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(15, 10))

# Plot each area as a line segment and annotate points
for area, coords in areas.items():
    latitudes = coords['latitude']
    longitudes = coords['longitude']
    ax.plot(longitudes, latitudes, label=area)
    for lat, lon in zip(latitudes, longitudes):
        ax.annotate(f'({lon:.6f}, {lat:.7f})', xy=(lon, lat), textcoords='offset points', xytext=(5,5), ha='right', fontsize=8, color='red')

# Plot the calculated positions
ax.plot(lons, lats, marker='o', linestyle='-', color='blue', label='Calculated Positions')

# Highlight the initial and final positions
ax.scatter(lons[0], lats[0], color='green', s=100, label='Calculated Start Point')
ax.scatter(lons[-1], lats[-1], color='red', s=100, label='Calculated End Point')
ax.scatter(initial_position[1], initial_position[0], color='orange', s=100, label='Actual Start Point', marker='x')
ax.scatter(final_position[1], final_position[0], color='purple', s=100, label='Actual End Point', marker='x')

# Highlight the nearest points
ax.scatter(results['longitude'], results['latitude'], color='cyan', s=100, label='Nearest Points', marker='x')

# Add annotations for the initial and final positions
ax.annotate('Calculated Start', xy=(lons[0], lats[0]), xytext=(10, 10), textcoords='offset points', color='green', fontsize=12)
ax.annotate('Calculated End', xy=(lons[-1], lats[-1]), xytext=(10, 10), textcoords='offset points', color='red', fontsize=12)
ax.annotate('Actual Start', xy=(initial_position[1], initial_position[0]), xytext=(10, 10), textcoords='offset points', color='orange', fontsize=12)
ax.annotate('Actual End', xy=(final_position[1], final_position[0]), xytext=(10, 10), textcoords='offset points', color='purple', fontsize=12)

# Add annotations for the nearest points
for lon, lat in zip(results['longitude'], results['latitude']):
    ax.annotate(f'Nearest Point', xy=(lon, lat), xytext=(10, 10), textcoords='offset points', color='cyan', fontsize=12)

# Set labels and title
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Areas in Coordinate System and Calculated Positions')

# Add legend
ax.legend()

# Show plot
plt.grid(True)
plt.show()
