import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic

# Daten definieren
data = pd.DataFrame({
    'time': [0.030152, 12.060731, 28.071222, 44.051562, 54.061972, 63.077402, 68.082622],
    'current_Richtung': [180.217272, 247.148515, 263.665159, 192.794154, 173.077552, 253.650897, 258.600945],
    'future_Richtung': [178.687612, 266.600010, 269.106801, 173.648501, 172.076205, 260.705702, None],
    'delta_time': [0.000000, 12.030579, 16.010491, 15.980340, 10.010410, 9.015430, 5.005220]
})

# Startposition definieren
starting_position = (53.2295966, 10.4053239)

# Feste Geschwindigkeit
fixed_speed = 0.95  # in Einheiten pro Sekunde

# Berechnung der Positionen
positions = [(starting_position[0], starting_position[1])]  # Startposition

for i in range(1, len(data)):
    delta_time = data['delta_time'][i]
    future_direction = data['future_Richtung'][i-1]
    
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
    positions.append((new_position.latitude, new_position.longitude))

# Extrahieren von Breiten- und Längengraden für die Darstellung des Pfads
path_lats, path_lons = zip(*positions)

# Parcoursbereiche definieren
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

# Erstellen des Plots für Pfad und Parcoursbereiche
fig, ax = plt.subplots(figsize=(15, 10))  # Größe des Plots

# Parcoursbereiche zeichnen
for area, coords in areas.items():
    latitudes = coords['latitude']
    longitudes = coords['longitude']
    ax.plot(longitudes, latitudes, color='orange', linewidth=2, label=area)  # Parcoursbereiche in Orange zeichnen

# Pfad zeichnen
ax.plot(path_lons, path_lats, marker='o', linestyle='-', color='blue', label='Path')  # Pfad in Blau zeichnen

# Startpunkt annotieren
ax.annotate('Start', xy=(starting_position[1], starting_position[0]), xytext=(-15, 10),
            textcoords='offset points', fontsize=12, color='green')

# Achsenbeschriftungen und Titel setzen
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Pfad und Parcoursbereiche')

# Gitter anzeigen
plt.grid(True)

# Legende anzeigen
ax.legend()

# Plot anzeigen
plt.show()
