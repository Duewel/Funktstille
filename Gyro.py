import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_path = "training/set8.csv"

data = pd.read_csv(test_path)

print(data.columns)
# deviceOrientation(Z)

data["time"] = data['gyroTimestamp_sinceReboot(s)'] - data['gyroTimestamp_sinceReboot(s)'].iloc[0]
data["timedelta"] = data["gyroTimestamp_sinceReboot(s)"].diff().fillna(0)
data["period_angle"] = data['gyroRotationZ(rad/s)'] * data["timedelta"]

data["cumulated_angle_rad"] = np.cumsum(data["period_angle"]) # kumulierter winkel 
data["angle_degree"] = data["cumulated_angle_rad"] / np.pi * 180

# Berechnung der Änderung des Winkels in Grad pro Sekunde
data["angle_change_per_sec"] = data["angle_degree"].diff().fillna(0) / data["timedelta"]

# Finden der Zeiten mit signifikanten Änderungen
significant_changes = data[np.abs(data["angle_change_per_sec"]) > 60]

# Gruppieren der signifikanten Änderungen und Auswahl der letzten Zeitpunkte in jeder Gruppe
significant_times = significant_changes["time"].tolist()

filtered_times = []
threshold = 1  # 1 Sekunde Unterschied

for i in range(1, len(significant_times)):
    if significant_times[i] - significant_times[i-1] > threshold:
        filtered_times.append(significant_times[i-1])

# Hinzufügen des letzten Elements
if significant_times:
    filtered_times.append(significant_times[-1])

print("Zeiten mit signifikanten Änderungen (>50 Grad pro Sekunde):")
for time in filtered_times:
    print(f"{time:.2f} Sekunden")

fig, axs = plt.subplots(3, sharex=True, sharey=False, figsize=(16, 8))

for ax in axs:
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))  # Major ticks every 1 second
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))  # Minor ticks every 0.1 second
    ax.grid(which='both')  # Grid for both major and minor ticks
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')

axs[0].plot(data['time'], data['gyroRotationZ(rad/s)'])
axs[0].set(title='Angular velocity of z axis', ylabel='rad/s')

axs[1].plot(data['time'], data['period_angle'])
axs[1].set(title='Angle per period', ylabel='rad')

axs[2].plot(data['time'], data['angle_degree'])
axs[2].set(title='Cumulated angle', ylabel='degree', xlabel='time (s)')
axs[2].xaxis.set_major_locator(plt.MultipleLocator(10))  # Major ticks every 10 seconds

plt.tight_layout()
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

# Anfangsausrichtung in Grad (ersetzen Sie dies durch Ihre tatsächliche Startausrichtung)
if 'locationTrueHeading(°)' in data.columns:
    initial_heading = data['locationTrueHeading(°)'].iloc[0]
elif 'locationCourse(°)' in data.columns:
    initial_heading = data['locationCourse(°)'].iloc[0]
else:
    raise ValueError("Keine Spalte für die Anfangsausrichtung gefunden.")

# Feste Geschwindigkeit (in Metern pro Sekunde)
fixed_speed = 0.9  # Beispiel: 3 m/s

# Berechnung der Zeitdifferenzen und Winkel
data["time"] = data['gyroTimestamp_sinceReboot(s)'] - data['gyroTimestamp_sinceReboot(s)'].iloc[0]
data["timedelta"] = data["gyroTimestamp_sinceReboot(s)"].diff().fillna(0)

# Glättung der Gyroskopdaten mit einem gleitenden Durchschnitt
window_size = 4  # Fenstergröße für den gleitenden Durchschnitt
data["smoothed_gyroZ"] = data['gyroRotationZ(rad/s)'].rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')

data["period_angle"] = data['smoothed_gyroZ'] * data["timedelta"]
data["cumulated_angle_rad"] = np.cumsum(data["period_angle"])
data["angle_degree"] = data["cumulated_angle_rad"] / np.pi * 180

# Berechnung der neuen Position
positions = [initial_position]
current_position = initial_position
current_angle = initial_heading  # Startwinkel in Grad
angle_threshold = 0  # Schwellenwert für die Winkeländerung in Grad

for index, row in data.iterrows():
    delta_time = row['timedelta']
    period_angle = row['angle_degree']
    
    # Update des aktuellen Winkels nur bei großen Änderungen
    if abs(period_angle) > angle_threshold:
        current_angle =  initial_heading - period_angle
        # Reset der kumulierten Winkeländerung nach der Anpassung
        data.loc[index:, 'angle_degree'] -= period_angle
    
    # Berechnung der Distanz, die zurückgelegt wurde
    distance = fixed_speed * delta_time
    
    # Berechnung der neuen Position basierend auf dem aktuellen Winkel und der Distanz
    current_position = geodesic(meters=distance).destination(current_position, current_angle)
    positions.append(current_position)

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
ax.scatter(lons[0], lats[0], color='green', s=100, label='Start Point')
ax.scatter(lons[-1], lats[-1], color='red', s=100, label='End Point')

# Add annotations for the initial and final positions
ax.annotate('Start', xy=(lons[0], lats[0]), xytext=(10, 10), textcoords='offset points', color='green', fontsize=12)
ax.annotate('End', xy=(lons[-1], lats[-1]), xytext=(10, 10), textcoords='offset points', color='red', fontsize=12)

# Set labels and title
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Areas in Coordinate System and Calculated Positions')

# Add legend
ax.legend()

# Display the plot
plt.grid(True)
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Funktion zum Erstellen eines Butterworth-Low-Pass-Filters
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

# Pfad zur CSV-Datei
test_path = "test/set11.csv"
data = pd.read_csv(test_path)

print(data.columns)

# Sampling-Frequenz und Cutoff-Frequenz
fs = 1 / data["gyroTimestamp_sinceReboot(s)"].diff().mean()  # Abtastrate berechnet aus Zeitdifferenzen
cutoff = 0.2  # Cutoff-Frequenz in Hz, anpassen je nach Bedarf

# Anwendung des Low-Pass-Filters auf die Gyroskopdaten
data["smoothed_gyroZ"] = lowpass_filter(data['gyroRotationZ(rad/s)'], cutoff, fs)

# Berechnung der Zeitdifferenzen und Winkel
data["time"] = data['gyroTimestamp_sinceReboot(s)'] - data['gyroTimestamp_sinceReboot(s)'].iloc[0]
data["timedelta"] = data["gyroTimestamp_sinceReboot(s)"].diff().fillna(0)
data["period_angle"] = data['smoothed_gyroZ'] * data["timedelta"]
data["cumulated_angle_rad"] = np.cumsum(data["period_angle"]) # kumulierter Winkel 
data["angle_degree"] = data["cumulated_angle_rad"] / np.pi * 180

# Plot der Ergebnisse
fig, axs = plt.subplots(3, sharex=True, sharey=False, figsize=(16, 8))
axs[0].plot(data['time'], data['gyroRotationZ(rad/s)'], label='Original Data')
axs[0].plot(data['time'], data['smoothed_gyroZ'], label='Filtered Data', color='red')
axs[0].set(title='Angular velocity of z axis', xlim=(0, data['time'].max()), ylabel='rad/s', xlabel='time')
axs[0].legend()

axs[1].plot(data['time'], data['period_angle'])
axs[1].set(title='Angle per period', xlim=(0, data['time'].max()), ylabel='rad', xlabel='time')

axs[2].plot(data['time'], data['angle_degree'])
axs[2].set(title='Cumulated angle', xlim=(0, data['time'].max()), ylabel='degree', xlabel='time')

plt.tight_layout()
plt.show()


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

# Anfangsausrichtung in Grad (ersetzen Sie dies durch Ihre tatsächliche Startausrichtung)
if 'locationTrueHeading(°)' in data.columns:
    initial_heading = data['locationTrueHeading(°)'].iloc[0]
elif 'locationCourse(°)' in data.columns:
    initial_heading = data['locationCourse(°)'].iloc[0]
else:
    raise ValueError("Keine Spalte für die Anfangsausrichtung gefunden.")

# Feste Geschwindigkeit (in Metern pro Sekunde)
fixed_speed = 0.9

# Berechnung der Zeitdifferenzen und Winkel
data["time"] = data['gyroTimestamp_sinceReboot(s)'] - data['gyroTimestamp_sinceReboot(s)'].iloc[0]
data["timedelta"] = data["gyroTimestamp_sinceReboot(s)"].diff().fillna(0)
data["period_angle"] = data['gyroRotationZ(rad/s)'] * data["timedelta"]
data["cumulated_angle_rad"] = np.cumsum(data["period_angle"])
data["angle_degree"] = data["cumulated_angle_rad"] / np.pi * 180

# Berechnung der neuen Position
positions = [initial_position]
current_position = initial_position
current_angle = initial_heading  # Startwinkel in Grad

for index, row in data.iterrows():
    delta_time = row['timedelta']
    period_angle = row['angle_degree']
    
    current_angle = initial_heading - period_angle 
    
    # Berechnung der Distanz, die zurückgelegt wurde
    distance = fixed_speed * delta_time
    
    # Berechnung der neuen Position basierend auf dem aktuellen Winkel und der Distanz
    current_position = geodesic(meters=distance).destination(current_position, current_angle)
    positions.append(current_position)



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

# Add annotations for the initial and final positions
ax.annotate('Calculated Start', xy=(lons[0], lats[0]), xytext=(10, 10), textcoords='offset points', color='green', fontsize=12)
ax.annotate('Calculated End', xy=(lons[-1], lats[-1]), xytext=(10, 10), textcoords='offset points', color='red', fontsize=12)
ax.annotate('Actual Start', xy=(initial_position[1], initial_position[0]), xytext=(10, 10), textcoords='offset points', color='orange', fontsize=12)
ax.annotate('Actual End', xy=(final_position[1], final_position[0]), xytext=(10, 10), textcoords='offset points', color='purple', fontsize=12)

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

test_path = "test/set11.csv"

data = pd.read_csv(test_path)

data["time"] = data['gyroTimestamp_sinceReboot(s)'] - data['gyroTimestamp_sinceReboot(s)'].iloc[0]
data["timedelta"] = data["gyroTimestamp_sinceReboot(s)"].diff().fillna(0)
data["period_angle"] = data['gyroRotationZ(rad/s)'] * data["timedelta"]

data["cumulated_angle_rad"] = np.cumsum(data["period_angle"]) # kumulierter winkel 
data["angle_degree"] = data["cumulated_angle_rad"] / np.pi * 180

# Berechnung der Änderung des Winkels in Grad pro Sekunde
data["angle_change_per_sec"] = data["angle_degree"].diff().fillna(0) / data["timedelta"]

# Finden der Zeiten mit signifikanten Änderungen
significant_changes = data[np.abs(data["angle_change_per_sec"]) > 50]

# Gruppieren der signifikanten Änderungen und Auswahl der letzten Zeitpunkte in jeder Gruppe
significant_times = significant_changes["time"].tolist()

filtered_times = []
threshold = 1  # 1 Sekunde Unterschied

for i in range(1, len(significant_times)):
    if significant_times[i] - significant_times[i-1] > threshold:
        filtered_times.append(significant_times[i-1])

# Hinzufügen des letzten Elements
if significant_times:
    filtered_times.append(significant_times[-1])

print("Zeiten mit signifikanten Änderungen (>50 Grad pro Sekunde):")
for time in filtered_times:
    print(f"{time:.2f} Sekunden")
