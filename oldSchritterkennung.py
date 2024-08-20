"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

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
test_path = "training/set8.csv"
data = pd.read_csv(test_path)

# Zeit normalisieren, so dass sie bei 0 beginnt
data["time"] = data["accelerometerTimestamp_sinceReboot(s)"] - data["accelerometerTimestamp_sinceReboot(s)"].iloc[0]

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

# Schrittlänge schätzen (beispielhaft, anpassen je nach Bedarf)
average_stride_length = 0.65  # Durchschnittliche Schrittlänge in Metern

# Berechnung der Gesamtzeitdauer des Datensatzes
total_time = data["time"].iloc[-1] - data["time"].iloc[0]

# Schrittfrequenz berechnen
step_frequency = num_steps / total_time  # Schritte pro Sekunde

# Geschwindigkeit berechnen
constant_speed = step_frequency * average_stride_length  # Geschwindigkeit in Meter pro Sekunde

# Ergebnisse ausgeben
print(f"Anzahl der Schritte: {num_steps}")
print(f"Geschätzte Gesamtdistanz: {num_steps * average_stride_length:.2f} Meter")
print(f"Schrittfrequenz: {step_frequency:.2f} Schritte pro Sekunde")
print(f"Konstante Geschwindigkeit: {constant_speed:.2f} Meter pro Sekunde")

# Plot der Ergebnisse
plt.figure(figsize=(15, 8))
plt.plot(data["time"], data["accel_magnitude"], label='Acceleration Magnitude')
plt.plot(data["time"].iloc[peaks], data["accel_magnitude"].iloc[peaks], "x", label='Detected Steps')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration Magnitude (G)')
plt.title('Step Detection from Accelerometer Data')
plt.legend()

# X-Achse in 10-Sekunden-Schritten beschriften
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10))

plt.show()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

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
test_path = "training/Royal 6.csv"
data = pd.read_csv(test_path)

# Zeit normalisieren, so dass sie bei 0 beginnt
data["time"] = data["accelerometerTimestamp_sinceReboot(s)"] - data["accelerometerTimestamp_sinceReboot(s)"].iloc[0]

# Sampling-Frequenz und Cutoff-Frequenz
fs = 1 / data["accelerometerTimestamp_sinceReboot(s)"].diff().mean()  # Abtastrate berechnet aus Zeitdifferenzen
cutoff = 5.0  # Cutoff-Frequenz in Hz, anpassen je nach Bedarf

# Anwendung des Low-Pass-Filters auf die Beschleunigungsdaten
data["accelX_filtered"] = lowpass_filter(data['accelerometerAccelerationX(G)'], cutoff, fs)
data["accelY_filtered"] = lowpass_filter(data['accelerometerAccelerationY(G)'], cutoff, fs)
data["accelZ_filtered"] = lowpass_filter(data['accelerometerAccelerationZ(G)'], cutoff, fs)

# Berechnung der Magnitude des Beschleunigungsvektors
data["accel_magnitude_filtered"] = np.sqrt(data["accelX_filtered"]**2 + data["accelY_filtered"]**2 + data["accelZ_filtered"]**2)

# Schritt-Erkennung durch Identifikation von Spitzen
peaks_filtered, _ = find_peaks(data["accel_magnitude_filtered"], height=1.1, distance=fs/2)  # Passen Sie die Parameter height und distance an

# Berechnung der Magnitude des Beschleunigungsvektors ohne Filter
data["accel_magnitude_unfiltered"] = np.sqrt(data["accelerometerAccelerationX(G)"]**2 + data["accelerometerAccelerationY(G)"]**2 + data["accelerometerAccelerationZ(G)"]**2)

# Schritt-Erkennung durch Identifikation von Spitzen ohne Filter
peaks_unfiltered, _ = find_peaks(data["accel_magnitude_unfiltered"], height=1.1, distance=fs/2)  # Passen Sie die Parameter height und distance an

# Plot der Ergebnisse
plt.figure(figsize=(15, 8))

# Plot ohne Filter
plt.subplot(2, 1, 1)
plt.plot(data["time"], data["accel_magnitude_unfiltered"], label='Acceleration Magnitude (Unfiltered)')
plt.plot(data["time"].iloc[peaks_unfiltered], data["accel_magnitude_unfiltered"].iloc[peaks_unfiltered], "x", label='Detected Steps')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration Magnitude (G)')
plt.title('Step Detection from Accelerometer Data (Unfiltered)')
plt.legend()

# Plot mit Filter
plt.subplot(2, 1, 2)
plt.plot(data["time"], data["accel_magnitude_filtered"], label='Acceleration Magnitude (Filtered)')
plt.plot(data["time"].iloc[peaks_filtered], data["accel_magnitude_filtered"].iloc[peaks_filtered], "x", label='Detected Steps')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration Magnitude (G)')
plt.title('Step Detection from Accelerometer Data (Filtered)')
plt.legend()

# X-Achse in 10-Sekunden-Schritten beschriften
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10))

plt.tight_layout()
plt.show()

# Anzahl der Schritte und andere Berechnungen (basierend auf gefilterten Daten)
num_steps_filtered = len(peaks_filtered)
average_stride_length = 0.65  # Durchschnittliche Schrittlänge in Metern
total_time = data["time"].iloc[-1] - data["time"].iloc[0]
step_frequency_filtered = num_steps_filtered / total_time  # Schritte pro Sekunde
constant_speed_filtered = step_frequency_filtered * average_stride_length  # Geschwindigkeit in Meter pro Sekunde

print(f"Gefilterte Daten: Anzahl der Schritte: {num_steps_filtered}")
print(f"Gefilterte Daten: Geschätzte Gesamtdistanz: {num_steps_filtered * average_stride_length:.2f} Meter")
print(f"Gefilterte Daten: Schrittfrequenz: {step_frequency_filtered:.2f} Schritte pro Sekunde")
print(f"Gefilterte Daten: Konstante Geschwindigkeit: {constant_speed_filtered:.2f} Meter pro Sekunde")

# Anzahl der Schritte und andere Berechnungen (basierend auf ungefilterten Daten)
num_steps_unfiltered = len(peaks_unfiltered)
step_frequency_unfiltered = num_steps_unfiltered / total_time  # Schritte pro Sekunde
constant_speed_unfiltered = step_frequency_unfiltered * average_stride_length  # Geschwindigkeit in Meter pro Sekunde

print(f"Ungefilterte Daten: Anzahl der Schritte: {num_steps_unfiltered}")
print(f"Ungefilterte Daten: Geschätzte Gesamtdistanz: {num_steps_unfiltered * average_stride_length:.2f} Meter")
print(f"Ungefilterte Daten: Schrittfrequenz: {step_frequency_unfiltered:.2f} Schritte pro Sekunde")
print(f"Ungefilterte Daten: Konstante Geschwindigkeit: {constant_speed_unfiltered:.2f} Meter pro Sekunde")
