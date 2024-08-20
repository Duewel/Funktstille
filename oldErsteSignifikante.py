"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_first_significant_change(file_path):
    data = pd.read_csv(file_path)

    data["time"] = data['gyroTimestamp_sinceReboot(s)'] - data['gyroTimestamp_sinceReboot(s)'].iloc[0]
    data["timedelta"] = data["gyroTimestamp_sinceReboot(s)"].diff().fillna(0)
    data["period_angle"] = data['gyroRotationZ(rad/s)'] * data["timedelta"]

    data["cumulated_angle_rad"] = np.cumsum(data["period_angle"])
    data["angle_degree"] = data["cumulated_angle_rad"] / np.pi * 180

    data["angle_change_per_sec"] = data["angle_degree"].diff().fillna(0) / data["timedelta"]

    significant_changes = data[np.abs(data["angle_change_per_sec"]) > 70]
    significant_times = significant_changes["time"].tolist()

    filtered_times = []
    threshold = 1  # 1 Sekunde Unterschied

    for i in range(1, len(significant_times)):
        if significant_times[i] - significant_times[i-1] > threshold:
            filtered_times.append(significant_times[i-1])
    
    # Hinzufügen des letzten Elements
    if significant_times:
        filtered_times.append(significant_times[-1])

    print(f'filtered Items: {filtered_times}')

   
        
    first_significant_change = filtered_times[0]
    if(first_significant_change < 5): 
         print(file_path)
    return first_significant_change
  
       

def main():
    folder_path = "training"
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

    times_of_first_significant_changes = []

    for file in all_files:
        first_significant_change = calculate_first_significant_change(file)
        if first_significant_change is not None:
            times_of_first_significant_changes.append(first_significant_change)

    plt.scatter(range(len(times_of_first_significant_changes)), times_of_first_significant_changes)
    plt.xlabel('Aufnahme Index')
    plt.ylabel('Zeit der ersten signifikanten Änderung (s)')
    plt.title('Erste signifikante Änderungen pro Aufnahme')
    plt.show()

if __name__ == "__main__":
    main()



"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_first_significant_change(file_path):
    data = pd.read_csv(file_path)

    data["time"] = data['gyroTimestamp_sinceReboot(s)'] - data['gyroTimestamp_sinceReboot(s)'].iloc[0]
    data["timedelta"] = data["gyroTimestamp_sinceReboot(s)"].diff().fillna(0)
    data["period_angle"] = data['gyroRotationZ(rad/s)'] * data["timedelta"]

    data["cumulated_angle_rad"] = np.cumsum(data["period_angle"])
    data["angle_degree"] = data["cumulated_angle_rad"] / np.pi * 180

    data["angle_change_per_sec"] = data["angle_degree"].diff().fillna(0) / data["timedelta"]

    significant_changes = data[np.abs(data["angle_change_per_sec"]) > 70]
    

    if not significant_changes.empty:
        
        first_significant_change = significant_changes.iloc[0]["time"]
        if(first_significant_change < 5): 
            print(file_path)
        return first_significant_change
    else:
        return None

def main():
    folder_path = "training"
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

    times_of_first_significant_changes = []

    for file in all_files:
        first_significant_change = calculate_first_significant_change(file)
        if first_significant_change is not None:
            times_of_first_significant_changes.append(first_significant_change)

    plt.scatter(range(len(times_of_first_significant_changes)), times_of_first_significant_changes)
    plt.xlabel('Aufnahme Index')
    plt.ylabel('Zeit der ersten signifikanten Änderung (s)')
    plt.title('Erste signifikante Änderungen pro Aufnahme')
    plt.show()

if __name__ == "__main__":
    main()


