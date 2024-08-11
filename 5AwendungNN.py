import pandas as pd
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt

def predict_trajectory(test_file, model, scaler, label_encoder):
    df = pd.read_csv(os.path.join('test', test_file))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Start with the first corrected position
    current_lat = df['corrected_latitude'][0]
    current_long = df['corrected_longitude'][0]
    predicted_positions = [(current_lat, current_long)]  # Include the start point in the predicted positions
    print(f"Länge: {len(df)}")
    # Go through each row in the file
    for i in range(len(df) -1):
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
        print(len(predicted_positions))
        print(f"Predicted: ({predicted_lat}, {predicted_long}), Actual: ({df['corrected_latitude'][i+1]}, {df['corrected_longitude'][i+1]})")
    
    
    # The actual path and the last predicted position
    actual_path = df[['corrected_latitude', 'corrected_longitude']].values
    final_predicted_position = predicted_positions[-1]
    
    return predicted_positions, final_predicted_position, actual_path

# Load the model and other resources
model = load_model('final_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Call the function
test_file = 'result_0_D- Park.csv'  # Make sure you specify the correct file name
predicted_positions, final_predicted_position, actual_path = predict_trajectory(test_file, model, scaler, label_encoder)

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


# Visualize the results
plt.figure(figsize=(12, 10))
# Plotting predicted positions



for i, (lat, long) in enumerate(predicted_positions):
    plt.text(long, lat, str(i), color='red')

for area, coords in areas.items():
    plt.plot(coords['longitude'], coords['latitude'], 'k-', marker = 'o', linewidth = 3)

#Echter pfad
actual_lats, actual_longs = zip(*actual_path)
plt.plot(actual_longs, actual_lats, color = 'blue', linestyle = '-', label='Actual Path', linewidth = 2)

#Vorausgesagt NN
predicted_lats, predicted_longs = zip(*predicted_positions)
plt.plot(predicted_longs, predicted_lats, color='fuchsia', linestyle = ':', marker = 'o', label='Predicted Path - NN', linewidth = 3)


plt.scatter(actual_longs[-1], actual_lats[-1], color='blue', s=100, label='Actual Final Position', zorder=5)
plt.text(actual_longs[-1], actual_lats[-1], 'End', color='blue')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Predicted and Actual Path')
plt.legend()
plt.grid(True)
plt.show()
