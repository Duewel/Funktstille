# Gute version ohne Grid: 

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_data_from_csv(folder_path):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            # Ensure there are no infinite or NaN values
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
            # Extract features and label for every step but the last
            for i in range(len(df) - 1):
                data.append([
                    df['corrected_latitude'].iloc[i],
                    df['corrected_longitude'].iloc[i],
                    df['future_angle'].iloc[i],
                    df['delta_time'].iloc[i+1],
                    df['corrected_latitude'].iloc[i+1],
                    df['corrected_longitude'].iloc[i+1]
                ])
    columns = ['initial_lat', 'initial_long', 'future_angle', 'delta_time', 'end_lat', 'end_long']
    return pd.DataFrame(data, columns=columns)


folder_path = 'outputcsv'  
df = load_data_from_csv(folder_path)


df['end_position'] = df[['end_lat', 'end_long']].apply(lambda row: f"{row['end_lat']}_{row['end_long']}", axis=1)


label_encoder = LabelEncoder()
df['end_position_label'] = label_encoder.fit_transform(df['end_position'])


X = df[['initial_lat', 'initial_long', 'future_angle', 'delta_time']]
y = df['end_position_label']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler and label encoder
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)


X_scaled = np.array(X_scaled)
y = np.array(y)

print(len(X_scaled))


X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Cross-Validation vorbereiten
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Modell definieren mit angepassten Hyperparametern
def create_model(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    
    # Layer 0
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.1))
    
    # Layer 1
    model.add(Dense(units=224, activation='relu'))
    model.add(Dropout(rate=0.2))
    
    # Layer 2
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(rate=0.0))
    
    # Layer 3
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(rate=0.0))
    
    # Layer 4
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(rate=0.0))
    
    # Output layer
    model.add(Dense(units=19, activation='softmax'))  # Passe die Anzahl der Klassen an dein Problem an

    # Compiling the model with the specified learning rate
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


# Cross-Validation ausführen
fold_no = 1
val_accuracies = []

for train_index, val_index in kf.split(X_train_val):
    print(f"Training fold {fold_no}...")
    X_train, X_val = X_train_val[train_index], X_train_val[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]

    model = create_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=200, batch_size=8, validation_data=(X_val, y_val))

    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    val_accuracies.append(val_accuracy)
    print(f"Fold {fold_no} - Validation Accuracy: {val_accuracy:.2f}")
    fold_no += 1

# Durchschnittliche Validierungsgenauigkeit berechnen
average_val_accuracy = np.mean(val_accuracies)
print(f"Average Validation Accuracy: {average_val_accuracy:.2f}")

# Endgültiges Modell auf gesamten Trainingsdaten trainieren
final_model = create_model(X_train_val.shape[1])
final_model.fit(X_train_val, y_train_val, epochs=100, batch_size=8, validation_split=0.2)

# Endgültiges Modell auf Testdaten evaluieren
test_loss, test_accuracy = final_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Vorhersagen machen
y_pred = final_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
final_model.save('final_model.h5')
# Vorhersagen dekodieren
predicted_positions = label_encoder.inverse_transform(y_pred_classes)

# Vorhersagen anzeigen (optional)
for i in range(10):  # Zeige die ersten 10 Vorhersagen
    print(f"Predicted: {predicted_positions[i]}, Actual: {label_encoder.inverse_transform([y_test[i]])[0]}")


# Konfusionsmatrix erstellen
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Konfusionsmatrix plotten
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Klassifikationsbericht anzeigen
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))



# Parameter Optimierung: 

"""
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch


# Funktion zum Laden der CSV-Dateien und Extrahieren der Features und Labels
def load_data_from_csv(folder_path):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            # Ensure there are no infinite or NaN values
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            # Extract features and label for every step but the last
            for i in range(len(df) - 1):
                data.append([
                    df['corrected_latitude'].iloc[i],
                    df['corrected_longitude'].iloc[i],
                    df['future_angle'].iloc[i],
                    df['delta_time'].iloc[i+1],
                    df['corrected_latitude'].iloc[i+1],
                    df['corrected_longitude'].iloc[i+1]
                ])
    columns = ['initial_lat', 'initial_long', 'future_angle', 'delta_time', 'end_lat', 'end_long']
    return pd.DataFrame(data, columns=columns)


# Load data from CSV
folder_path = 'outputcsv'
df = load_data_from_csv(folder_path)

# Combine end_lat and end_long into a single class label
df['end_position'] = df[['end_lat', 'end_long']].apply(lambda row: f"{row['end_lat']}_{row['end_long']}", axis=1)

# Encode class labels
label_encoder = LabelEncoder()
df['end_position_label'] = label_encoder.fit_transform(df['end_position'])

# Select features and labels
X = df[['initial_lat', 'initial_long', 'future_angle', 'delta_time']]
y = df['end_position_label']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler and label encoder
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Convert X_scaled and y to numpy arrays
X_scaled = np.array(X_scaled)
y = np.array(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define model builder function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32), activation='relu'))
        model.add(Dropout(rate=hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(len(np.unique(y)), activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-1,1e-2, 1e-3, 1e-4,1e-5])),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Define Keras Tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='keras_tuner_logs',
    project_name='neural_network_tuning3'
)

# Perform hyperparameter search
tuner.search(X_train, y_train, epochs=200, validation_split=0.2)

# Get best model and evaluate on test set
best_model = tuner.get_best_models(num_models=1)[0]
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_accuracy:.2f}")

best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:")
print(best_hyperparameters.values)

# Save the best model
best_model.save('best_model.h5')
"""