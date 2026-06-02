# ESP32 Wind Turbine Sensor Control & Machine Learning Failure Prediction

This project is a comprehensive end-to-end IoT and Machine Learning system designed to monitor wind turbine conditions, analyze sensor data, and perform real-time failure prediction. It utilizes an ESP32 microcontroller to gather environmental and operational data, and a suite of Machine Learning algorithms to classify the system's status as "Normal" or "Failure" on the fly.

## **❗⚠️ IMPORTANT ⚠️❗** 

Please note that all files were bulk-uploaded at the conclusion of the project instead of individual commits by each team member!

## 🚀 Project Overview

The core objective of this project is to map physical prototype sensor readings to real-world SCADA (Supervisory Control and Data Acquisition) limits and use advanced machine learning models to predict anomalies and potential failures.

### Key Features
- **Real-Time Data Acquisition:** Gathers live sensor data (Temperatures, Wind Speed, Wind Direction, Voltage, Power, etc.) from an ESP32 via serial communication.
- **Data Scaling & Mapping:** Maps the prototype's physical readings to industrial SCADA standards in real-time.
- **Automated Data Labeling:** Scripts to automatically label raw `.csv` sensor data based on pre-defined thresholds or a baseline prototype model.
- **Extensive ML Training Pipeline:** Trains, tunes, and evaluates 10+ classification algorithms (Random Forest, XGBoost, LightGBM, CatBoost, KNN, SVM, Naive Bayes, MLP, and Stacking Ensembles) using 5-fold stratified cross-validation.
- **Model Explainability:** Integrates SHAP and LIME to interpret model decisions and feature importance.
- **Live Prediction Dashboard:** A terminal-based real-time dashboard that displays incoming sensor values, scaled features, and the AI's current status prediction (Normal / Failure).

## 🛠️ Technologies Used

### Hardware
- **Microcontroller:** ESP32
- **Sensors:** 
  - DHT11 / Temperature Sensors (Ambient, Nacelle, Gearbox)
  - Anemometer & Wind Vane (Wind Speed & Direction)
  - Voltage & Current Sensors (Turbine Power tracking)
  - Additional sensors: Gas, Line, Sound Level, Vibration

### Software & Libraries
- **Language:** Python 3.x, C++ (Arduino IDE for ESP32)
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
- **Visualization:** `matplotlib`, `seaborn`
- **Model Explainability:** `shap`, `lime`
- **Communication:** `pyserial` (for ESP32 PC communication)

## 📁 Project Structure

- `new_sensors_model.py`: The main training script. It loads `master_labeled_data.csv`, trains various ML models, performs hyperparameter tuning (`RandomizedSearchCV`), generates confusion matrices/ROC curves, and exports the `.pkl` models.
- `real_time_predict.py`: The live inference script. It connects to the ESP32 via Serial (e.g., `COM3`), reads the data, applies `MinMaxScaler` mapping, and uses the best model (e.g., `prototype_model_stacking.pkl`) to print real-time predictions.
- `labelling_CSV_*.py`: Utility scripts used to parse raw CSV data and add `failure` labels to generate the training dataset.
- `master_labeled_data.csv`: The primary dataset used for 5-fold CV training.
- `plots/` & `confusion_matrices/`: Directories where the training script automatically saves visual outputs (e.g., ROC curves, correlation heatmaps, class distribution).
- `*.pkl`: Serialized machine learning models and scalers (e.g., `prototype_scaler.pkl`) saved after training.

## ⚙️ How to Use

### 1. Training the Models
If you want to re-train the models with new data or adjust hyperparameters, run the model training script. This script will evaluate all models, output metrics to the console, save plots, and generate the `.pkl` files.
```bash
python new_sensors_model.py
```

### 2. Auto-labeling New Data
If you have gathered new raw data from the ESP32 and need to label it for future training:
```bash
python labelling_CSV_V0.py
```
*(Make sure to adjust the `CSV_PATH`, `MODEL_PATH`, and `OUTPUT_PATH` inside the script).*

### 3. Running Real-Time Prediction
Once the ESP32 is running and connected via USB, you can start the real-time prediction script. 
1. Open `real_time_predict.py`.
2. Update the `SERIAL_PORT` variable to match your ESP32's port (e.g., `COM3` for Windows, `/dev/ttyUSB0` for Linux/Mac).
3. Ensure the `BAUD_RATE` matches the ESP32's setting (default `9600`).
4. Run the script:
```bash
python real_time_predict.py
```

The terminal will begin displaying live sensor readings alongside the scaled SCADA values, followed by the AI Prediction Status (`NORMAL ✅` or `ARIZALI ❌`).

## 📊 Model Evaluation
During training, the system evaluates models based on **Accuracy, Precision, Recall, and F1-Score**. 
It utilizes a **Stacking Classifier** combining MLP, Random Forest, Logistic Regression, and Decision Trees (along with boosting models if installed) to achieve the highest predictive accuracy and robustness against false positives.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
