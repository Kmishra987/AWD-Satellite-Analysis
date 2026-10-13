# AWD Satellite Analysis

A satellite-based Alternate Wetting and Drying (AWD) monitoring prototype that processes Sentinel-1 and Sentinel-2 observations, extracts vegetation and water-related indicators, engineers temporal features, and explores integration with IoT water-level measurements.

## Project Overview

The goal is to develop a data-driven workflow for studying field water conditions using satellite observations and, eventually, calibrated IoT sensor measurements.

The current implementation focuses on satellite data processing, temporal feature engineering, and exploratory AWD transition indicators. An experimental Random Forest classifier has also been evaluated using simulated IoT data.

**Project status:** Research and prototype stage. Real-field validation is not yet complete.

## Key Features

- **Satellite data extraction:** Uses Google Earth Engine to retrieve Sentinel-1 and Sentinel-2 observations.
- **Optical indicators:** Processes NDVI and NDWI time series.
- **SAR indicators:** Processes Sentinel-1 VV and VH backscatter features.
- **Temporal feature engineering:** Calculates changes and rolling statistics while preventing temporal calculations from crossing long observation gaps.
- **AWD transition indicators:** Identifies potential changes in satellite observations for further investigation.
- **IoT data validation:** Inspects and validates water-level measurements obtained from the prototype.
- **Experimental machine learning:** Trains a Random Forest classifier on simulated IoT readings and compares it with the existing threshold rule.
- **Visualization:** Generates time-series plots for satellite features.

## Technology Stack

- Python
- Google Earth Engine
- Sentinel-1 SAR
- Sentinel-2 optical imagery
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Joblib

## Project Structure

```text
AWD-Satellite-Analysis/
├── data/
│   ├── farms/
│   ├── raw/
│   │   └── iot/
│   └── processed/
├── outputs/
│   ├── plots/
│   ├── models/
│   └── metrics/
├── src/
│   ├── extract_satellite_features.py
│   ├── feature_engineering.py
│   ├── awd_detection.py
│   ├── validate_iot_data.py
│   ├── iot_random_forest_prototype.py
│   └── compare_iot_baseline.py
├── main.py
└── README.md
```

The listed structure summarizes the main components; additional project files may also be present.

## Satellite Processing Workflow

1. Load the farm registry and area of interest.
2. Retrieve available satellite observations.
3. Extract NDVI and NDWI from Sentinel-2 and VV/VH backscatter from Sentinel-1.
4. Organize observations by farm and date.
5. Validate dates, duplicates, and missing measurements.
6. Engineer temporal features without calculating transitions across long observation gaps.
7. Generate exploratory AWD-related transition indicators.
8. Save processed data and time-series plots.

## IoT Prototype Experiment

The current IoT dataset contains 79 readings from a Wokwi simulation. The target variable is `awd_status`, with 44 readings in class 0 and 35 in class 1.

An experimental Random Forest uses these input features:

- `water_depth_cm`
- `water_level_percent`
- `sensor_value`

The model was evaluated using a chronological 80/20 holdout split: 63 training readings and 16 test readings.

### Baseline Comparison

The prototype uses a water-depth threshold of −20 cm, assuming negative values represent the water level below the soil surface.

| Metric | Threshold baseline | Random Forest |
|---|---:|---:|
| Holdout accuracy | 100% | 100% |
| Balanced accuracy | 100% | 100% |
| Correct test readings | 16/16 | 16/16 |

Both approaches made identical predictions on the 16 test readings. The inspected dataset's AWD labels also agreed perfectly with the −20 cm threshold.

These results show agreement with the existing prototype rule, not independent evidence that machine learning improves AWD detection.

## Important Limitations

- The IoT readings are simulated in Wokwi, not collected from a validated physical field deployment.
- The dataset is small and contains repeated readings from a single prototype source.
- The AWD target appears to be derived from the existing threshold rule.
- Sensor measurements may require calibration when the prototype is transferred to physical hardware.
- Satellite transition indicators are exploratory and are not independently verified AWD labels.
- Satellite observations and IoT readings have not yet been established as synchronized ground-truth pairs.
- The current results do not demonstrate generalization to unseen farms or real-world sensor hardware.

## Next Steps

1. Calibrate physical sensor readings and verify the water-depth sign convention.
2. Collect field observations with independently verified AWD status.
3. Synchronize field measurements with satellite observation dates.
4. Evaluate the satellite indicators against verified field labels.
5. Compare simple threshold-based methods with machine-learning models using appropriate farm-level or time-based validation.
6. Document limitations and evaluation metrics for reproducibility.

## Reproducibility

Activate the project's Python environment or run its interpreter directly.

For example, on Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe main.py
```

To run the experimental IoT classifier:

```powershell
.\.venv\Scripts\python.exe src\iot_random_forest_prototype.py
```

To compare the threshold baseline and the trained Random Forest:

```powershell
.\.venv\Scripts\python.exe src\compare_iot_baseline.py
```

Google Earth Engine authentication and access must be configured before running satellite extraction.

## Conclusion

This project establishes a prototype workflow for satellite time-series processing, temporal feature engineering, exploratory AWD indicators, and IoT experiment evaluation. The next major milestone is validating the workflow with calibrated sensors and independently verified field observations.