### Intstalling neccessary packages:
1. Download and install anaconda
2. Install required packages using
    - conda env create -f environment.yml
    - conda activate ml_env

### Main scripts:
1. Use **sensor_values_prep.ipynb** to download sensor values from prestodb and download weather data from data.gov

2. Use **time_series_ml.ipynb** to train and test deep neural network

### Other notes:
1. Trained model weights are saved in models/
2. Min max scaler fitted on train dataset is saved in minmaxscaler/
3. Python scripts are stored in scripts/
4. Datasets are stored in/ downloaded to dataset/