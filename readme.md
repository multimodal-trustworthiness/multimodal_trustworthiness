
## Introducing Machine-Learning-Based Data Fusion Methods for Analyzing Multimodal Data: An Application of Measuring Trustworthiness of Microenterprises

## Overview
This repository contains the code and data used in the research paper "Introducing Machine-Learning-Based Data Fusion Methods for Analyzing Multimodal Data: An Application of Measuring Trustworthiness of Microenterprises". 

## Installation
Before running the scripts, ensure you have the following dependencies installed:
- Python 3.x
- PyTorch
- Pandas
- Numpy
- Pickle
- Statsmodels
- Scikit-learn

You can install these packages using pip:
```bash
pip install torch pandas statsmodels numpy scikit-learn
```

## Dataset
Due to GitHub's storage limitations, only the test data is displayed in this repository. The complete dataset, including training, validation, and test data, has been uploaded to Google Drive. You can access the full dataset using the following link: [Google Drive Dataset](https://drive.google.com/drive/folders/1EpMjVBAh1d9Zh73QkjpChoufGOqxBgvt).



## Usage
To run the analysis, follow these steps:
1. Clone the repository:
   ```bash
   git clone [repository URL]
   ```
2. Navigate to the cloned directory.
3. Run the main script:
   ```bash
   python main.py
   ```

## Code Structure
- `11 main models`: The main script that orchestrates the data loading,  pretrained model loading, and evaluation,including the LSTM-based model for early fusion of vocal and visual modalities.
- `collect_data.py`: Contains functions for data loading and preprocessing.
- `pre_train_model`: Contains the pretrained models of 11 models.
  

## Model Description
The codebase includes 11 different models, each designed to handle various modalities and fusion techniques in the context of measuring trustworthiness of microenterprises:

1. **Single Modality LSTM Models (3 Models):**
   - `LSTM (verbal).py`: LSTM model for verbal data.
   - `LSTM (visual).py`: LSTM model for visual data.
   - `LSTM (vocal).py`: LSTM model for vocal data.

2. **Dual Modality without Fusion - Late Fusion LSTM Models (3 Models):**
   - `LF LSTM (verbal visual).py`: Late Fusion LSTM for verbal and visual data.
   - `LF LSTM (verbal vocal).py`: Late Fusion LSTM for verbal and vocal data.
   - `LF LSTM (vocal visual).py`: Late Fusion LSTM for vocal and visual data.

3. **Dual Modality with Fusion - Early Fusion LSTM Models (3 Models):**
   - `CTC EF LSTM (verbal visual).py`: Early Fusion LSTM for verbal and visual data.
   - `CTC EF LSTM (verbal vocal).py`: Early Fusion LSTM for verbal and vocal data.
   - `CTC EF LSTM (vocal visual).py`: Early Fusion LSTM for vocal and visual data.

4. **Tri Modality without Fusion - Late Fusion LSTM Model (1 Model):**
   - `LF LSTM (verbal vocal visual).py`: Late Fusion LSTM for verbal, vocal, and visual data.

5. **Tri Modality with Fusion - Early Fusion LSTM Model (1 Model):**
   - `CTC EF LSTM (verbal vocal visual).py`: Early Fusion LSTM for verbal, vocal, and visual data.

Each model is based on pretrained models and demonstrates the prediction results for different data modalities and fusion techniques. The models are specifically tailored for analyzing the trustworthiness of microenterprises using multimodal data.


## Evaluation Metrics
The code includes functions for evaluating the model performance:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Correlation Coefficient
- Accuracy
- F1 Score


