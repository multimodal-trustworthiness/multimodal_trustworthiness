
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
--

## Dataset
Due to GitHub's storage limitations, only the test data is displayed in this repository. The complete dataset, including training, validation, and test data, has been uploaded to Google Drive. You can access the full dataset using the following link: [Google Drive Dataset](https://drive.google.com/drive/folders/1EpMjVBAh1d9Zh73QkjpChoufGOqxBgvt).


### Multimodal Features 
#### Vocal Features
- **Tool Used**: Covarep.
- **Features**: 74 vocal features per second.
- **Data Representation**: 135 seconds x 74 dimensions feature matrix.

#### Visual Features
- **Tool Used**: OpenFace 2.0.
- **Features**: 49 facial features per image.
- **Data Representation**: 135 seconds x 49 dimensions matrix.

#### Verbal Features
- **Tool Used**: AliNLP.
- **Features**: 200-dimensional vector sequence.
- **Model Details**: PyTorch with Adam optimizer.



## Usage
To run the analysis, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/multimodal-trustworthiness/multimodal_trustworthiness.
   ```
2. Navigate to the cloned directory.
3. Run the main script:
   ```bash
   python CTC_EF_LSTM (verbal_vocal_visual).py
   ```


## Code Structure
- `11 main models`: The main script that orchestrates the data loading,  pretrained model loading, and evaluation.
  
- `collect_data.py`: Contains functions for data loading and preprocessing.
  
- `pre_train_model`: Contains 11 pretrained models.
  


## Model Description
The codebase includes 11 different models, each designed to handle various modalities and fusion techniques in the context of measuring trustworthiness of microenterprises:

1. **No Fusion + Unimodal Data, LSTM Models (3 Models):**
   - `LSTM (verbal).py`: LSTM model for verbal data.
   - `LSTM (visual).py`: LSTM model for visual data.
   - `LSTM (vocal).py`: LSTM model for vocal data.

2. **Partial Fusion + Bimodal Data, Late Fusion LSTM Models (3 Models):**
   - `LF_LSTM (verbal_visual).py`: Late Fusion LSTM for verbal and visual data.
   - `LF_LSTM (verbal_vocal).py`: Late Fusion LSTM for verbal and vocal data.
   - `LF_LSTM (vocal_visual).py`: Late Fusion LSTM for vocal and visual data.

3. **Full Fusion + Bimodal Data, Early Fusion LSTM Models (3 Models):**
   - `CTC_EF_LSTM (verbal_visual).py`: Early Fusion LSTM for verbal and visual data.
   - `CTC_EF_LSTM (verbal_vocal).py`: Early Fusion LSTM for verbal and vocal data.
   - `CTC_EF_LSTM (vocal_visual).py`: Early Fusion LSTM for vocal and visual data.

4. **Partial Fusion + Trimodal Data, Late Fusion LSTM Model (1 Model):**
   - `LF_LSTM (verbal_vocal_visual).py`: Late Fusion LSTM for verbal, vocal, and visual data.

5. **Full Fusion + Trimodal Data, Early Fusion LSTM Model (1 Model):**
   - `CTC_EF_LSTM (verbal_vocal_visual).py`: Early Fusion LSTM for verbal, vocal, and visual data.

Each model is based on pretrained models and demonstrates the prediction results for different data modalities and fusion techniques. The models are specifically tailored for analyzing the trustworthiness of microenterprises using multimodal data.



## Evaluation Metrics
The code includes functions for evaluating the model performance:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Correlation Coefficient
- Accuracy
- F1 Score


