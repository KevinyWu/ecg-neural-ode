# ECG Classification with Neural ODE

<img src="https://github.com/KevinyWu/KevinyWu/blob/main/images/neural_ode.png" alt="neural_ode" width="600"/>

## Data Preprocessing

Data comes from the [MIT-BIH](https://physionet.org/content/mitdb/1.0.0/) database and is prepared by this project: [GitHub](https://github.com/intsav/RealtimeArrhythmiaMonitoring).

1. Download train data: [link](https://docs.google.com/uc?export=download&id=1KIBxRB12tbEop02Dj_sLBuZvPgu3ua6e)
2. Download test data: [link](https://docs.google.com/uc?export=download&id=1epF6BHCrTUOrpILBUp4xg160guVy_Jsr)
3. Put them in `data/`

## Model Training and Evaluation

There are three models with a similar number of trainable parameters:

1. 1-block ResNet
2. Neural ODE
3. Deep Equilibrium Network

Simply run the Jupyter Notebook `ecg.ipynb`. Model training will take a while for the Neural ODE and DEQ models.

**Requirements**

```
numpy
pandas
scikit-learn
pytorch
torchinfo
torchdiffeq
```

## File Tree

```
python_experiments
│   ecg.ipynb - Notebook for training and testing models 
│   models.py - Model functions
│   utils.py - Other helper functions
└───data
        mitdb_360_test.csv - Training data
        mitdb_360_train.csv - Test data
```
