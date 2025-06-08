import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_dataset(name):
    if name == 'cic_iot':
        df = pd.read_csv('data/CIC-IoT-2023.csv')
    elif name == 'bot_iot':
        df = pd.read_csv('data/Bot-IoT.csv')
    else:
        raise ValueError('Unknown dataset')
    labels = df['label'].apply(lambda x: 1 if x != 'Benign' else 0).values
    features = df.drop(columns=['label']).values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features, labels
