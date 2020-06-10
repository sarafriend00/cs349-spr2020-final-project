import pandas as pd
import matplotlib.pyplot
import numpy as np

def load_csv_data(path_to_csv):
    df = pd.read_csv(path_to_csv)
    return df

def filter_by_attribute(df, attribute, value):
    return df[df[attribute] == value]

def get_cases_chronologically(df):
    cases = []
    labels = []
    for i in range(df.shape[0]):
        _cases = df.iloc[i, 11:]
        _labels = df.iloc[i, :11]
        cases.append(_cases)
        labels.append(_labels)
    
    cases = np.array(cases)
    labels = np.array(labels)
    return cases, labels

def get_lock_idx(df):
    idx = []
    labels = []
    for i in range(df.shape[0]):
        _idx = df.iloc[i, 2]
        _labels = df.iloc[i, 0]
        idx.append(_idx)
        labels.append(_labels)
    
    idx = np.array(idx)
    labels = np.array(labels)
    return idx, labels