import pandas as pd
import numpy as np
from tqdm import tqdm
import torch as t
import zipfile
import os

def unzip_dataset(folder = '../dataset/'):
    if not os.path.isfile(folder + 'result.csv'):
        print("Extracting dataset...")
        assert os.path.isfile(folder + 'result.zip'), "result.zip file is absent, cannot extract"            
        with zipfile.ZipFile(folder + 'result.zip', 'r') as zip_ref:
            zip_ref.extractall(folder)
    else:
        print("Dataset already extracted")
    return True

def generate_labels(dataset:pd.DataFrame, fwd:int, thr:float)->np.array:
    mid_prices = dataset[['p_ask_1', 'p_bid_1']].mean(axis=1)
    mid_prices_cum = np.zeros(dataset.shape[0] - fwd)
    for i in tqdm(range(mid_prices_cum.shape[0])):
        mid_prices_cum[i] = mid_prices[i:i+fwd].sum()

    mid_prices_cum /= fwd
    mid_prices_cum = np.stack([mid_prices_cum, mid_prices[:-fwd]], axis=1)

    def categorize(x):
        m = x[0] / x[1]
        if m > 1 + thr: return 1
        if m < 1 - thr: return 0
        else: return 2

    labels = np.apply_along_axis(categorize, 1, mid_prices_cum)

    return labels


class Pipe:
    def __init__(self, window_size=500, window_step=1):
        self.mean = None
        self.w_size=window_size
        self.w_step=window_step
        self.columns = None

    def fit(self, x:pd.DataFrame):
        self.max = t.tensor(x.max(axis=0).to_numpy(), dtype=t.float32)
        self.columns = x.columns + ['mid_price']

    def transform(self, x:pd.DataFrame) -> t.tensor:
        x_t = t.tensor(x.to_numpy(), dtype=t.float32)
        x_t /= self.max
        x_t = t.concat([x_t,
                        x_t[:, [0,2]]
                       .mean(axis=1)
                       .unsqueeze(0).T]
                       , axis=1)
        x_t = x_t.unfold(0, self.w_size, self.w_step)
        x_t = x_t.permute(0,2,1)
        return x_t[:-1]

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
