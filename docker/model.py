import torch as t
from src.models import Att_GRU
import pandas as pd
from src.preprocess import generate_labels
from src.utils import train, eval, save_model
import numpy as np
import pickle
import os

class Predictor():
    def __init__(self, path_to_model, path_to_eval_data, max_samples, client=None):
        model_dict = t.load(path_to_model, map_location=t.device('cpu'))
        self.path_to_model = path_to_model
        self.model = model_dict['model']
        self.params = model_dict['params']
        self.max_samples = max_samples
        if os.path.isfile(path_to_eval_data):
            self.eval_data = t.load(path_to_eval_data)
        else: 
            self.eval_data = t.utils.data.TensorDataset(t.tensor([]), t.tensor([], dtype=t.int64))
        self.pipe = model_dict['pipe']
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=1e-1)
        self.path_to_eval_data = path_to_eval_data
        self.loss_f = t.nn.CrossEntropyLoss()
        self.client = client

    def predict(self, data: t.tensor):
        loader = t.utils.data.TensorDataset(data)
        loader = t.utils.data.DataLoader(loader, batch_size=self.params['BATCH_SIZE'])
        res = t.tensor([], dtype=t.int64)

        for X in loader:
            with t.no_grad():
                out = t.argmax(self.model(X[0]), dim=1)
                res = t.cat((res, out), dim=0)

        return res




    def train(self, train_data: pd.DataFrame, epochs=5):
        if 'Unnamed: 0' in train_data.columns:
            print('deleting 0 column')
            train_data.drop(['Unnamed: 0'], axis=1, inplace=True)

        train_end = int(self.params['TRAIN_TEST_R'] * train_data.shape[0])
        train_df = train_data.iloc[:train_end]
        test_df = train_data.iloc[train_end:]

        label_train_offset = (self.params['LOOK_FWD'] - self.params['W_SIZE']) // 2
        train_df_t = self.pipe.transform(train_df)[:-(label_train_offset)]
        test_df_t = self.pipe.transform(test_df)[:-(label_train_offset)]

        train_l = generate_labels(train_df, self.params['LOOK_FWD'], self.params['THR']
                                 )[:int(train_df_t.shape[0] * self.params['W_STEP']):self.params['W_STEP']]
        test_l = generate_labels(test_df, self.params['LOOK_FWD'], self.params['THR']
                                 )[:int(test_df_t.shape[0] * self.params['W_STEP']):self.params['W_STEP']]

        eval_index_start = int(np.max([self.max_samples - test_df_t.shape[0], 0]))

        eval_t = self.eval_data.tensors[0][-eval_index_start:]
        eval_l = self.eval_data.tensors[1][-eval_index_start:]

        test_df_t = t.cat((eval_t, test_df_t), dim=0)
        test_l = t.cat((eval_l, t.tensor(test_l, dtype=t.int64)), dim=0)

        train_ds = t.utils.data.TensorDataset(train_df_t, t.tensor(train_l, dtype=t.int64))
        test_ds = t.utils.data.TensorDataset(test_df_t, test_l)

        train_ldr = t.utils.data.DataLoader(train_ds, batch_size=64)
        test_ldr = t.utils.data.DataLoader(test_ds, batch_size=64)

        train(train_ldr, test_ldr, self.model, self.optimizer, self.loss_f, epochs)
        results = eval(test_ldr, self.model, self.loss_f)

        t.save(test_ds, self.path_to_eval_data)
        save_model(self.model, self.params, results, self.pipe, self.path_to_model, generate_name=False)
        print(results)

        return results

    def save(self, path_model):
        pass
    
    @staticmethod
    def load_dataset(path):
        return t.load(path)


