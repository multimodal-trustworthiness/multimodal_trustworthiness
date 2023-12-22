# -*- coding=utf-8 -*-
import os
import sys
import copy
import pandas
import statsmodels.api as sm
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import DataLoader
from collect_data.dataset import get_data
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path


class TradLstm(nn.Module):

    def __init__(self, batch_size, input_size, hide_size, out_size=1, res_drop=0.1, bi=False, num_layers=1, use_cv=False):
        super(TradLstm, self).__init__()

        self.batch_size = batch_size
        self.res_drop = res_drop
        self.hide_size = hide_size
        self.num_layers = num_layers
        self.bi = bi
        self.use_cv = use_cv
        if self.bi:
            self.lstm_model = nn.LSTM(input_size, hide_size, num_layers, bidirectional=True, batch_first=True, dropout=0.1)

        else:
            self.lstm_model = nn.LSTM(input_size, hide_size, num_layers, batch_first=True, dropout=0.1)


        self.proj = nn.Linear(hide_size * 2, hide_size * 2) if bi else nn.Linear(hide_size, hide_size)

        #  assert whether considering control variables
        if not self.use_cv:
            self.linear = nn.Linear(hide_size * 2, out_size) if bi else nn.Linear(hide_size, out_size)
        else:
            self.linear = nn.Linear(hide_size * 2+10, out_size) if bi else nn.Linear(hide_size+10, out_size)

    def forward(self, inputs, controls):

        bs = inputs.shape[0]
        if self.bi:
            init_hide_state = torch.zeros(self.num_layers * 2, bs, self.hide_size)
            init_cell_state = torch.zeros(self.num_layers * 2, bs, self.hide_size)
        else:
            init_hide_state = torch.zeros(self.num_layers, bs, self.hide_size)
            init_cell_state = torch.zeros(self.num_layers, bs, self.hide_size)
        model_out, _ = self.lstm_model(inputs, (init_hide_state, init_cell_state))
        last_state = model_out[:, -1, :]
        ls_state = func.dropout(self.proj(last_state), self.res_drop, self.training)
        last_state = last_state + ls_state
        #  assert whether considering control variables
        if not self.use_cv:
            res = self.linear(last_state)
        else:
            last_state = torch.cat((last_state, controls), dim=1)
            res = self.linear(last_state)
        return res


def evaluate(model, t_loader, criterion):
    """
       evaluation model performance
    """
    model.eval()
    avg_loss, total_sample = 0.0, 0
    results, truths = list(), list()
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y, batch_meta, batch_cv, batch_info) in enumerate(t_loader):
            sample_idx, language, audio, vision = batch_x
            id_variable = torch.cat((audio, vision), axis=-1)
            eval_attr = batch_y.squeeze(-1)
            if torch.cuda.is_available():
                with torch.cuda.device(0):
                    id_variable, eval_attr, cv_attr = id_variable.cuda(), eval_attr.cuda(), batch_cv.cuda()
            else:
                cv_attr = batch_cv

            net = nn.DataParallel(model) if id_variable.shape[0] > 10 else model
            preds = net(id_variable, cv_attr)
            tem = criterion(preds, eval_attr)
            loss = criterion(preds, eval_attr).item() * id_variable.shape[0]
            avg_loss = avg_loss + loss
            total_sample = total_sample + id_variable.shape[0]
            results.append(preds)
            truths.append(eval_attr)

    avg_loss = avg_loss / total_sample
    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths


def load_model(name):
    """
    load pretrained model
    """
    model_dir = f"{Path.cwd()}\\pre_train_model\\{name}.pt"
    if os.path.exists(model_dir):
        model = torch.load(model_dir)
    else:
        model = None
    return model


def eval_trustworthiness(results, truths):
    """
    evaluation
    """
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()
    mse = np.mean((test_preds - test_truth)**2)
    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    print("MSE: ", mse)
    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)

    median_value = 4
    binary_truth = (test_truth > median_value)
    binary_preds = (test_preds > median_value)
    print("Accuracy: ", accuracy_score(binary_truth, binary_preds))
    f_score = f1_score((test_preds >= median_value), (test_truth >= median_value), average='weighted')
    print("f_score: ", f_score)
    print("-" * 50)
    return {
        "mse": mse, "mae": mae, "corr": corr,
        "acc": accuracy_score(binary_truth, binary_preds),
        "f1_score": f_score
    }


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


if __name__ == "__main__":
    # Early Fusion vocala and visual modality
    seed = 1234
    # load data
    data_dir = r"..\collect_data"
    test_data_loader = DataLoader(get_data(data_dir, "trust", "test"), batch_size=32,
                                  shuffle=False, generator=torch.Generator(device="cuda"))

    # load pretrained model
    model_name = "ef_lstm_va"
    model = load_model(model_name)

    # evaluation
    _, results, truths = evaluate(model, test_data_loader,  criterion=nn.MSELoss()  )
    eval_trustworthiness(results, truths)


