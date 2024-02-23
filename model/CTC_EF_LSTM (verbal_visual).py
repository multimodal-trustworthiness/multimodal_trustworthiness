# -*- coding=utf-8 -*-
import os
import sys
import time
import copy
import numpy as np
import torch
from torch import nn
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


class CTCModule(nn.Module):

    def __init__(self, input_size, out_len):
        super(CTCModule, self).__init__()
        self.pred_output_blank = nn.LSTM(input_size=input_size, hidden_size=out_len+1, num_layers=2, batch_first=True)

        self.out_len = out_len
        self.softmax = nn.Softmax(dim=2)

    def forward(self, inputs):
        last_state, _ = self.pred_output_blank(inputs)
        last_state_include_blank = self.softmax(last_state)
        position = last_state_include_blank[:, :, 1:]
        position = position.transpose(1, 2)
        align_out = torch.bmm(position, inputs)
        # last_state = torch.log_softmax(last_state, dim=2)
        return align_out, last_state



def evaluate(model, ctc_model, t_loader, criterion, ctc_criterion=None):
    """
       evaluation model performance
    """

    model.eval()
    ctc_model.eval()


    avg_loss, total_sample = 0.0, 0
    results, truths, meta_info = list(), list(), list()
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y, batch_meta, batch_cv, batch_info) in enumerate(t_loader):
            sample_idx, language, audio, vision = batch_x
            eval_attr = batch_y.squeeze(-1)

            if torch.cuda.is_available():
                with torch.cuda.device(0):
                    language, audio, vision, eval_attr, cv_attr = language.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda(), batch_cv.cuda()
            else:
                cv_attr = batch_cv

            ctc_l2a_net = nn.DataParallel(ctc_model)
            language, _ = ctc_l2a_net(language)  # audio aligned to text

            combine_input = torch.cat((language, vision), dim=2)
            combine_input = combine_input.cuda() if torch.cuda.is_available() else combine_input
            net = nn.DataParallel(model)
            preds = net(combine_input, cv_attr)
            # tem = criterion(preds, eval_attr)
            loss = criterion(preds, eval_attr).item() * combine_input.shape[0]
            avg_loss = avg_loss + loss
            total_sample = total_sample + combine_input.shape[0]
            item_preds = preds.view(-1).cpu().detach().numpy()
            for i in range(preds.shape[0]):
                meta_info.append((batch_info[0][i], batch_info[1][i], item_preds[i]))
            results.append(preds)
            truths.append(eval_attr)

    avg_loss = avg_loss / total_sample
    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths, meta_info




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
if __name__ == "__main__":

    seed = 1234

    #load data
    data_dir = r"..\collect_data"
    test_data_loader = DataLoader(get_data(data_dir, "trust", "test"), batch_size=32,
                                  shuffle=False, generator=torch.Generator(device="cuda"))

    # load pretrained model
    ctc_model = load_model(f"EarlyFusion_lstm_ctc_lv")
    model = load_model(f"EarlyFusion_lstm_lv")

    # evaluation
    _, results, truths, _ = evaluate(model, ctc_model, test_data_loader, criterion=nn.MSELoss(), ctc_criterion=None)
    eval_trustworthiness(results, truths)
