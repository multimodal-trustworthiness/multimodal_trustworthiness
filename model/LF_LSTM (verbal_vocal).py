# -*- coding=utf-8 -*-
import os
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
time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())


class TradLstm(nn.Module):

    def __init__(self, batch_size, input_size, hide_size, out_size=1, res_drop=0.0, bi=False, num_layers=1, use_cv=False):
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
        return res, last_state


class FusionModel(nn.Module):
    """three modality fusion model"""
    def __init__(self, batch_size, use_cv, modal_list):
        super(FusionModel, self).__init__()
        self.control = use_cv
        self.hide_size = 200
        self.bi = True
        self.coe = 1 if not self.bi else 2
        self.modal_list = modal_list
        self.modality = len(modal_list)
        self.verbal_model = TradLstm(batch_size=batch_size, input_size=200, hide_size=self.hide_size, num_layers=2,
                                     bi=self.bi, use_cv=False).cuda()
        self.vocal_model = TradLstm(batch_size=batch_size, input_size=74, hide_size=self.hide_size, num_layers=2,
                                    bi=self.bi, use_cv=False).cuda()
        self.visual_model = TradLstm(batch_size=batch_size, input_size=49, hide_size=self.hide_size, num_layers=2,
                                     bi=self.bi, use_cv=False).cuda()
        #  assert whether considering control variables
        if self.control:
            self.linear = nn.Linear(self.modality * self.hide_size * self.coe + 10, 1)
        else:
            self.linear = nn.Linear(self.modality * self.hide_size * self.coe, 1)

    def forward(self, verbal, vocal, visual, cv_attr):
        """combine predicts with three modality"""
        con_out = torch.zeros(size=[0])
        if "language" in self.modal_list:
            verbal_out, verbal_state = self.verbal_model(verbal, cv_attr)
            con_out = verbal_state
        if "audio" in self.modal_list:
            vocal_out, vocal_state = self.vocal_model(vocal, cv_attr)
            con_out = torch.cat((con_out, vocal_state), axis=-1) if con_out.size(0) > 0 else vocal_state
        if "vision" in self.modal_list:
            visual_out, visual_state = self.visual_model(visual, cv_attr)
            con_out = torch.cat((con_out, visual_state), axis=-1) if con_out.size(0) > 0 else visual_state

        all_state = con_out
        if self.control:
            all_state = torch.cat((all_state, cv_attr), axis=-1)
        res = self.linear(all_state)
        return res


def model_train(model, modal, train_loader, valid_loader, test_loader, lr=0.001):
    """
   train model
    """
    best_valid = 1000000
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5, verbose=True)
    epochs = 40
    sample = [f"lf_lstm_{modal}_{time_stamp}.pt", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    record_loss = {
        "loss": copy.deepcopy(sample), "mse": copy.deepcopy(sample), "mae": copy.deepcopy(sample),
        "corr": copy.deepcopy(sample), "acc": copy.deepcopy(sample), "f1_score": copy.deepcopy(sample)
    }
    for epoch in range(1, epochs+1):
        start_time = time.time()
        model.train()
        proc_loss, proc_size = 0, 0
        for idx_batch, (batch_x, batch_y, batch_meta, batch_cv, batch_info) in enumerate(train_loader):
            batch_start = time.time()
            sample_idx, language, audio, vision = batch_x
            if "LateFusion" not in modal:
                if modal == "language":
                    id_variable = language
                elif modal == "audio":
                    id_variable = audio
                else:
                    id_variable = vision
                eval_attr = batch_y.squeeze(-1)
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    with torch.cuda.device(0):
                        id_variable, eval_attr, cv_attr = id_variable.cuda(), eval_attr.cuda(), batch_cv.cuda()
                else:
                    cv_attr = batch_cv

                net = nn.DataParallel(model) if batch_y.shape[0] > 3 else model
                preds, _ = net(id_variable, cv_attr)
                mse_loss = loss_func(preds, eval_attr)
                mse_loss.backward()
                optimizer.step()
                batch_end = time.time()
                if idx_batch % 30 == 0:
                    print("Epoch {:2d} | Batch {:2d}/{:2d} | Time {:5.4f} sec | Train Loss {:5.4f}"
                          "".format(epoch, idx_batch, len(train_loader), batch_end-batch_start, mse_loss))
            else:
                eval_attr = batch_y.squeeze(-1)
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    with torch.cuda.device(0):
                        verbal, vocal, visual, eval_attr, cv_attr = language.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda(), batch_cv.cuda()
                else:
                    cv_attr = batch_cv
                net = nn.DataParallel(model) if batch_y.shape[0] > 3 else model
                combine_predict = net(verbal, vocal, visual, cv_attr)
                mse_loss = loss_func(combine_predict, eval_attr)
                mse_loss.backward()
                optimizer.step()
                batch_end = time.time()
                proc_loss += mse_loss.item() * batch_size
                proc_size += batch_size
                if idx_batch % 30 == 0:
                    print("Epoch {:2d} | Batch {:2d}/{:2d} | Time {:5.4f} sec | Train Loss {:5.4f}"
                          "".format(epoch, idx_batch, len(train_loader), batch_end-batch_start, proc_loss/proc_size))

        # evaluation and test
        val_loss, val_res, val_truth = evaluate(model, valid_loader, modal, criterion=loss_func, loader_name="valid")
        test_loss, _, _ = evaluate(model, test_loader, modal, criterion=loss_func, loader_name="test")
        scheduler.step(val_loss)
        record_loss["loss"].append(val_loss)

        valid_eval = eval_trustworthiness(val_res, val_truth, loader_name="valid_loader")
        for key in valid_eval.keys():
            record_loss[key].append(valid_eval.get(key))

        print("-"*50)
        duration = time.time() - start_time
        print("Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}".format(epoch, duration,
                                                                                               val_loss, test_loss))
        print("-" * 50)

        # save model if  model has a better performance
        if val_loss < best_valid:
            models_dir = os.path.join(os.getcwd(), "pre_train_model")
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            print(f"Save model at pre_train_models/lf_lstm_{modal}_{time_stamp}.pt!")
            save_model(name=f"lf_lstm_{modal}_{time_stamp}", model=model)
            best_valid = val_loss

    # recording the results of epochs
    with open(f"{Path.cwd()}\\pre_train_model\\record_val_loss.txt", "a+") as file:
        for key in record_loss.keys():
            item_str = key+"\t"
            for item in record_loss.get(key):
                item_str = item_str + str(item) + "\t"
            file.write(item_str[:-1] + "\n")
        file.close()

    # test
    model = load_model(f"lf_lstm_{modal}_{time_stamp}")
    _, results, truths = evaluate(model, test_loader, modal, criterion=loss_func, loader_name="test_loader")
    eval_trustworthiness(results, truths, loader_name="test")


def evaluate(model, t_loader, modal, criterion, loader_name):
    """
   evaluation model performance
    """
    model.eval()
    avg_loss, total_sample = 0.0, 0
    results, truths, meta_info = list(), list(), list()
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y, batch_meta, batch_cv, batch_info) in enumerate(t_loader):
            sample_idx, language, audio, vision = batch_x
            eval_attr = batch_y.squeeze(-1)
            if "LateFusion" not in modal:
                if "language" in modal:
                    id_variable = language
                elif "audio" in modal:
                    id_variable = audio
                else:
                    id_variable = vision

                if torch.cuda.is_available():
                    with torch.cuda.device(0):
                        id_variable, eval_attr, cv_attr = id_variable.cuda(), eval_attr.cuda(), batch_cv.cuda()
                else:
                    cv_attr = batch_cv

                net = nn.DataParallel(model) if id_variable.shape[0] > 3 else model
                preds, _ = net(id_variable, cv_attr)
            else:
                if torch.cuda.is_available():
                    with torch.cuda.device(0):
                        verbal, vocal, visual, eval_attr, cv_attr = language.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda(), batch_cv.cuda()
                else:
                    cv_attr = batch_cv

                net = nn.DataParallel(model) if verbal.shape[0] > 3 else model
                preds = net(verbal, vocal, visual, cv_attr)

            loss = criterion(preds, eval_attr).item() * language.size(0)
            avg_loss = avg_loss + loss
            total_sample = total_sample + language.size(0)
            results.append(preds)
            truths.append(eval_attr)
            meta_info.extend([batch_info[0][index]+"&"+batch_info[1][index] for index in range(len(batch_info[0]))])

    avg_loss = avg_loss / total_sample
    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths


def save_model(name, model):
    """
       save  pretrained model
    """
    model_dir = f"{Path.cwd()}\\pre_train_model\\{name}.pt"
    if os.path.exists(model_dir):
        os.remove(model_dir)
    torch.save(model, model_dir)


def load_model(name):
    """
    load   pretrained model
    """
    model_dir = f"{Path.cwd()}\\pre_train_model\\{name}.pt"
    if os.path.exists(model_dir):
        model = torch.load(model_dir)
    else:
        model = None
    return model


def eval_trustworthiness(results, truths, loader_name):
    """
    evaluation
    """
    print(f" -------------{loader_name}--------------")
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
    #   verval and vocal  modality
    model_name = "LateFusion_la"
    batch_size = 32
    data_dir = r"..\collect_data"
    train_data_loader = DataLoader(get_data(data_dir, "trust", "train"), batch_size=batch_size,
                                   shuffle=False, generator=torch.Generator(device="cuda"))
    valid_data_loader = DataLoader(get_data(data_dir, "trust", "valid"), batch_size=batch_size,
                                   shuffle=False, generator=torch.Generator(device="cuda"))
    test_data_loader = DataLoader(get_data(data_dir, "trust", "test"), batch_size=batch_size,
                                  shuffle=False, generator=torch.Generator(device="cuda"))

    # test
    model = load_model(model_name)
    loss_func = nn.MSELoss()
    _, results, truths = evaluate(model, test_data_loader, model_name, criterion=loss_func, loader_name="test_loader")
    eval_trustworthiness(results, truths, loader_name="test")
