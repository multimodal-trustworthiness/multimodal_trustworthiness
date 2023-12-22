# -*- coding=utf-8 -*-
import os
import json
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
time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())

class TradLstm(nn.Module):

    def __init__(self, batch_size, input_size, hide_size, out_size=1, res_drop=0.1, bi=False, num_layers=1,
                 use_cv=False, layer_dropout=0.1):
        super(TradLstm, self).__init__()
        self.layer_drop = layer_dropout
        self.batch_size = batch_size
        self.res_drop = res_drop
        self.hide_size = hide_size
        self.num_layers = num_layers
        self.bi = bi
        self.use_cv = use_cv
        if self.bi:
            self.lstm_model = nn.LSTM(input_size, hide_size, num_layers, bidirectional=True,
                                      batch_first=True, dropout=self.layer_drop)
 
        else:
            self.lstm_model = nn.LSTM(input_size, hide_size, num_layers, batch_first=True, dropout=self.layer_drop)
           

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


def train(args, train_loader, valid_loader, test_loader, time_stamp):
    """
    train model
    """
    language_align_audio = CTCModule(input_size=200, out_len=200).cuda()
    if torch.cuda.is_available():
        language_align_audio = language_align_audio.cuda()

    #   CTC  loss and optimizer
    ctc_criterion = nn.CTCLoss()
    l2a_optimizer = torch.optim.Adam(language_align_audio.parameters(), lr=args.get("lr", 0.001))

    # lr  optimizer
    model = TradLstm(
        batch_size=args.get("batch_size", 12),
        input_size=323,
        hide_size=args.get("hide_size", 200),
        bi=args.get("bi", True),
        num_layers=args.get("num_layers", 2),
        use_cv=args.get("use_cv", False),
        layer_dropout=args.get("layer_dropout", 0.1)
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.get("lr", 0.001))
    loss_func = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.get("patience", 4), factor=args.get("factor", 0.5), verbose=True
    )
    epochs = args.get("epochs", 20)
    best_valid = 100000
    sample = [f"ctc_ef_lstm_{time_stamp}.pt", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]
    record_loss = {
        "loss": copy.deepcopy(sample), "mse": copy.deepcopy(sample), "mae": copy.deepcopy(sample),
        "corr": copy.deepcopy(sample), "acc": copy.deepcopy(sample), "f1_score": copy.deepcopy(sample)
    }
    for epoch in range(1, 1+epochs):
        start_time = time.time()
        model.train()
        language_align_audio.train()
        for idx_batch, (batch_x, batch_y, batch_meta, batch_cv, batch_info) in enumerate(train_loader):
            batch_start = time.time()
            sample_idx, language, audio, vision = batch_x
            eval_attr = batch_y.squeeze(-1)
            if torch.cuda.is_available():
                with torch.cuda.device(0):
                    language, audio, vision, eval_attr, cv_attr = language.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda(), batch_cv.cuda()
            else:
                cv_attr = batch_cv

         
            model.zero_grad()
            language_align_audio.zero_grad()

            #  ctc loss
            ctc_language_audio = nn.DataParallel(language_align_audio)
            language, a2l_position = ctc_language_audio(language)

            l_len, a_len = 300, 200
            bs = a2l_position.shape[0]
            # Output Labels
            target = torch.tensor([i + 1 for i in range(a_len)] * bs).int().cpu()
            # Specifying each output length
            target_length = torch.tensor([a_len] * bs).int().cpu()
            # Specifying each input length
            a_length = torch.tensor([l_len] * bs).int().cpu()

            ctc_loss = ctc_criterion(a2l_position.transpose(0, 1).log_softmax(2), target, a_length, target_length)
            ctc_loss = ctc_loss.cuda() if torch.cuda.is_available() else ctc_loss

            # features concat
            vision = vision[:, :200, :]
            combine_input = torch.cat((language, audio, vision), dim=2)
            combine_input = combine_input.cuda() if torch.cuda.is_available() else combine_input
            net = nn.DataParallel(model)
            preds = net(combine_input, cv_attr)
            model_loss = loss_func(preds, eval_attr)
            combine_loss = model_loss + ctc_loss
            combine_loss.backward()

            torch.nn.utils.clip_grad_norm_(language_align_audio.parameters(), 0.8)
            l2a_optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            optimizer.step()
            batch_end = time.time()
            if idx_batch % 30 == 0:
                print("Epoch {:2d} | Batch {:2d}/{:2d} | Time {:5.4f} sec | "
                      "Train Loss {:5.4f}".format(epoch, idx_batch, len(train_loader),
                                                  batch_end - batch_start, combine_loss))
        # evaluation and test
        val_loss, val_res, val_truth, _ = evaluate(
            model,
            language_align_audio,
            valid_loader,
            criterion=loss_func,
            ctc_criterion=nn.CTCLoss()
        )
        test_loss, test_res, test_truth, _ = evaluate(
            model,
            language_align_audio,
            test_loader,
            criterion=loss_func,
            ctc_criterion=nn.CTCLoss()
        )
        scheduler.step(val_loss)
        record_loss["loss"].append(val_loss)
        valid_eval = eval_trustworthiness(val_res, val_truth)
        for key in valid_eval.keys():
            record_loss[key].append(valid_eval.get(key))

        eval_trustworthiness(test_res, test_truth)
 
        print("-" * 50)
        duration = time.time() - start_time
        print("Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}".format(epoch, duration,
                                                                                               val_loss, test_loss))
        print("-" * 50)

        # save model if  model has a better performance
        if val_loss < best_valid:
            models_dir = os.path.join(os.getcwd(), "pre_train_model")
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            print(f"Save model at pre_train_models/ctc_ef_lstm_lva.pt!")
            save_model(name=f"ctc_ef_lstm_lva{time_stamp}", model=model)
            best_valid = val_loss

            print(f"Save model at pre_train_models/ctc_model.pt!")
            save_model(name=f"ctc_ef_lstm_lva_{time_stamp}", model=language_align_audio)

    # recording the results of epochs
    with open(f"{Path.cwd()}\\pre_train_model\\record_val_loss.txt", "a+") as file:
        for key in record_loss.keys():
            item_str = key + "\t"
            for item in record_loss.get(key):
                item_str = item_str + str(item) + "\t"
            file.write(item_str[:-1] + "\n")
        file.close()

    # test
    val_loss, val_res, val_truth, _ = evaluate(
        model,
        language_align_audio,
        valid_loader,
        criterion=loss_func,
        ctc_criterion=nn.CTCLoss()
    )
    test_loss, test_res, test_truth, _ = evaluate(
        model,
        language_align_audio,
        test_loader,
        criterion=loss_func,
        ctc_criterion=nn.CTCLoss()
    )
    # ctc_model_path = f"ctc_model_{time_stamp}.pt"
    # language_align_audio = load_model(ctc_model_path)
    # mode_path = f"ctc_ef_lstm_{time_stamp}.pt"
    # model = load_model(mode_path)
    # _, results, truths, _ = evaluate(
    #     model,
    #     language_align_audio,
    #     test_loader,
    #     criterion=loss_func,
    #     ctc_criterion=nn.CTCLoss()
    # )
    test_eval = eval_trustworthiness(test_res, test_truth)
    return test_eval, f"ctc_model_{time_stamp}", f"ctc_ef_lstm_{time_stamp}"


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

            vision = vision[:, :514, :]
            combine_input = torch.cat((language, audio, vision), dim=2)
            combine_input = combine_input.cuda() if torch.cuda.is_available() else combine_input
            net = nn.DataParallel(model)
            preds = net(combine_input, cv_attr)
            # tem = criterion(preds, eval_attr)
            loss = criterion(preds, eval_attr).item() * combine_input.shape[0]
            avg_loss = avg_loss + loss
            total_sample = total_sample + combine_input.shape[0]
            item_preds = preds.view(-1).cpu().detach().numpy()
            item_attr = eval_attr.view(-1).cpu().detach().numpy()
            for i in range(preds.shape[0]):
                meta_info.append((batch_info[0][i], batch_info[1][i], item_preds[i], item_attr[i]))
            results.append(preds)
            truths.append(eval_attr)

    avg_loss = avg_loss / total_sample
    results = torch.cat(results)
    truths = torch.cat(truths)
    return avg_loss, results, truths, meta_info


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
    load  pretrained model
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
    seed = 111
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    args_parameter = {
        "batch_size": 16,
        "hide_size": 200,
        "bi": True,
        "num_layers": 2,
        "use_cv": False,
        "layer_dropout": 0,
        "patience": 4,
        "factor": 0.5,
        "lr": 0.001,
        "epochs": 40
    }

    data_dir = r"..\collect_data"
    train_data_loader = DataLoader(get_data(data_dir,"trust", "train"),
                                   batch_size=args_parameter.get("batch_size"),
                                   shuffle=False, generator=torch.Generator(device="cuda"))
    valid_data_loader = DataLoader(get_data(data_dir,"trust", "valid"),
                                   batch_size=args_parameter.get("batch_size"),
                                   shuffle=False, generator=torch.Generator(device="cuda"))
    test_data_loader = DataLoader(get_data(data_dir,"trust", "test"),
                                  batch_size=args_parameter.get("batch_size"),
                                  shuffle=False, generator=torch.Generator(device="cuda"))
    for i in range(10):
       print(i)
       test_res, ctc_model_name, ef_model_name = train(args_parameter, train_data_loader,
                                                    valid_data_loader, test_data_loader,time_stamp)


