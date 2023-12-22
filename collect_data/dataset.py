import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch
from sklearn.preprocessing import minmax_scale
from torch.utils.data import DataLoader
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################
class Multimodal_Datasets(Dataset):

    def __init__(self, file_dir= None, dataset_path = None, split_type='train' ):
        dataset_path = os.path.join(file_dir , "trustworthiness_data.pkl")
        dataset = pickle.load(open(dataset_path, 'rb'))


        tem = dataset[split_type]['labels'].astype(np.float32)
        print(tem.shape, "  ", np.mean(tem))
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()

        #  assert whether considering control variables
        if 'control' in dataset[split_type].keys():
            self.control = dataset[split_type]['control'].astype(np.float32)
            for col_index in range(self.control.shape[1]):
                if col_index != 3:
                    self.control[np.isnan(self.control[:, col_index]), col_index] = self.control[np.isnan(self.control[:, col_index]) == False, col_index].mean()
                else:
                    self.control[np.isnan(self.control[:, col_index]), col_index] = 0
            self.control = minmax_scale(self.control)
            self.control = torch.tensor(self.control).cpu().detach()
        else:
            self.control = None


        self.meta_info = dataset[split_type]["info"] if 'info' in dataset[split_type].keys() else None
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None

        self.n_modalities = 3  # vision/ text/ audio

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        META = (0, 0, 0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        control_variable = torch.tensor([0.0]) if self.control is None else self.control[index]
        clip_info = (0, 0) if self.meta_info is None else (self.meta_info[index][0], self.meta_info[index][1])
        return X, Y, META, control_variable, clip_info


def get_data(file_dir,dataset, split):
    data_dir = os.path.join(file_dir, f"{dataset}_{split}.dt")
    if not os.path.exists(data_dir):
        print(f"Create New {dataset}_{split}.dt")
        print(data_dir, dataset, split)
        data = Multimodal_Datasets(file_dir, dataset, split )
        torch.save(data, data_dir)
    else:
        print(f"Find cached dataset {dataset}_{split}.dt")
        data = torch.load(data_dir)
    return data


