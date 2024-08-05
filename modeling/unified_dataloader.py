"""
The unified dataloader for all models' dataset loading.

If you use code in this repository, please cite our paper as below. Many thanks.

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}

or

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619
"""

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def parse_delta(masks, seq_len, feature_num):
    """generate deltas from masks, used in BRITS"""
    deltas = []
    for h in range(seq_len):
        if h == 0:
            deltas.append(np.zeros(feature_num))
        else:
            deltas.append(np.ones(feature_num) + (1 - masks[h]) * deltas[-1])
    return np.asarray(deltas)


def fill_with_last_observation(arr):
    """namely forward-fill nan values
    https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    """
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    out = np.nan_to_num(out)  # if nan still exists then fill with 0
    return out


class LoadDataset(Dataset):
    def __init__(self, file_path, seq_len, feature_num, model_type):
        super(LoadDataset, self).__init__()
        self.file_path = file_path
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.model_type = model_type


class LoadValTestDataset(LoadDataset):
    """Loading process of val or test set"""

    def __init__(self, file_path, set_name, seq_len, feature_num, model_type):
        super(LoadValTestDataset, self).__init__(
            file_path, seq_len, feature_num, model_type
        )
        with h5py.File(self.file_path, "r") as hf:  # read data from h5 file
            self.X = hf[set_name]["X"][:]
            self.X_hat = hf[set_name]["X_hat"][:]
            self.missing_mask = hf[set_name]["missing_mask"][:]
            self.indicating_mask = hf[set_name]["indicating_mask"][:]

        # fill missing values with 0
        self.X = np.nan_to_num(self.X)
        self.X_hat = np.nan_to_num(self.X_hat)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.model_type in ["Transformer", "SAITS"]:
            sample = (
                torch.tensor(idx),
                torch.from_numpy(self.X_hat[idx].astype("float32")),
                torch.from_numpy(self.missing_mask[idx].astype("float32")),
                torch.from_numpy(self.X[idx].astype("float32")),
                torch.from_numpy(self.indicating_mask[idx].astype("float32")),
            )
        elif self.model_type in ["BRITS", "MRNN"]:
            forward = {
                "X_hat": self.X_hat[idx],
                "missing_mask": self.missing_mask[idx],
                "deltas": parse_delta(
                    self.missing_mask[idx], self.seq_len, self.feature_num
                ),
            }
            backward = {
                "X_hat": np.flip(forward["X_hat"], axis=0).copy(),
                "missing_mask": np.flip(forward["missing_mask"], axis=0).copy(),
            }
            backward["deltas"] = parse_delta(
                backward["missing_mask"], self.seq_len, self.feature_num
            )
            sample = (
                torch.tensor(idx),
                # for forward
                torch.from_numpy(forward["X_hat"].astype("float32")),
                torch.from_numpy(forward["missing_mask"].astype("float32")),
                torch.from_numpy(forward["deltas"].astype("float32")),
                # for backward
                torch.from_numpy(backward["X_hat"].astype("float32")),
                torch.from_numpy(backward["missing_mask"].astype("float32")),
                torch.from_numpy(backward["deltas"].astype("float32")),
                torch.from_numpy(self.X[idx].astype("float32")),
                torch.from_numpy(self.indicating_mask[idx].astype("float32")),
            )
        else:
            assert ValueError, f"Error model type: {self.model_type}"
        return sample


class LoadTrainDataset(LoadDataset):
    """Loading process of train set"""

    def __init__(
        self, file_path, seq_len, feature_num, model_type, masked_imputation_task
    ):
        super(LoadTrainDataset, self).__init__(
            file_path, seq_len, feature_num, model_type
        )
        self.masked_imputation_task = masked_imputation_task
        if masked_imputation_task:
            self.artificial_missing_rate = 0.2
            assert (
                0 < self.artificial_missing_rate < 1
            ), "artificial_missing_rate should be greater than 0 and less than 1"

        with h5py.File(self.file_path, "r") as hf:  # read data from h5 file
            self.X = hf["train"]["X"][:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        if self.masked_imputation_task:
            X = X.reshape(-1)
            indices = np.where(~np.isnan(X))[0].tolist()
            indices = np.random.choice(
                indices,
                round(len(indices) * self.artificial_missing_rate),
            )
            X_hat = np.copy(X)
            X_hat[indices] = np.nan  # mask values selected by indices
            missing_mask = (~np.isnan(X_hat)).astype(np.float32)
            indicating_mask = ((~np.isnan(X)) ^ (~np.isnan(X_hat))).astype(np.float32)
            X = np.nan_to_num(X)
            X_hat = np.nan_to_num(X_hat)
            # reshape into time series
            X = X.reshape(self.seq_len, self.feature_num)
            X_hat = X_hat.reshape(self.seq_len, self.feature_num)
            missing_mask = missing_mask.reshape(self.seq_len, self.feature_num)
            indicating_mask = indicating_mask.reshape(self.seq_len, self.feature_num)

            if self.model_type in ["Transformer", "SAITS"]:
                sample = (
                    torch.tensor(idx),
                    torch.from_numpy(X_hat.astype("float32")),
                    torch.from_numpy(missing_mask.astype("float32")),
                    torch.from_numpy(X.astype("float32")),
                    torch.from_numpy(indicating_mask.astype("float32")),
                )
            elif self.model_type in ["BRITS", "MRNN"]:
                forward = {
                    "X_hat": X_hat,
                    "missing_mask": missing_mask,
                    "deltas": parse_delta(missing_mask, self.seq_len, self.feature_num),
                }

                backward = {
                    "X_hat": np.flip(forward["X_hat"], axis=0).copy(),
                    "missing_mask": np.flip(forward["missing_mask"], axis=0).copy(),
                }
                backward["deltas"] = parse_delta(
                    backward["missing_mask"], self.seq_len, self.feature_num
                )
                sample = (
                    torch.tensor(idx),
                    # for forward
                    torch.from_numpy(forward["X_hat"].astype("float32")),
                    torch.from_numpy(forward["missing_mask"].astype("float32")),
                    torch.from_numpy(forward["deltas"].astype("float32")),
                    # for backward
                    torch.from_numpy(backward["X_hat"].astype("float32")),
                    torch.from_numpy(backward["missing_mask"].astype("float32")),
                    torch.from_numpy(backward["deltas"].astype("float32")),
                    torch.from_numpy(X.astype("float32")),
                    torch.from_numpy(indicating_mask.astype("float32")),
                )
            else:
                assert ValueError, f"Error model type: {self.model_type}"
        else:
            # if training without masked imputation task, then there is no need to artificially mask out observed values
            missing_mask = (~np.isnan(X)).astype(np.float32)
            X = np.nan_to_num(X)
            if self.model_type in ["Transformer", "SAITS"]:
                sample = (
                    torch.tensor(idx),
                    torch.from_numpy(X.astype("float32")),
                    torch.from_numpy(missing_mask.astype("float32")),
                )
            elif self.model_type in ["BRITS", "MRNN"]:
                forward = {
                    "X": X,
                    "missing_mask": missing_mask,
                    "deltas": parse_delta(missing_mask, self.seq_len, self.feature_num),
                }
                backward = {
                    "X": np.flip(forward["X"], axis=0).copy(),
                    "missing_mask": np.flip(forward["missing_mask"], axis=0).copy(),
                }
                backward["deltas"] = parse_delta(
                    backward["missing_mask"], self.seq_len, self.feature_num
                )
                sample = (
                    torch.tensor(idx),
                    # for forward
                    torch.from_numpy(forward["X"].astype("float32")),
                    torch.from_numpy(forward["missing_mask"].astype("float32")),
                    torch.from_numpy(forward["deltas"].astype("float32")),
                    # for backward
                    torch.from_numpy(backward["X"].astype("float32")),
                    torch.from_numpy(backward["missing_mask"].astype("float32")),
                    torch.from_numpy(backward["deltas"].astype("float32")),
                )
            else:
                assert ValueError, f"Error model type: {self.model_type}"
        return sample


class LoadDataForImputation(LoadDataset):
    """Load all data for imputation, we don't need do any artificial mask here,
    just input original data into models and let them impute missing values"""

    def __init__(self, file_path, set_name, seq_len, feature_num, model_type):
        super(LoadDataForImputation, self).__init__(
            file_path, seq_len, feature_num, model_type
        )
        with h5py.File(self.file_path, "r") as hf:  # read data from h5 file
            self.X = hf[set_name]["X"][:]
        self.missing_mask = (~np.isnan(self.X)).astype(np.float32)
        self.X = np.nan_to_num(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.model_type in ["Transformer", "SAITS"]:
            sample = (
                torch.tensor(idx),
                torch.from_numpy(self.X[idx].astype("float32")),
                torch.from_numpy(self.missing_mask[idx].astype("float32")),
            )
        elif self.model_type in ["BRITS", "MRNN"]:
            forward = {
                "X": self.X[idx],
                "missing_mask": self.missing_mask[idx],
                "deltas": parse_delta(
                    self.missing_mask[idx], self.seq_len, self.feature_num
                ),
            }

            backward = {
                "X": np.flip(forward["X"], axis=0).copy(),
                "missing_mask": np.flip(forward["missing_mask"], axis=0).copy(),
            }
            backward["deltas"] = parse_delta(
                backward["missing_mask"], self.seq_len, self.feature_num
            )
            sample = (
                torch.tensor(idx),
                # for forward
                torch.from_numpy(forward["X"].astype("float32")),
                torch.from_numpy(forward["missing_mask"].astype("float32")),
                torch.from_numpy(forward["deltas"].astype("float32")),
                # for backward
                torch.from_numpy(backward["X"].astype("float32")),
                torch.from_numpy(backward["missing_mask"].astype("float32")),
                torch.from_numpy(backward["deltas"].astype("float32")),
            )
        else:
            assert ValueError, f"Error model type: {self.model_type}"
        return sample


import os
import numpy as np
from torch.utils.data import DataLoader


import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
class UnifiedDataLoader:
    def __init__(
        self,
        dataset_path,
        seq_len,
        feature_num,
        model_type,
        batch_size=1024,
        num_workers=4,
        masked_imputation_task=False,
    ):
        self.dataset_path = dataset_path
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        self.masked_imputation_task = masked_imputation_task

        self.train_dataset, self.train_loader, self.train_set_size = None, None, None
        self.val_dataset, self.val_loader, self.val_set_size = None, None, None
        self.test_dataset, self.test_loader, self.test_set_size = None, None, None

    def load_data(self):
        data = np.load(self.dataset_path)
        self.train_dataset = data['X']  # Adjust based on the available keys
        self.val_dataset = data['X_holdout']
        self.test_dataset = data['X']
        self.train_mask = data['missing_mask']
        self.val_mask = data['indicating_mask']
        self.test_mask = data['missing_mask']
        print(f"train_dataset shape: {self.train_dataset.shape}")
        print(f"train_mask shape: {self.train_mask.shape}")
        print(f"val_dataset shape: {self.val_dataset.shape}")
        print(f"val_mask shape: {self.val_mask.shape}")
        print(f"test_dataset shape: {self.test_dataset.shape}")
        print(f"test_mask shape: {self.test_mask.shape}")

    def get_train_val_dataloader(self):
        self.load_data()  # Load data inside this function

        self.train_loader = DataLoader(
            list(zip(self.train_dataset, self.train_mask)),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.val_loader = DataLoader(
            list(zip(self.val_dataset, self.val_mask)),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.train_set_size = len(self.train_dataset)
        self.val_set_size = len(self.val_dataset)
        return self.train_loader, self.val_loader

    def get_test_dataloader(self):
        self.load_data()  # Load data inside this function

        self.test_loader = DataLoader(
            list(zip(self.test_dataset, self.test_mask)),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.test_set_size = len(self.test_dataset)
        return self.test_loader

    def prepare_dataloader_for_imputation(self, set_name):
        if set_name == "train":
            dataset = self.train_dataset
            mask = self.train_mask
        elif set_name == "val":
            dataset = self.val_dataset
            mask = self.val_mask
        elif set_name == "test":
            dataset = self.test_dataset
            mask = self.test_mask
        else:
            raise ValueError(f"Unknown set name: {set_name}")

        data_for_imputation = {
            'dataset': dataset,
            'mask': mask
        }
        dataloader_for_imputation = DataLoader(
            data_for_imputation, batch_size=self.batch_size, shuffle=False
        )
        return dataloader_for_imputation

    def prepare_all_data_for_imputation(self):
        train_set_for_imputation = self.prepare_dataloader_for_imputation("train")
        val_set_for_imputation = self.prepare_dataloader_for_imputation("val")
        test_set_for_imputation = self.prepare_dataloader_for_imputation("test")
        return train_set_for_imputation, val_set_for_imputation, test_set_for_imputation



class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_groups,
        n_group_inner_layers,
        d_time,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout,
        **kwargs
    ):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = kwargs["input_with_mask"]
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature
        self.param_sharing_strategy = kwargs["param_sharing_strategy"]
        self.MIT = kwargs["MIT"]
        self.device = kwargs["device"]

        if kwargs["param_sharing_strategy"] == "between_group":
            self.layer_stack = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        dropout,
                        **kwargs
                    )
                    for _ in range(n_group_inner_layers)
                ]
            )
        else:
            self.layer_stack = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        dropout,
                        **kwargs
                    )
                    for _ in range(n_groups)
                ]
            )

        self.embedding = nn.Linear(actual_d_feature, d_model)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        self.dropout = nn.Dropout(p=dropout)
        self.reduce_dim = nn.Linear(d_model, d_feature)

    def impute(self, inputs):
        X, masks = inputs["X"], inputs["missing_mask"]
        print(f"X shape: {X.shape}")
        print(f"masks shape: {masks.shape}")

        if X.dim() == 3 and masks.dim() == 3:
            input_X = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        else:
            raise ValueError("X and masks must be 3-dimensional")

        input_X = self.embedding(input_X)
        enc_output = self.dropout(self.position_enc(input_X))

        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        learned_presentation = self.reduce_dim(enc_output)
        imputed_data = (
            masks * X + (1 - masks) * learned_presentation
        )  # replace non-missing part with original data
        return imputed_data, learned_presentation

    def forward(self, inputs, stage):
        X, masks = inputs["X"], inputs["missing_mask"]
        imputed_data, learned_presentation = self.impute(inputs)
        reconstruction_MAE = masked_mae_cal(learned_presentation, X, masks)
        if (self.MIT or stage == "val") and stage != "test":
            imputation_MAE = masked_mae_cal(
                learned_presentation, inputs["X_holdout"], inputs["indicating_mask"]
            )
        else:
            imputation_MAE = torch.tensor(0.0)

        return {
            "imputed_data": imputed_data,
            "reconstruction_loss": reconstruction_MAE,
            "imputation_loss": imputation_MAE,
            "reconstruction_MAE": reconstruction_MAE,
            "imputation_MAE": imputation_MAE,
        }





class LoadTrainDataset:
    def __init__(self, data, seq_len, feature_num, model_type, masked_imputation_task):
        self.data = data
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.model_type = model_type
        self.masked_imputation_task = masked_imputation_task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class LoadValTestDataset:
    def __init__(self, data, set_type, seq_len, feature_num, model_type):
        self.data = data
        self.set_type = set_type
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.model_type = model_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class LoadDataForImputation:
    def __init__(self, data, set_name, seq_len, feature_num, model_type):
        self.data = data
        self.set_name = set_name
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.model_type = model_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

