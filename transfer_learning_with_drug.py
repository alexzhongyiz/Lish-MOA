import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import datetime
import seaborn as sns
 
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import _WeightedLoss

import warnings
warnings.filterwarnings('ignore')



def transfer_learning_with_drug(model_path='transfer-learning-with-drug'):
    data_dir = '../input/lish-moa/'

    train_features = pd.read_csv(data_dir + 'train_features.csv')
    train_targets_scored = pd.read_csv(data_dir + 'train_targets_scored.csv')
    train_targets_nonscored = pd.read_csv(data_dir + 'train_targets_nonscored.csv')
    train_drug = pd.read_csv(data_dir + 'train_drug.csv')
    test_features = pd.read_csv(data_dir + 'test_features.csv')
    sample_submission = pd.read_csv(data_dir + 'sample_submission.csv')



    GENES = [col for col in train_features.columns if col.startswith('g-')]
    CELLS = [col for col in train_features.columns if col.startswith('c-')]

    for col in (GENES + CELLS):
        transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal")
        vec_len = len(train_features[col].values)
        vec_len_test = len(test_features[col].values)
        raw_vec = train_features[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)

        train_features[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
        test_features[col] = transformer.transform(test_features[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]



    SEED_VALUE = 42
    def seed_everything(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    seed_everything(seed=SEED_VALUE)



    n_comp = 600
    data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
    data2 = (PCA(n_components=n_comp, random_state=SEED_VALUE).fit_transform(data[GENES]))
    train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

    train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
    test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])

    # drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)


    # CELLS
    n_comp = 50

    data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
    data2 = (PCA(n_components=n_comp, random_state=SEED_VALUE).fit_transform(data[CELLS]))
    train2 = data2[:train_features.shape[0]]; test2 = data2[-test_features.shape[0]:]

    train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
    test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])

    # drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
    train_features = pd.concat((train_features, train2), axis=1)
    test_features = pd.concat((test_features, test2), axis=1)



    var_thresh = VarianceThreshold(0.8)
    data = train_features.append(test_features)
    data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

    train_features_transformed = data_transformed[ : train_features.shape[0]]
    test_features_transformed = data_transformed[-test_features.shape[0] : ]

    train_features = pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                  columns=['sig_id','cp_type','cp_time','cp_dose'])

    train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)

    test_features = pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                 columns=['sig_id','cp_type','cp_time','cp_dose'])

    test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)



    train = train_features.merge(train_targets_scored, on='sig_id')
    train = train.merge(train_targets_nonscored, on='sig_id')
    train = train.merge(train_drug, on='sig_id')
    train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    test = test_features[test_features['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)

    train = train.drop('cp_type', axis=1)
    test = test.drop('cp_type', axis=1)



    target_cols = [x for x in train_targets_scored.columns if x != 'sig_id']
    aux_target_cols = [x for x in train_targets_nonscored.columns if x != 'sig_id']
    all_target_cols = target_cols + aux_target_cols

    num_targets = len(target_cols)
    num_aux_targets = len(aux_target_cols)
    num_all_targets = len(all_target_cols)



    class MoADataset:
        def __init__(self, features, targets):
            self.features = features
            self.targets = targets

        def __len__(self):
            return (self.features.shape[0])

        def __getitem__(self, idx):
            dct = {
                'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
                'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)
            }

            return dct

    class TestDataset:
        def __init__(self, features):
            self.features = features

        def __len__(self):
            return (self.features.shape[0])

        def __getitem__(self, idx):
            dct = {
                'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
            }

            return dct

    def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
        model.train()
        final_loss = 0

        for data in dataloader:
            optimizer.zero_grad()
            inputs, targets = data['x'].to(device), data['y'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            final_loss += loss.item()

        final_loss /= len(dataloader)
        return final_loss

    def valid_fn(model, loss_fn, dataloader, device):
        model.eval()
        final_loss = 0
        valid_preds = []

        for data in dataloader:
            inputs, targets = data['x'].to(device), data['y'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            final_loss += loss.item()
            valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

        final_loss /= len(dataloader)
        valid_preds = np.concatenate(valid_preds)
        return final_loss, valid_preds

    def inference_fn(model, dataloader, device):
        model.eval()
        preds = []

        for data in dataloader:
            inputs = data['x'].to(device)

            with torch.no_grad():
                outputs = model(inputs)

            preds.append(outputs.sigmoid().detach().cpu().numpy())

        preds = np.concatenate(preds)
        return preds

    class SmoothBCEwLogits(_WeightedLoss):
        def __init__(self, weight=None, reduction='mean', smoothing=0.0):
            super().__init__(weight=weight, reduction=reduction)
            self.smoothing = smoothing
            self.weight = weight
            self.reduction = reduction

        @staticmethod
        def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
            assert 0 <= smoothing < 1

            with torch.no_grad():
                targets = targets * (1.0 - smoothing) + 0.5 * smoothing

            return targets

        def forward(self, inputs, targets):
            targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
                self.smoothing)
            loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

            if  self.reduction == 'sum':
                loss = loss.sum()
            elif  self.reduction == 'mean':
                loss = loss.mean()

            return loss

    class Model(nn.Module):
        def __init__(self, num_features, num_targets):
            super(Model, self).__init__()
            self.hidden_size = [1500, 1250, 1000, 750]
            self.dropout_value = [0.5, 0.35, 0.3, 0.25]

            self.batch_norm1 = nn.BatchNorm1d(num_features)
            self.dense1 = nn.Linear(num_features, self.hidden_size[0])

            self.batch_norm2 = nn.BatchNorm1d(self.hidden_size[0])
            self.dropout2 = nn.Dropout(self.dropout_value[0])
            self.dense2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])

            self.batch_norm3 = nn.BatchNorm1d(self.hidden_size[1])
            self.dropout3 = nn.Dropout(self.dropout_value[1])
            self.dense3 = nn.Linear(self.hidden_size[1], self.hidden_size[2])

            self.batch_norm4 = nn.BatchNorm1d(self.hidden_size[2])
            self.dropout4 = nn.Dropout(self.dropout_value[2])
            self.dense4 = nn.Linear(self.hidden_size[2], self.hidden_size[3])

            self.batch_norm5 = nn.BatchNorm1d(self.hidden_size[3])
            self.dropout5 = nn.Dropout(self.dropout_value[3])
            self.dense5 = nn.utils.weight_norm(nn.Linear(self.hidden_size[3], num_targets))

        def forward(self, x):
            x = self.batch_norm1(x)
            x = F.leaky_relu(self.dense1(x))

            x = self.batch_norm2(x)
            x = self.dropout2(x)
            x = F.leaky_relu(self.dense2(x))

            x = self.batch_norm3(x)
            x = self.dropout3(x)
            x = F.leaky_relu(self.dense3(x))

            x = self.batch_norm4(x)
            x = self.dropout4(x)
            x = F.leaky_relu(self.dense4(x))

            x = self.batch_norm5(x)
            x = self.dropout5(x)
            x = self.dense5(x)
            return x

    class LabelSmoothingLoss(nn.Module):
        def __init__(self, classes, smoothing=0.0, dim=-1):
            super(LabelSmoothingLoss, self).__init__()
            self.confidence = 1.0 - smoothing
            self.smoothing = smoothing
            self.cls = classes
            self.dim = dim

        def forward(self, pred, target):
            pred = pred.log_softmax(dim=self.dim)

            with torch.no_grad():
                true_dist = torch.zeros_like(pred)
                true_dist.fill_(self.smoothing / (self.cls - 1))
                true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

            return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

    class FineTuneScheduler:
        def __init__(self, epochs):
            self.epochs = epochs
            self.epochs_per_step = 0
            self.frozen_layers = []

        def copy_without_top(self, model, num_features, num_targets, num_targets_new):
            self.frozen_layers = []

            model_new = Model(num_features, num_targets)
            model_new.load_state_dict(model.state_dict())

            # Freeze all weights
            for name, param in model_new.named_parameters():
                layer_index = name.split('.')[0][-1]

                if layer_index == 5:
                    continue

                param.requires_grad = False

                # Save frozen layer names
                if layer_index not in self.frozen_layers:
                    self.frozen_layers.append(layer_index)

            self.epochs_per_step = self.epochs // len(self.frozen_layers)

            # Replace the top layers with another ones
            model_new.batch_norm5 = nn.BatchNorm1d(model_new.hidden_size[3])
            model_new.dropout5 = nn.Dropout(model_new.dropout_value[3])
            model_new.dense5 = nn.utils.weight_norm(nn.Linear(model_new.hidden_size[-1], num_targets_new))
            model_new.to(DEVICE)
            return model_new

        def step(self, epoch, model):
            if len(self.frozen_layers) == 0:
                return

            if epoch % self.epochs_per_step == 0:
                last_frozen_index = self.frozen_layers[-1]

                # Unfreeze parameters of the last frozen layer
                for name, param in model.named_parameters():
                    layer_index = name.split('.')[0][-1]

                    if layer_index == last_frozen_index:
                        param.requires_grad = True

                del self.frozen_layers[-1]  # Remove the last layer as unfrozen



    def process_data(data):
        data = pd.get_dummies(data, columns=['cp_time','cp_dose'])
        return data

    feature_cols = [c for c in process_data(train).columns if c not in all_target_cols]
    feature_cols = [c for c in feature_cols if c not in ['kfold', 'sig_id', 'drug_id']]
    num_features = len(feature_cols)



    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 24
    BATCH_SIZE = 128

    WEIGHT_DECAY = {'ALL_TARGETS': 1e-5, 'SCORED_ONLY': 3e-6}
    MAX_LR = {'ALL_TARGETS': 1e-2, 'SCORED_ONLY': 3e-3}
    DIV_FACTOR = {'ALL_TARGETS': 1e3, 'SCORED_ONLY': 1e2}
    PCT_START = 0.1
    num_fold = 10



    def run_training(fold_id, seed_id):
        seed_everything(seed_id)

        test_ = process_data(test)

        # Load the fine-tuned model with the best loss
        model = Model(num_features, num_targets)
        model.load_state_dict(torch.load(f"{model_path}/SCORED_ONLY_SEED{seed_id}_FOLD{fold_id}.pth"))
        model.to(DEVICE)

        # prediction
        x_test = test_[feature_cols].values
        testdataset = TestDataset(x_test)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

        predictions = np.zeros((len(test_), num_targets))
        predictions = inference_fn(model, testloader, DEVICE)

        return predictions

    def run_k_fold(NFOLDS, seed_id):
        predictions = np.zeros((len(test), len(target_cols)))

        for fold_id in range(NFOLDS):
            pred_ = run_training(fold_id, seed_id)
            predictions += pred_ / NFOLDS

        return predictions



    # Averaging on multiple SEEDS
    SEED = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    predictions = np.zeros((len(test), len(target_cols)))

    for seed_id in SEED:
        predictions_ = run_k_fold(num_fold, seed_id)
        predictions += predictions_ / len(SEED)

    test[target_cols] = predictions

    df_submission = pd.concat([test_features[['sig_id', 'cp_type']].set_index('sig_id'), test[['sig_id'] + target_cols].set_index('sig_id')], axis=1)
    df_submission = df_submission.fillna(0.0).drop(columns=['cp_type'])
    df_submission.index.name = 'sig_id'
    
    return df_submission