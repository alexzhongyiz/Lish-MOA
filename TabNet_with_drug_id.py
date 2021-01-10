# TabNet
#!pip install --no-index --find-links /kaggle/input/pytorchtabnet/pytorch_tabnet-2.0.0-py3-none-any.whl pytorch-tabnet
# Iterative Stratification
#!pip install /kaggle/input/iterative-stratification/iterative-stratification-master/

import os
import sys
import copy
import tqdm
import pickle
import random
import warnings
warnings.filterwarnings("ignore")
#sys.path.append("../input/rank-gauss")
sys.path.append(os.path.abspath("../input/inference-files/"))
sys.path.append(os.path.abspath("../input/models/"))
sys.path.append(os.path.abspath("../input/lish-moa/"))
sys.path.append(os.path.abspath("../input/rank-gauss/"))
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

### Data Wrangling ###
import numpy as np
import pandas as pd
from scipy import stats
from gauss_rank_scaler import GaussRankScaler
from LogitLoss import LogitsLogLoss

### Data Visualization ###
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

### Machine Learning ###
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

### Deep Learning ###
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
# Tabnet 
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor

### Make prettier the prints ###
from colorama import Fore
c_ = Fore.CYAN
m_ = Fore.MAGENTA
r_ = Fore.RED
b_ = Fore.BLUE
y_ = Fore.YELLOW
g_ = Fore.GREEN


def TabNet_with_drug_id():

    seed = 42

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    set_seed(seed)

    # Parameters
    data_path = "../input/lish-moa/"
    #data_path = "/content/drive/My Drive/lish-moa/"
    inference_path = "../input/inference-files/"
    model_path =  "../input/models/"
    guass_rank_path = "../input/rank-gauss/"
    cv_path =  "../input/cv-data/"
    no_ctl = True
    scale = "rankgauss"
    variance_threshould = 0.5
    decompo = "PCA"
    ncompo_genes = 80
    ncompo_cells = 20
    encoding = "dummy"

    train = pd.read_csv(data_path + "train_features.csv")
    #train.drop(columns = ["sig_id"], inplace = True)

    targets = pd.read_csv(data_path + "train_targets_scored.csv")
    #train_targets_scored.drop(columns = ["sig_id"], inplace = True)

    #train_targets_nonscored = pd.read_csv(data_path + "train_targets_nonscored.csv")

    test = pd.read_csv(data_path + "test_features.csv")
    #test.drop(columns = ["sig_id"], inplace = True)

    submission = pd.read_csv(data_path + "sample_submission.csv")

    # train_test_split from Depei
    train_cv_without_drug = pd.read_csv("../input/cv-data/cv-10fold-without-drug_id.csv")
    train_cv_with_drug = pd.read_csv("../input/cv-data/cv-10fold-with-drug_id.csv")
    #print("train_without-drug.shape", train_cv_without_drug.shape)

    train_with_drug = pd.merge(train,train_cv_with_drug, on = "sig_id")
    train_without_drug = pd.merge(train,train_cv_without_drug, on = "sig_id")


    use_drug = True
    if use_drug:
        train = train_with_drug
    else:
        train = train_without_drug

    if no_ctl:
        # cp_type == ctl_vehicle
        print(b_, "not_ctl")
        train = train[train["cp_type"] != "ctl_vehicle"]
        test = test[test["cp_type"] != "ctl_vehicle"]
        targets = targets.iloc[train.index]
        train.reset_index(drop = True, inplace = True)
        test.reset_index(drop = True, inplace = True)
        targets.reset_index(drop = True, inplace = True)

    GENES = [col for col in train.columns if col.startswith("g-")]
    CELLS = [col for col in train.columns if col.startswith("c-")] 


    # Rank Gaussian Process:

    data_all = pd.concat([train, test], ignore_index = True)
    cols_numeric = [feat for feat in list(data_all.columns) if feat not in ["sig_id", "cp_type", "cp_time", "cp_dose","drug_id","fold_id"]]
    mask = (data_all[cols_numeric].var() >= variance_threshould).values
    tmp = data_all[cols_numeric].loc[:, mask]
    data_all = pd.concat([data_all[["sig_id", "cp_type", "cp_time", "cp_dose","drug_id","fold_id"]], tmp], axis = 1)
    cols_numeric = [feat for feat in list(data_all.columns) if feat not in ["sig_id", "cp_type", "cp_time", "cp_dose","drug_id","fold_id"]]

    with open(inference_path + "cols_numeric.txt","rb") as f:
        cols_numeric = pickle.load(f) 

    data_all = pd.concat([data_all[["sig_id", "cp_type", "cp_time", "cp_dose","drug_id","fold_id"]], data_all[cols_numeric]], axis = 1)



    def scale_minmax(col):
        return (col - col.min()) / (col.max() - col.min())

    def scale_norm(col):
        return (col - col.mean()) / col.std()

    if scale == "boxcox":
        print(b_, "boxcox")
        data_all[cols_numeric] = data_all[cols_numeric].apply(scale_minmax, axis = 0)
        trans = []
        for feat in cols_numeric:
            trans_var, lambda_var = stats.boxcox(data_all[feat].dropna() + 1)
            trans.append(scale_minmax(trans_var))
        data_all[cols_numeric] = np.asarray(trans).T
        
    elif scale == "norm":
        print(b_, "norm")
        data_all[cols_numeric] = data_all[cols_numeric].apply(scale_norm, axis = 0)
        
    elif scale == "minmax":
        print(b_, "minmax")
        data_all[cols_numeric] = data_all[cols_numeric].apply(scale_minmax, axis = 0)
        
    elif scale == "rankgauss":
        ### Rank Gauss ###
        print(b_, "Rank Gauss")
        scaler = GaussRankScaler()
        data_all[cols_numeric] = scaler.fit_transform(data_all[cols_numeric])

        
    
        
        
    else:
        pass


    # PCA:

    if decompo == "PCA":
        print(b_, "PCA")
        GENES = [col for col in data_all.columns if col.startswith("g-")]
        CELLS = [col for col in data_all.columns if col.startswith("c-")]


        
        #pca_genes = PCA(n_components = ncompo_genes,random_state = seed).fit_transform(data_all[GENES])
        #pca_cells = PCA(n_components = ncompo_cells,random_state = seed).fit_transform(data_all[CELLS])
        with open(inference_path + "PCAG.pickle", "rb") as f:
            PCAG = pickle.load(f)
    
        with open(inference_path + "PCAC.pickle", "rb") as f:
            PCAC = pickle.load(f)
    
        pca_genes = PCAG.transform(data_all[GENES])
        pca_cells = PCAC.transform(data_all[CELLS])
        
        pca_genes = pd.DataFrame(pca_genes, columns = [f"pca_g-{i}" for i in range(ncompo_genes)])
        pca_cells = pd.DataFrame(pca_cells, columns = [f"pca_c-{i}" for i in range(ncompo_cells)])
        data_all = pd.concat([data_all, pca_genes, pca_cells], axis = 1)
    else:
        pass

    # One hot Encoding
    if encoding == "lb":
        print(b_, "Label Encoding")
        for feat in ["cp_time", "cp_dose"]:
            data_all[feat] = LabelEncoder().fit_transform(data_all[feat])
    elif encoding == "dummy":
        print(b_, "One-Hot")
        data_all = pd.get_dummies(data_all, columns = ["cp_time", "cp_dose"])

    GENES = [col for col in data_all.columns if col.startswith("g-")]
    CELLS = [col for col in data_all.columns if col.startswith("c-")]

    for stats in tqdm.tqdm(["sum", "mean", "std", "kurt", "skew"]):
        data_all["g_" + stats] = getattr(data_all[GENES], stats)(axis = 1)
        data_all["c_" + stats] = getattr(data_all[CELLS], stats)(axis = 1)    
        data_all["gc_" + stats] = getattr(data_all[GENES + CELLS], stats)(axis = 1)

    # save train data:

    

    train_df = data_all[:train.shape[0]]
    test_df = data_all[train.shape[0]:]


    features_to_drop = ["sig_id", "cp_type","drug_id"]
    try:
        data_all.drop(features_to_drop, axis = 1, inplace = True)
    except:
        pass
    try:
        targets.drop("sig_id", axis = 1, inplace = True)
    except:
        pass
    train_df = data_all[: train.shape[0]]
    train_df.reset_index(drop = True, inplace = True)

    test_df = data_all[train_df.shape[0]: ]
    test_df.reset_index(drop = True, inplace = True)

    try:
        test_df.drop(['fold_id'],axis = 1, inplace = True)
    except:
        pass
    X_test = test_df.values

    # Model Parameters:

    MAX_EPOCH = 200 
    optimizer_lr = 1e-2
    optimizer_weight_decay = 1e-5
    tabnet_params = dict(
        n_d = 32,
        n_a = 32,
        n_steps = 1,
        gamma = 1.3,
        lambda_sparse = 0,
        optimizer_fn = optim.Adam,
        optimizer_params = dict(lr = optimizer_lr, weight_decay = optimizer_weight_decay),
        mask_type = "entmax",
        scheduler_params = dict(
            mode = "min", patience = 5, min_lr = 2e-5, factor = 0.9),
        scheduler_fn = ReduceLROnPlateau,
        seed = seed,
        verbose = 10
        
    )

    # Custom Metric:
    """
    class LogitsLogLoss(Metric):


        def __init__(self):
            self._name = "logits_ll"
            self._maximize = False

        def __call__(self, y_true, y_pred):
            
            Compute LogLoss of predictions.

            Parameters
            ----------
            y_true: np.ndarray
                Target matrix or vector
            y_score: np.ndarray
                Score matrix or vector

            Returns
            -------
                float
                LogLoss of predictions vs targets.
            
            logits = 1 / (1 + np.exp(-y_pred))
            aux = (1 - y_true) * np.log(1 - logits + 1e-15) + y_true * np.log(logits + 1e-15)
            return np.mean(-aux)

    """
    """
    scores_auc_all = []
    test_cv_preds = []    
    NB_SPLITS = 10
    #mskf = MultilabelStratifiedKFold(n_splits = NB_SPLITS, random_state = 0, shuffle = True)

    
    oof_preds = []
    oof_targets = []
    scores = []
    scores_auc = []
    #for fold_nb, (train_idx, val_idx) in enumerate(mskf.split(train_df, targets)):
    for fold_nb in range(10):


        val_idx = train_df[train_df.fold_id == fold_nb].index
        train_idx = train_df.index.difference(val_idx)
        print(b_,"FOLDS: ", r_, fold_nb + 1)
        print(g_, '*' * 60, c_)

        train_df_dropfold = train_df.drop(["fold_id"],axis = 1, inplace = False)

        X_train, y_train = train_df_dropfold.values[train_idx, :], targets.values[train_idx, :]
        X_val, y_val = train_df_dropfold.values[val_idx, :], targets.values[val_idx, :]
        ### Model ###
        model = TabNetRegressor(**tabnet_params)

        ### Fit ###
        
        model.fit(
            X_train = X_train,
            y_train = y_train,
            eval_set = [(X_val, y_val)],
            eval_name = ["val"],
            eval_metric = ["logits_ll"],
            max_epochs = MAX_EPOCH,
            patience = 20,
            batch_size = 512, 
            virtual_batch_size = 64,
            num_workers = 1,
            drop_last = False,
            # To use binary cross entropy because this is not a regression problem
            loss_fn = F.binary_cross_entropy_with_logits
        )
        print(y_, '-' * 60)



        ### Predict on validation ###
        preds_val = model.predict(X_val)
        # Apply sigmoid to the predictions
        preds = 1 / (1 + np.exp(-preds_val))
        score = np.min(model.history["val_logits_ll"])

        ### Save OOF for CV ###
        oof_preds.append(preds_val)
        oof_targets.append(y_val)
        scores.append(score)

        ### Predict on test ###
        preds_test = model.predict(X_test)
        test_cv_preds.append(1 / (1 + np.exp(-preds_test)))

    oof_preds_all = np.concatenate(oof_preds)
    oof_targets_all = np.concatenate(oof_targets)
    test_preds_all = np.stack(test_cv_preds)

    
    aucs = []
    for task_id in range(oof_preds_all.shape[1]):
        aucs.append(roc_auc_score(y_true = oof_targets_all[:, task_id],
                                  y_score = oof_preds_all[:, task_id]
                                 ))
    print(f"{b_}Overall AUC: {r_}{np.mean(aucs)}")
    print(f"{b_}Average CV: {r_}{np.mean(scores)}")

    """

    # save model:
    #with open("model_with_drugid.pickle", "wb") as f:
    #    pickle.dump(model, f)

    with open(model_path + "model_with_drugid.pickle",  "rb") as f:
        model = pickle.load(f)

    preds_test = model.predict(X_test)
    test_preds_model = 1/ (1 + np.exp(-preds_test))

    # submission:
    all_feat = [col for col in submission.columns if col not in ["sig_id"]]
    # To obtain the same lenght of test_preds_all and submission
    test = pd.read_csv(data_path + "test_features.csv")
    sig_id = test[test["cp_type"] != "ctl_vehicle"].sig_id.reset_index(drop = True)
    #tmp = pd.DataFrame(test_preds_all.mean(axis = 0), columns = all_feat)
    tmp = pd.DataFrame(test_preds_model, columns = all_feat)
    tmp["sig_id"] = sig_id

    submission = pd.merge(test[["sig_id"]], tmp, on = "sig_id", how = "left")
    submission.fillna(0, inplace = True) 

    #submission[all_feat] = tmp.mean(axis = 0)
    submission_df = submission.copy()
    # Set control to 0
    #submission.loc[test["cp_type"] == 0, submission.columns[1:]] = 0
    #submission.to_csv("submission_drug.csv", index = None)

    return submission_df


