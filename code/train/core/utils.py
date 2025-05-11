import os 
import re
from pathlib import Path
import argparse
import torch
import torch.nn
import numpy as np
import pandas as pd
import mlflow as mf
from shutil import rmtree
from pyampd.ampd import find_peaks
from mlflow.tracking import MlflowClient
from mlflow.utils.autologging_utils.safety import try_mlflow_log
import scipy
    
def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def set_device(gpu_id):
    print(gpu_id)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_id))
        print("Using GPU: ", torch.cuda.current_device())
    else:
        n_threads = torch.get_num_threads()
        n_threads = min(n_threads, 8)
        torch.set_num_threads(n_threads)
        print("Using {} CPU Core".format(n_threads))

def get_nested_fold_idx(kfold):
    for fold_test_idx in range(kfold):
        fold_val_idx = (fold_test_idx+1)%kfold
        fold_train_idx = [fold for fold in range(kfold) if fold not in [fold_test_idx, fold_val_idx]]
        yield fold_train_idx, [fold_val_idx], [fold_test_idx]

def get_ckpt(r):
    ckpts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "restored_model_checkpoint")]
    return r.info.artifact_uri, ckpts

def mat2df(data):
    data.pop('__header__')
    data.pop('__version__')
    data.pop('__globals__')
    # convert to dataframe
    df = pd.DataFrame()
    for k, v in data.items():
        v = list(np.squeeze(v))
        # deal with trailing whitespace
        if isinstance(v[0], str):
            v = [re.sub(r"\s+$","",ele) for ele in v]
        # convert string nan to float64
        v = [np.nan if ele=='nan' else ele for ele in v]
        # df[k] = list(np.squeeze(v))
        df[k] = v
    return df
        
def norm_data(train_df, val_df, test_df, labels_feats=['patient','trial','SP', 'DP']):
    from sklearn.preprocessing import MinMaxScaler

    df_train = train_df.copy()
    df_val = val_df.copy()
    df_test = test_df.copy()

    df_train_norm=df_train[labels_feats].reset_index(drop=True)
    df_train_norm['SP'] = global_norm(df_train['SP'].values, 'SP')
    df_train_norm['DP'] = global_norm(df_train['DP'].values, 'DP')
    df_train_feats = df_train.drop(columns=labels_feats)
    feats = df_train_feats.columns

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_train_feats.values)
    df_train_norm = pd.concat([df_train_norm,pd.DataFrame(X_train,columns=feats)],axis=1)

    df_val_norm=df_val[labels_feats].reset_index(drop=True)
    df_val_norm['SP'] = global_norm(df_val['SP'].values, 'SP')
    df_val_norm['DP'] = global_norm(df_val['DP'].values, 'DP')
    df_val_feats = df_val.drop(columns=labels_feats)
    X_val = scaler.transform(df_val_feats.values)
    df_val_norm = pd.concat([df_val_norm,pd.DataFrame(X_val,columns=feats)],axis=1)

    df_test_norm=df_test[labels_feats].reset_index(drop=True)
    df_test_norm['SP'] = global_norm(df_test['SP'].values, 'SP')
    df_test_norm['DP'] = global_norm(df_test['DP'].values, 'DP')
    df_test_feats = df_test.drop(columns=labels_feats)
    X_test = scaler.transform(df_test_feats.values)
    df_test_norm = pd.concat([df_test_norm,pd.DataFrame(X_test,columns=feats)],axis=1)

    return df_train_norm, df_val_norm, df_test_norm
    
#%% Global Normalization
def global_norm(x, signal_type): 
    if signal_type == "SP": (x_min, x_max) = (80, 200)   # mmHg
    elif signal_type == "DP": (x_min, x_max) = (50, 110)   # mmHg
    elif signal_type == "ptt": (x_min, x_max) = (100, 900)  # 100ms - 900ms
    else: return None

    # Return values normalized between 0 and 1
    return (x - x_min) / (x_max - x_min)
    
def global_denorm(x, signal_type):
    if signal_type == "SP": (x_min, x_max) = (80, 200)   # mmHg
    elif signal_type == "DP": (x_min, x_max) = (50, 110)   # mmHg
    elif signal_type == "ptt": (x_min, x_max) = (100, 900)  # 100ms - 900ms
    else: return None

    # Return values normalized between 0 and 1
    return x * (x_max-x_min) + x_min

def freq_norm(x):
    x_min, _ = torch.min(x, dim=2, keepdim=True)
    x_max, _ = torch.max(x, dim=2, keepdim=True)
    
    if torch.any(x_max - x_min == 0):
        return None

    normalized = (x - x_min) / (x_max - x_min)
    return normalized, x_min, x_max

def freq_denorm(x, x_min, x_max):
    return x * (x_max - x_min) + x_min

def glob_demm(x, config, type='SP'): 
    # sensors global max, min
    if type=='SP':
        x_min, x_max = config.param_loader.SP_min, config.param_loader.SP_max
    elif type=='DP':
        x_min, x_max = config.param_loader.DP_min, config.param_loader.DP_max
    elif type=='ppg':
        x_min, x_max = config.param_loader.ppg_min, config.param_loader.ppg_max
    elif type=='abp':
        x_min, x_max = config.param_loader.abp_min, config.param_loader.abp_max
    return x * (x_max-x_min) + x_min

def glob_mm(x, config, type='SP'): 
    # sensors global max, min
    if type=='SP':
        x_min, x_max = config.param_loader.SP_min, config.param_loader.SP_max
    elif type=='DP':
        x_min, x_max = config.param_loader.DP_min, config.param_loader.DP_max
    elif type=='ppg':
        x_min, x_max = config.param_loader.ppg_min, config.param_loader.ppg_max
    elif type=='abp':
        x_min, x_max = config.param_loader.abp_min, config.param_loader.abp_max
    return (x - x_min) / (x_max - x_min)

def glob_dez(x, config, type='SP'): 
    # sensors global max, min
    if type=='SP':
        x_mean, x_std = config.param_loader.SP_mean, config.param_loader.SP_std
    elif type=='DP':
        x_mean, x_std = config.param_loader.DP_mean, config.param_loader.DP_std
    elif type=='ppg':
        x_mean, x_std = config.param_loader.ppg_mean, config.param_loader.ppg_std
    elif type=='abp':
        x_mean, x_std = config.param_loader.abp_mean, config.param_loader.abp_std
    return x * (x_std + 1e-6) + x_mean

def glob_z(x, config, type='sbp'): 
    # sensors global max, min
    if type=='SP':
        x_mean, x_std = config.param_loader.SP_mean, config.param_loader.SP_std
    elif type=='DP':
        x_mean, x_std = config.param_loader.DP_mean, config.param_loader.DP_std
    elif type=='ppg':
        x_mean, x_std = config.param_loader.ppg_mean, config.param_loader.ppg_std
    elif type=='abp':
        x_mean, x_std = config.param_loader.abp_mean, config.param_loader.abp_std
    return (x - x_mean)/(x_std + 1e-6)

#%% Local normalization
def loc_mm(x,config, type='SP'):
    return (x - x.min())/(x.max() - x.min() + 1e-6)

def loc_demm(x,config, type='SP'):
    return x * (x.max() - x.min() + 1e-6) + x.min()

def loc_z(x,config, type='SP'):
    return (x - x.mean())/(x.std() + 1e-6)

def loc_dez(x,config, type='SP'):
    return x * (x.std() + 1e-6) + x.mean()

#%% Compute bps
def compute_sp_dp(sig, fs=125, pk_th=0.6):
    sig = sig.astype(np.float64)
    try:
        peaks = find_peaks(sig,fs)
    except: # if prediction is too jitering.
        peaks = find_peaks(butter_lowpass_filter(sig, 8, fs, 5),fs)
        
    try:
        valleys = find_peaks(-sig,fs)
    except: # if prediction is too jitering.
        valleys = find_peaks(-butter_lowpass_filter(sig, 8, fs, 5),fs)
    
    sp, dp = -1 , -1
    flag1 = False
    flag2 = False
    
    ### Remove first or last if equal to 0 or len(sig)-1
    if peaks[0] == 0:
        peaks = peaks[1:]
    if valleys[0] == 0:
        valleys = valleys[1:]
    
    if peaks[-1] == len(sig)-1:
        peaks = peaks[:-1]
    if valleys[-1] == len(sig)-1:
        valleys = valleys[:-1]
    
    '''
    ### HERE WE SHOULD REMOVE THE FIRST AND LAST PEAK/VALLEY
    if peaks[0] < valleys[0]:
        peaks = peaks[1:]
    else:
        valleys = valleys[1:]
        
    if peaks[-1] > valleys[-1]:
        peaks = peaks[:-1]
    else:
        valleys = valleys[:-1]
    '''
    
    ### START AND END IN VALLEYS
    while len(peaks)!=0 and peaks[0] < valleys[0]:
        peaks = peaks[1:]
    
    while len(peaks)!=0 and peaks[-1] > valleys[-1]:
        peaks = peaks[:-1]
    
    ## Remove consecutive peaks with one considerably under the other
    new_peaks = []
    mean_vly_amp = np.mean(sig[valleys])
    if len(peaks)==1:
        new_peaks = peaks
    else:
        # define base case:

        for i in range(len(peaks)-1):
            if sig[peaks[i]]-mean_vly_amp > (sig[peaks[i+1]]-mean_vly_amp)*pk_th:
                new_peaks.append(peaks[i])
                break

        for j in range(i+1,len(peaks)):
            if sig[peaks[j]]-mean_vly_amp > (sig[new_peaks[-1]]-mean_vly_amp)*pk_th:
                new_peaks.append(peaks[j])
                
        if not np.array_equal(peaks,new_peaks):
            flag1 = True
            
        if len(valleys)-1 != len(new_peaks):
            flag2 = True
            
        if len(valleys)-1 == len(new_peaks):
            for i in range(len(valleys)-1):
                if not(valleys[i] < new_peaks[i] and new_peaks[i] < valleys[i+1]):
                    flag2 = True
        
        
    return np.median(sig[new_peaks]), np.median(sig[valleys]), flag1, flag2, new_peaks, valleys

def butter_lowpass_filter(data, lowcut, fs, order):
    """ Butterworth band-pass filter
    Parameters
    ----------
    data : array
        Signal to be filtered.
    lowcut : float
        Frequency lowcut for the filter. 
    highcut : float}
        Frequency highcut for the filter.
    fs : float
        Sampling rate.
    order: int
        Filter's order.

    Returns
    -------
    array
        Signal filtered with butterworth algorithm.
    """  
    nyq = fs * 0.5  # https://en.wikipedia.org/wiki/Nyquist_frequency
    lowcut = lowcut / nyq  # Normalize
    #highcut = highcut / nyq
    # Numerator (b) and denominator (a) polynomials of the IIR filter
    b, a = scipy.signal.butter(order, lowcut, btype='low', analog=False)
    return scipy.signal.filtfilt(b, a, data)
    
def get_bp_pk_vly_mask(data):
    try:
        _,_,_,_,pks, vlys = compute_sp_dp(data, 125, pk_th=0.6)

        pk_mask = np.zeros_like(data)
        vly_mask = np.zeros_like(data)
        pk_mask[pks] = 1
        vly_mask[vlys] = 1

    except:
        # print("!!! No peaks and vlys found for peak_vly_mask !!!")
        pk_mask = np.zeros_like(data)
        vly_mask = np.zeros_like(data)
    
    return np.array(pk_mask, dtype=bool), np.array(vly_mask, dtype=bool)

#%% Compute statistics for normalization
def cal_statistics(config, all_df):
    import pandas as pd
    from omegaconf import OmegaConf,open_dict
    all_df = pd.concat(all_df)
    OmegaConf.set_struct(config, True)

    with open_dict(config):
        for x in ['SP', 'DP']:
            config.param_loader[f'{x}_mean'] = float(all_df[x].mean())
            config.param_loader[f'{x}_std'] = float(all_df[x].std())
            config.param_loader[f'{x}_min'] = float(all_df[x].min())
            config.param_loader[f'{x}_max'] = float(all_df[x].max())
        
        # ppg
        if config.param_loader.ppg_norm.startswith('glob'):
            config.param_loader[f'ppg_mean'] = float(np.vstack(all_df['signal']).mean())
            config.param_loader[f'ppg_std'] = float(np.vstack(all_df['signal']).std())
            config.param_loader[f'ppg_min'] = float(np.vstack(all_df['signal']).min())
            config.param_loader[f'ppg_max'] = float(np.vstack(all_df['signal']).max())
            
        if 'abp_signal' in all_df.columns:
            config.param_loader[f'abp_mean'] = float(np.vstack(all_df['abp_signal']).mean())
            config.param_loader[f'abp_std'] = float(np.vstack(all_df['abp_signal']).std())
            config.param_loader[f'abp_min'] = float(np.vstack(all_df['abp_signal']).min())
            config.param_loader[f'abp_max'] = float(np.vstack(all_df['abp_signal']).max())
        else: #dummy stats
            config.param_loader[f'abp_mean'] = 0.0
            config.param_loader[f'abp_std'] = 1.0
            config.param_loader[f'abp_min'] = 0.0
            config.param_loader[f'abp_max'] = 1.0

    return config

#%% Compute metric
def cal_metric(err_dict, metric={}, mode='val'):
    for k, v in err_dict.items():
        metric[f'{k}_mae'] = np.mean(np.abs(v))
        metric[f'{k}_std'] = np.std(v)
        metric[f'{k}_me'] = np.mean(v)
    metric = {f'{mode}/{k}':round(v.item(),3) for k,v in metric.items()}
    return metric

#%% print/logging tools
def print_criterion(sbps, dbps):
    print("The percentage of SBP above 160: (0.10)", len(np.where(sbps>=160)[0])/len(sbps)) 
    print("The percentage of SBP above 140: (0.20)", len(np.where(sbps>=140)[0])/len(sbps)) 
    print("The percentage of SBP below 100: (0.10)", len(np.where(sbps<=100)[0])/len(sbps)) 
    print("The percentage of DBP above 100: (0.05)", len(np.where(dbps>=100)[0])/len(dbps)) 
    print("The percentage of DBP above 85: (0.20)", len(np.where(dbps>=85)[0])/len(dbps)) 
    print("The percentage of DBP below 70: (0.10)", len(np.where(dbps<=70)[0])/len(dbps)) 
    print("The percentage of DBP below 60: (0.05)", len(np.where(dbps<=60)[0])/len(dbps)) 

def get_cv_logits_metrics(fold_errors, loader, pred_sbp, pred_dbp, pred_abp, 
                            true_sbp, true_dbp, true_abp, 
                            sbp_naive, dbp_naive, mode="val"):

    fold_errors[f"{mode}_subject_id"].append(loader.dataset.subjects)
    fold_errors[f"{mode}_record_id"].append(loader.dataset.records)
    fold_errors[f"{mode}_sbp_naive"].append([sbp_naive])
    fold_errors[f"{mode}_sbp_pred"].append([pred_sbp])
    fold_errors[f"{mode}_sbp_label"].append([true_sbp])
    fold_errors[f"{mode}_dbp_naive"].append([dbp_naive])
    fold_errors[f"{mode}_dbp_pred"].append([pred_dbp])
    fold_errors[f"{mode}_dbp_label"].append([true_dbp])
    fold_errors[f"{mode}_abp_true"].append([true_abp])
    fold_errors[f"{mode}_abp_pred"].append([pred_abp])

#%% mlflow
def init_mlflow(config):
    mf.set_tracking_uri(str(Path(config.path.mlflow_dir).absolute()))  # set up connection
    mf.set_experiment(config.exp.exp_name)          # set the experiment

def log_params_mlflow(config):
    mf.log_params(config.get("exp"))
    # mf.log_params(config.get("param_feature"))
    try_mlflow_log(mf.log_params, config.get("param_preprocess"))
    try_mlflow_log(mf.log_params, config.get("param_trainer"))
    try_mlflow_log(mf.log_params, config.get("param_early_stop"))
    mf.log_params(config.get("param_loader"))
    # mf.log_params(config.get("param_trainer"))
    # mf.log_params(config.get("param_early_stop"))
    if config.get("param_aug"):
        if config.param_aug.get("filter"):
            for k,v in dict(config.param_aug.filter).items():
                mf.log_params({k:v})
    # mf.log_params(config.get("param_aug"))
    mf.log_params(config.get("param_model"))

def log_config(config_path):
    # mf.log_artifact(os.path.join(os.getcwd(), 'core/config/unet_sensors_5s.yaml'))
    # mf.log_dict(config, "config.yaml")
    mf.log_artifact(config_path)

def log_hydra_mlflow(name):
    mf.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))
    mf.log_artifact(os.path.join(os.getcwd(), '.hydra/hydra.yaml'))
    mf.log_artifact(os.path.join(os.getcwd(), '.hydra/overrides.yaml'))
    mf.log_artifact(os.path.join(os.getcwd(), f'{name}.log'))
    rmtree(os.path.join(os.getcwd()))

#####################################################
#####################################################
def to_group(pred, true, config, normalizer):
    """
    Assign group and divide
    """
    pred_group_bp = {}; true_group_bp = {}

    #upper_sp = normalizer(200, config, "SP"); upper_dp = normalizer(130, config, "DP")
    crisis_sp = normalizer(180, config, "SP"); crisis_dp = normalizer(120, config, "DP")
    hyper2_sp = normalizer(140, config, "SP"); hyper2_dp = normalizer(90, config, "DP")
    prehyper_sp = normalizer(120, config, "SP"); prehyper_dp = normalizer(80, config, "DP")
    normal_sp = normalizer(90, config, "SP"); normal_dp = normalizer(60, config, "DP")
    hypo_sp = normalizer(80, config, "SP"); hypo_dp = normalizer(40, config, "DP")

    pred_sp_crisis = []; pred_sp_hyper2 = []; pred_sp_prehyper = []; pred_sp_normal = []; pred_sp_hypo = [];
    pred_dp_crisis = []; pred_dp_hyper2 = []; pred_dp_prehyper = []; pred_dp_normal = []; pred_dp_hypo = [];
    true_sp_crisis = []; true_sp_hyper2 = []; true_sp_prehyper = []; true_sp_normal = []; true_sp_hypo = [];
    true_dp_crisis = []; true_dp_hyper2 = []; true_dp_prehyper = []; true_dp_normal = []; true_dp_hypo = [];

    for i, example in enumerate(true):
        if (example[0] <= hypo_sp) or (example[0] >= crisis_sp) or (example[1] >= crisis_dp) or (example[1] <= hypo_dp):  # Not considered
            continue
        # elif (crisis_sp <= example[0]) or (crisis_dp <= example[1]):
        #     pred_sp_crisis.append(pred[i][0])
        #     pred_dp_crisis.append(pred[i][1])
        #     true_sp_crisis.append(true[i][0])
        #     true_dp_crisis.append(true[i][1])
        elif (hyper2_sp <= example[0]) or (hyper2_dp <= example[1]):
            pred_sp_hyper2.append(pred[i][0])
            pred_dp_hyper2.append(pred[i][1])
            true_sp_hyper2.append(true[i][0])
            true_dp_hyper2.append(true[i][1])
        elif (prehyper_sp <= example[0]) or (prehyper_dp <= example[1]):
            pred_sp_prehyper.append(pred[i][0])
            pred_dp_prehyper.append(pred[i][1])
            true_sp_prehyper.append(true[i][0])
            true_dp_prehyper.append(true[i][1])
        elif (normal_sp <= example[0]) or (normal_dp <= example[1]):
            pred_sp_normal.append(pred[i][0])
            pred_dp_normal.append(pred[i][1])
            true_sp_normal.append(true[i][0])
            true_dp_normal.append(true[i][1])
        elif (hypo_sp < example[0]) or (hypo_dp < example[1]):
            pred_sp_hypo.append(pred[i][0])
            pred_dp_hypo.append(pred[i][1])
            true_sp_hypo.append(true[i][0])
            true_dp_hypo.append(true[i][1])
    pred_group_bp["SP"] = {#"crisis": np.array(pred_sp_crisis),
                           "hyper2": np.array(pred_sp_hyper2),
                           "prehyper": np.array(pred_sp_prehyper),
                           "normal": np.array(pred_sp_normal),
                           "hypo": np.array(pred_sp_hypo)}
    pred_group_bp["DP"] = {#"crisis": np.array(pred_dp_crisis),
                           "hyper2": np.array(pred_dp_hyper2),
                           "prehyper": np.array(pred_dp_prehyper),
                           "normal": np.array(pred_dp_normal),
                           "hypo": np.array(pred_dp_hypo)}
    true_group_bp["SP"] = {#"crisis": np.array(true_sp_crisis),
                           "hyper2": np.array(true_sp_hyper2),
                           "prehyper": np.array(true_sp_prehyper),
                           "normal": np.array(true_sp_normal),
                           "hypo": np.array(true_sp_hypo)}
    true_group_bp["DP"] = {#"crisis": np.array(true_dp_crisis),
                           "hyper2": np.array(true_dp_hyper2),
                           "prehyper": np.array(true_dp_prehyper),
                           "normal": np.array(true_dp_normal),
                           "hypo": np.array(true_dp_hypo)}
    return pred_group_bp, true_group_bp
    # Criterion Normalization
"""    upper_sp = normalizer(200, config, "SP"); upper_dp = normalizer(130, config, "DP")
    crisis_sp = normalizer(180, config, "SP"); crisis_dp = normalizer(120, config, "DP")
    hyper2_sp = normalizer(140, config, "SP"); hyper2_dp = normalizer(90, config, "DP")
    prehyper_sp = normalizer(120, config, "SP"); prehyper_dp = normalizer(80, config, "DP")
    normal_sp = normalizer(90, config, "SP"); normal_dp = normalizer(60, config, "DP")
    hypo_sp = normalizer(80, config, "SP"); hypo_dp = normalizer(40, config, "DP")"""


#####################################################
#####################################################

def save_result(metric, path):
    if os.path.exists(path):
        frame = pd.read_csv(path, index_col=0)
        frame =frame.append(metric, ignore_index=True)
        frame.to_csv(path)
    else:
        frame = pd.DataFrame.from_dict([metric])
        frame.to_csv(path)

def remove_outlier(list_of_df):
    output_list = []
    for df in list_of_df:
        df_ = df[(80<df.SP) | (df.SP<180) | (40<df.DP) | (df.DP<120)].reset_index()
        output_list.append(df_)
    return output_list

# def group_annot(list_of_df):
#     output_list = []
#     for df in list_of_df:
#         df['group'] = 100
#         df.loc[(140 <= df.SP) | (90 <= df.DP), 'group'] = 3  # Hyper2
#         df.loc[((120 <= df.SP) & (df.SP <= 140)) | ((80 <= df.DP) & (df.DP <= 90)), 'group'] = 2  # Prehyper
#         df.loc[((90 <= df.SP) & (df.SP <= 120)) | ((60 <= df.DP) & (df.DP <= 80)), 'group'] = 1  # Normal
#         df.loc[((80 <= df.SP) & (df.SP <= 90)) | ((40 <= df.DP) & (df.DP <= 60)), 'group'] = 0 # Hypo
#         value_counts = df['group'].value_counts()
#         remain = value_counts.get(100, 0)

#         if remain:
#             assert 1==2, "Annotating Group is Failed"
#         output_list.append(df)
#     return output_list

def group_annot(list_of_df):
    output_list = []
    for df in list_of_df:
        df['group'] = 100
        df.loc[((140 <= df.SP) | (90 <= df.DP)) , 'group'] = 3  # Hyper2
        df.loc[(((120 <= df.SP) & (df.SP <= 140)) | ((80 <= df.DP) & (df.DP <= 90))) & (df.group==100), 'group'] = 2  # Prehyper
        df.loc[(((90 <= df.SP) & (df.SP <= 120)) | ((60 <= df.DP) & (df.DP <= 80))) & (df.group==100), 'group'] = 1  # Normal
        df.loc[(((80 <= df.SP) & (df.SP <= 90)) | ((40 <= df.DP) & (df.DP <= 60))) & (df.group==100), 'group'] = 0 # Hypo
        df.loc[((df.SP < 80) & (df.DP < 40)) & (df.group==100), 'group'] = 0 # Hypo
        value_counts = df['group'].value_counts()
        remain = value_counts.get(100, 0)

        if remain:
            assert 1==2, "Annotating Group is Failed"
        output_list.append(df)
    return output_list

def group_shot(df, n=5): # n=5 --> Validation set shots 5
    list_of_shots = []
    for g in range(4): # (hypo, normal, pre_hyper, hyper2)
        #sampled = df[df["group"]==g].sample(n=n, random_state=0) # Fixing the sampled ppg signal
        group_df = df[df["group"]==g]
        sampled = group_df.sample(n=n if len(group_df) >= n else len(group_df), replace=False) 
        list_of_shots.append(sampled)
    shot_df = pd.concat(list_of_shots)
    shot_df = shot_df.sample(frac=1) # Shuffle the rows of merged data frame
    return shot_df

def group_count(list_of_df):
    xx = [[],[],[],[]]
    bp_group = ["Hypo", "Normal", "Prehyper", "Hyper2"]
    for df in list_of_df:
        for i in range(4):
            xx[i].append((df['group']==i).sum())
    print("##"*20)
    print("Instances per BP Group in Total Dataset")
    for i, bp in zip(range(len(xx)), bp_group):
        print(f"{bp}: {np.sum(xx[i])}")
    print("##"*20)
    print("Instances per BP Group in Each Fold")
    dic = dict()
    for ids,bp in enumerate(bp_group):
        dic[f"{bp}"] = xx[ids]
    df_ = pd.DataFrame(dic)
    df_.index.name = "Fold"
    print(df_)
    print("##"*20)

def transferring(config, transfer_config):
    transfer_config.param_model.wd = config.param_model.wd
    transfer_config.param_model.lr = config.param_model.lr
    transfer_config.param_model.batch_size = config.param_model.batch_size
    transfer_config.exp.random_state = config.exp.random_state
    return transfer_config


########### PCA

def perform_pca(X, n_components=20):
    X = X.squeeze()
    X_mean = torch.mean(X, dim=0)
    X_centered = X - X_mean
    
    covariance_matrix = torch.matmul(X_centered.T, X_centered) / (X_centered.shape[0] - 1)

    torch.backends.cuda.preferred_linalg_library("magma")
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix, UPLO='U')
    
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:, idx]
    
    principal_components = eigenvectors_sorted[:, :n_components]
    return principal_components, X_mean

def perform_pca_w_fft(X, n_components=20, trunc_leng=None):
    # Get FFT
    if trunc_leng is None:
        fft_num_components = (X.size(2) // 2) + 1 # FFT has symentic matrix
    else:
        fft_num_components = trunc_leng
    reshaped_data = X.view(X.size(0), -1).cpu().numpy()
    transformed_data = np.fft.fft(reshaped_data, axis=1)
    fft_emb = torch.tensor(np.real(transformed_data[:, :fft_num_components]), dtype=torch.float32, device=X.device)

    X = fft_emb.squeeze()

    X_mean = torch.mean(X, dim=0)
    X_centered = X - X_mean
    
    covariance_matrix = torch.matmul(X_centered.T, X_centered) / (X_centered.shape[0] - 1)

    torch.backends.cuda.preferred_linalg_library("magma")
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix, UPLO='U')
    
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:, idx]
    
    principal_components = eigenvectors_sorted[:, :n_components]
    return principal_components, X_mean

def project_to_pca_plane(new_data, principal_components, mean):
    new_data = new_data.squeeze()
    new_data_centered = new_data - mean
    projected_data = torch.matmul(new_data_centered, principal_components)
    return projected_data


def extract_pca_statistics(data, labels, groups, pca, boundary=2, group_labels=[0, 1, 2, 3], device='cuda'):
    """
    주어진 데이터셋의 PCA 변환 버전에서 그룹별 평균과 분산을 추출합니다.
    """
    all_data = data
    pca.fit(all_data.cpu().numpy())
    all_data = pca.transform(all_data.cpu().numpy())
    all_data = torch.tensor(all_data).to(device)
    
    all_groups = groups
    grouped_data = get_grouped_data(all_data, all_groups, group_labels)

    statistics = {}
    for group_label in group_labels:
        statistics[group_label] = get_statistics(grouped_data[group_label], boundary)

    return statistics

def filter_dataset_based_on_statistics(data, labels, groups, statistics, pca=None, group_labels=[0, 1, 2, 3], is_pca=False, device='cuda'):
    """
    주어진 데이터셋을 주어진 통계치를 기준으로 필터링합니다.
    """
    all_data = data.unsqueeze(1).to(device)
    real_data = all_data
    all_groups = groups.to(device)

    if is_pca:
        print('pca transformed')
        
        all_data = pca.fit_transform(all_data.squeeze().cpu().numpy())
        all_data = torch.tensor(all_data).to(device)
    else:
        print('raw data')
    
    grouped_data = get_grouped_data(all_data, all_groups, group_labels)

    filtered_data = {}
    for group_label in group_labels:
        mean, var, lower_bound, upper_bound = statistics[group_label]
        current_data = grouped_data[group_label]
        filtered_data[group_label], mask = filter_data(current_data, lower_bound, upper_bound, real_data=real_data)
        print(f"Group {group_label}: {current_data.size(0)} to {filtered_data[group_label].size(0)}")
        if not mask.shape[0] == 0:
            labels = labels[mask]
    return filtered_data, labels
