import os
from shutil import rmtree
from functools import reduce
import operator
from pathlib import Path
import warnings

import pandas as pd
import numpy as np 
from scipy.io import loadmat
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import wandb
import joblib
import mlflow as mf
import coloredlogs, logging

# 파일 및 경로 처리 라이브러리
from glob import glob

from omegaconf import OmegaConf

from core.loaders import *
from core.solver_s2s import Solver
from core.utils import (
    get_nested_fold_idx, get_ckpt, cal_metric, cal_statistics, mat2df, to_group,
    remove_outlier, group_annot, group_count, group_shot, transferring, dbg
)
from core.load_model import model_fold
from core.model_config import model_configuration
from core.models import *
from core.prompt_tuning import MQFP_wrapper

import pdb

warnings.filterwarnings("ignore")

coloredlogs.install()
logger = logging.getLogger(__name__)

#%%
class SolverS2l(Solver):
    def __init__(self, config):
        super(SolverS2l, self).__init__(config)
        self.config = config
        self.config.exp.data_name = self.config.target
        
    def _get_model(self, ckpt_path_abs=None, fold=None):
        model = None
        if self.config.transfer:
            backbone_name = f"{self.config.transfer}-{self.config.exp.model_type}"
            if self.config.method == "original": # TODO
                if self.config.baseline == "scratch":
                    if self.config.exp.model_type == "resnet1d":
                        self.transfer_config_path = f"./core/config/dl/resnet/resnet_{self.config.transfer}.yaml"
                    elif self.config.exp.model_type == "spectroresnet":
                        self.transfer_config_path = f"./core/config/dl/spectroresnet/spectroresnet_{self.config.transfer}.yaml"
                    elif self.config.exp.model_type == "mlpbp":
                        self.transfer_config_path = f"./core/config/dl/mlpbp/mlpbp_{self.config.transfer}.yaml"
                    elif self.config.exp.model_type == "bptransformer":
                        self.transfer_config_path = f"./core/config/dl/bptransformer/bptransformer_{self.config.transfer}.yaml"
                    self.transfer_config = OmegaConf.load(self.transfer_config_path)
                    self.transfer_config = transferring(self.config, self.transfer_config)
                    if self.config.exp.model_type == "resnet1d":
                        model = Resnet1dRegressor_original(self.transfer_config.param_model, random_state=self.transfer_config.exp.random_state)
                    elif self.config.exp.model_type == "spectroresnet":
                        model = SpectroResnet(self.transfer_config.param_model, random_state=self.transfer_config.exp.random_state)
                    elif self.config.exp.model_type == "mlpbp":
                        model = MLPBP(self.transfer_config.param_model, random_state=self.transfer_config.exp.random_state)
                    elif self.config.exp.model_type == "bptransformer":
                        model = BPTransformerRegressor(self.transfer_config.param_model, random_state=self.transfer_config.exp.random_state)
                else:
                    print(f"pretrained_models/{backbone_name}/fold{fold}.ckpt")
                    if self.config.exp.model_type == "resnet1d":
                        model = Resnet1dRegressor_original.load_from_checkpoint(f"pretrained_models/{backbone_name}/fold{fold}.ckpt")
                    elif self.config.exp.model_type == "spectroresnet":
                        if self.config.target.endswith("ecg"):
                            model = SpectroResnet.load_from_checkpoint(f"pretrained_models/{backbone_name}/fold{fold}.ckpt")
                        else:
                            checkpoint = torch.load(f"pretrained_models/{backbone_name}/fold{fold}.ckpt", map_location="cpu")
                            keys_to_remove = [k for k in checkpoint['state_dict'] if 'mid_spec_layers.0.mlp' in k]
                            for k in keys_to_remove:
                                del checkpoint['state_dict'][k]
                            self.transfer_config = OmegaConf.load(f"./core/config/dl/spectroresnet/spectroresnet_{self.config.transfer}.yaml")
                            self.transfer_config = transferring(self.config, self.transfer_config)
                            model = SpectroResnet(self.transfer_config.param_model, random_state=self.transfer_config.exp.random_state)
                            model.load_state_dict(checkpoint['state_dict'], strict=False)
                    elif self.config.exp.model_type == "mlpbp":
                        if self.config.target.endswith("ecg"):
                            model = MLPBP.load_from_checkpoint(f"pretrained_models/{backbone_name}/fold{fold}.ckpt")
                        else:
                            checkpoint = torch.load(f"pretrained_models/{backbone_name}/fold{fold}.ckpt", map_location="cpu")
                            self.transfer_config = OmegaConf.load(f"./core/config/dl/mlpbp/mlpbp_{self.config.transfer}.yaml")
                            self.transfer_config = transferring(self.config, self.transfer_config)
                            model = MLPBP(self.transfer_config.param_model, random_state=self.transfer_config.exp.random_state)
                            model_state_dict = model.state_dict()
                            keys_to_remove = []
                            for k in checkpoint['state_dict']:
                                if k in model_state_dict and checkpoint['state_dict'][k].shape != model_state_dict[k].shape:
                                    keys_to_remove.append(k)

                            for k in keys_to_remove:
                                print(f"[!] Removing mismatched key: {k}, shape ckpt={checkpoint['state_dict'][k].shape}, model={model_state_dict[k].shape}")
                                del checkpoint['state_dict'][k]
                            model.load_state_dict(checkpoint['state_dict'], strict=False)
                    elif self.config.exp.model_type == "bptransformer":
                        model = BPTransformerRegressor.load_from_checkpoint(f"pretrained_models/{backbone_name}/fold{fold}.ckpt")
                    model.param_model.lr = self.config.param_model.lr
                    model.param_model.wd = self.config.param_model.wd
                    model.param_model.batch_size = self.config.param_model.batch_size
            else:
                if self.config.exp.model_type == "resnet1d":
                    model = Resnet1dRegressor.load_from_checkpoint(f"pretrained_models/{backbone_name}/fold{fold}.ckpt", strict=False)
                elif self.config.exp.model_type == "spectroresnet":
                    model = SpectroResnet.load_from_checkpoint(f"pretrained_models/{backbone_name}/fold{fold}.ckpt", strict=False)
                elif self.config.exp.model_type == "mlpbp":
                    model = MLPBP.load_from_checkpoint(f"pretrained_models/{backbone_name}/fold{fold}.ckpt", strict=False)
                elif self.config.exp.model_type == "bptransformer":
                    model = BPTransformerRegressor.load_from_checkpoint(f"pretrained_models/{backbone_name}/fold{fold}.ckpt", strict=False)
            # Initialize Classifier
            if self.config.baseline == "lp" or self.config.reset_head:
                model.model.main_clf = nn.Linear(in_features=model.model.main_clf.in_features,
                                                out_features=model.model.main_clf.out_features,
                                                bias=model.model.main_clf.bias is not None)
            print(f"####### Load {self.config.exp.model_type} backbone model pre-trained by {self.config.transfer} #######")

            return model
        if not ckpt_path_abs:
            if self.config.exp.model_type == "resnet1d":
                if self.config.method == "original":
                    model = Resnet1dRegressor_original(self.config.param_model, random_state=self.config.exp.random_state)
                else:
                    model = Resnet1dRegressor(self.config.param_model, random_state=self.config.exp.random_state)
            elif self.config.exp.model_type == "spectroresnet":
                model = SpectroResnet(self.config.param_model, random_state=self.config.exp.random_state)
            elif self.config.exp.model_type == "mlpbp":
                model = MLPBP(self.config.param_model, random_state=self.config.exp.random_state)
            elif self.config.exp.model_type == "bptransformer":
                model = BPTransformerRegressor(self.config.param_model, random_state=self.config.exp.random_state)
            else:
                model = eval(self.config.exp.model_type)(self.config.param_model, random_state=self.config.exp.random_state)
            return model
        else:
            if self.config.exp.model_type == "resnet1d":
                model = Resnet1dRegressor.load_from_checkpoint(ckpt_path_abs)
                if self.config.method == "original":
                    model = Resnet1dRegressor_original.load_from_checkpoint(ckpt_path_abs)
            elif self.config.exp.model_type == "spectroresnet":
                model = SpectroResnet.load_from_checkpoint(ckpt_path_abs)
            elif self.config.exp.model_type == "mlpbp":
                model = MLPBP.load_from_checkpoint(ckpt_path_abs)
            else:
                model = eval(self.config.exp.model_type).load_from_checkpoint(ckpt_path_abs)
            return model
    
    def get_cv_metrics(self, fold_errors, dm, model, outputs, mode="val", fold=None):
        if mode=='val':
            loader = dm.val_dataloader()
        elif mode=='test':
            loader = dm.test_dataloader()

        bp_denorm = loader.dataset.bp_denorm

        #--- Predict
        pred = outputs["pred_bp"].numpy()
        true = outputs["true_bp"].numpy()
        naive =  np.mean(dm.train_dataloader(is_print=False).dataset._target_data, axis=0)
        #####################################################
        #####################################################
        pred_group_bp, true_group_bp = to_group(pred, true, self.config, loader.dataset.bp_norm)  # TODO
        #####################################################
        #####################################################
        #--- Evaluate
        err_dict = {}
        for i, tar in enumerate(['SP', 'DP']):
            tar_acrny = 'sbp' if tar=='SP' else 'dbp'
            pred_bp = bp_denorm(pred[:,i], self.config, tar)
            true_bp = bp_denorm(true[:,i], self.config, tar)
            naive_bp = bp_denorm(naive[i], self.config, tar)

            # error
            err_dict[tar_acrny] = pred_bp - true_bp
            fold_errors[f"{mode}_{tar_acrny}_pred"].append(pred_bp)
            fold_errors[f"{mode}_{tar_acrny}_label"].append(true_bp)
            #####################################################
            #####################################################
            for group in ["hypo", "normal", "prehyper", "hyper2",]: #"crisis"]:
                if len(pred_group_bp[tar][group]) != 0:
                    pr_group_bp = bp_denorm(pred_group_bp[tar][group], self.config, tar)
                    tr_group_bp = bp_denorm(true_group_bp[tar][group], self.config, tar)
                    fold_errors[f"{mode}_{tar_acrny}_{group}_pred"].append(pr_group_bp)
                    fold_errors[f"{mode}_{tar_acrny}_{group}_label"].append(tr_group_bp)
            #####################################################
            #####################################################
            fold_errors[f"{mode}_{tar_acrny}_naive"].append([naive_bp]*len(pred_bp))
        fold_errors[f"{mode}_subject_id"].append(loader.dataset.subjects)
        fold_errors[f"{mode}_record_id"].append(loader.dataset.records)
        
        results = []
        if fold is not None:
            for i, (s, d) in enumerate(zip(err_dict['sbp'], err_dict['dbp'])):
                results.append(
                    {
                        'fold':fold,
                        'group':loader.dataset.all_group[i],
                        'sbp_error': s,
                        'dpb_error': d
                    }
                )
            
        metrics = cal_metric(err_dict, mode=mode) 
        return metrics, results
            
#%%
    def evaluate(self):
        #####################################################
        #####################################################
        fold_errors_template = {"subject_id":[], "record_id": [],
                                "sbp_naive":[],  "sbp_pred":[], "sbp_label":[],
                                "dbp_naive":[],  "dbp_pred":[],   "dbp_label":[],
                                "sbp_hypo_pred":[], "dbp_hypo_pred":[],
                                "sbp_normal_pred": [], "dbp_normal_pred": [],
                                "sbp_prehyper_pred": [], "dbp_prehyper_pred": [],
                                "sbp_hyper2_pred": [], "dbp_hyper2_pred": [],
                                #"sbp_crisis_pred": [], "dbp_crisis_pred": [],
                                "sbp_hypo_label": [], "dbp_hypo_label": [],
                                "sbp_normal_label": [], "dbp_normal_label": [],
                                "sbp_prehyper_label": [], "dbp_prehyper_label": [],
                                "sbp_hyper2_label": [], "dbp_hyper2_label": [],
                                #"sbp_crisis_label": [], "dbp_crisis_label": [],
                                }
        #####################################################
        #####################################################
        fold_errors = {f"{mode}_{k}":[] for k,v in fold_errors_template.items() for mode in ["val","test"]}
        
        #--- Data module
        dm = self._get_loader()
        #--- Load data
        if self.config.exp.subject_dict.endswith('.pkl'):
            all_split_df = joblib.load(self.config.exp.subject_dict)
        elif self.config.exp.subject_dict.endswith('fold'):
            all_split_df = [mat2df(loadmat(f"{self.config.exp.subject_dict}_{i}.mat")) for i in range(self.config.exp.N_fold)]
        
        #####################################################
        #####################################################
        all_split_df = remove_outlier(all_split_df)
        # if self.config.group_avg:
        all_split_df = group_annot(all_split_df)  # TODO: Make Group Annotation in DataFrame
        #####################################################
        #####################################################
        #--- Nested cv
        self.config = cal_statistics(self.config, all_split_df) # TODO: Write Statistics of Data in config
        print(self.config)

        folds_results = []
        for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(self.config.exp.N_fold)):
            # get_nested_fold_idx : [[0,1,2],[3],[4]] ## Generator
            if (self.config.exp.cv=='HOO') and (foldIdx==1):  break
            #if foldIdx==1: break

            # train_df = pd.concat((np.array(all_split_df)[folds_train]))
            # Concatenate training dataframes using for loop
            train_df = pd.concat(np.array(all_split_df)[folds_train])
            val_df = pd.concat(np.array(all_split_df)[folds_val])
            if self.config.shots: # train and validate with few-shot
                train_df = group_shot(train_df, n=self.config.shots)
                val_df = group_shot(val_df, n=5)
            test_df = pd.concat(np.array(all_split_df)[folds_test])
            dm.setup_kfold(train_df, val_df, test_df)

            # Find scaled ppg_max, ppg_min
            ppg_min = np.min(dm.train_dataloader().dataset.all_ppg)
            ppg_max = np.max(dm.train_dataloader().dataset.all_ppg)
            stats = [ppg_min, ppg_max]
            
            #####################################################
            #####################################################
            if self.config.method.startswith("prompt"):
                #--- Init model\ ##
                data_name = self.config.exp["data_name"]
                ck_path = os.path.join(self.config.root_dir, "models", model_fold[self.config.backbone][data_name][self.config.seed][foldIdx])  # yt
                # ck_path = os.path.join(self.config.root_dir, "models", model_fold[data_name][self.config.seed][foldIdx])                      # yewon
                if self.config.transfer:
                    base_model = self._get_model(fold=foldIdx)
                else:
                    base_model = self._get_model(ck_path)
                model_config = model_configuration[data_name]
                data_shape = model_config["data_dim"]
                model = MQFP_wrapper(base_model, data_shape, model_config, self.config, stats, foldIdx)

                for name, param in model.named_parameters():
                    train_list = ['prompt_learner', 'main_clf', 'penultimate_layer_prompt'] if self.config.train_head else ['prompt_learner', 'layer_wise_prompt']
                    if any(train_param in name for train_param in train_list):
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
            
            if self.config.method == "original":       
                if self.config.transfer:
                    model = self._get_model(fold=foldIdx)
                else:         
                    model = self._get_model()
                    
                for name, param in model.named_parameters():
                    if self.config.baseline == "lp":
                        # train only main_clf
                        if 'main_clf' in name:
                            param.requires_grad_(True)
                        else:
                            param.requires_grad_(False)
                    elif self.config.baseline == "ft":
                        # train all parameters
                        param.requires_grad_(True)

            enabled = set()
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    enabled.add(name)
            print(f"Parameters to be updated: {enabled}")
            #####################################################
            #####################################################
            # Define optimizer with different learning rates for prompt_learner and other parameters

            early_stop_callback = EarlyStopping(**dict(self.config.param_early_stop))
            # checkpoint_callback = ModelCheckpoint(**dict(self.config.logger.param_ckpt))
            lr_logger = LearningRateMonitor()
            # trainer = MyTrainer(**dict(self.config.param_trainer), callbacks=[early_stop_callback, checkpoint_callback, lr_logger ])
            #--- trainer main loop

            mf.pytorch.autolog()
            with mf.start_run(run_name=f"cv{foldIdx}", nested=True) as run:
                artifact_uri = mf.get_artifact_uri()
                checkpoint_callback = ModelCheckpoint(
                    **dict(self.config.logger.param_ckpt),
                    dirpath=f"{artifact_uri}/restored_model_checkpoint"
                )
                trainer = MyTrainer(**dict(self.config.param_trainer), callbacks=[early_stop_callback, checkpoint_callback, lr_logger])
                # train
                trainer.fit(model, dm)
                
                print("run_id", run.info.run_id)
                artifact_uri, ckpt_path = get_ckpt(mf.get_run(run_id=run.info.run_id))

                # load best ckpt
                print('load best ckpt')
                ckpt_path_abs = str(Path(artifact_uri)/ckpt_path[0])
                model.load_state_dict(torch.load(ckpt_path_abs)["state_dict"])
                # model = self._get_model(ckpt_path_abs=ckpt_path_abs)

                # evaluate
                val_outputs = trainer.validate(model=model, val_dataloaders=dm.val_dataloader(), verbose=False)
                test_outputs = trainer.test(model=model, test_dataloaders=dm.test_dataloader(), verbose=True)
                if self.config.get_emb:
                    raw_inputs = torch.cat(model.raw_input, dim=0)
                    prompted_input = torch.cat(model.prompted_input, dim=0)
                    hidden_embs = torch.cat(model.hidden_embs, dim=0)
                    prompt_hist = torch.cat(model.prompt_hist, dim=0)

                    if self.config.sym_prompt:
                        file_name = f'{wandb.run.group}_fold{foldIdx}_shot{self.config.shots}_sym_prompt_{self.config.sym_prompt}'
                    else:
                        file_name = f'{wandb.run.group}_fold{foldIdx}_shot{self.config.shots}_sym_prompt_{self.config.sym_prompt}'

                    torch.save(raw_inputs, f'./results/embeddings/{file_name}_train_head_{self.config.train_head}_raw_inputs.pt')
                    torch.save(prompted_input, f'./results/embeddings/{file_name}_train_head_{self.config.train_head}_prompted_inputs.pt')
                    torch.save(hidden_embs, f'./results/embeddings/{file_name}_train_head_{self.config.train_head}_hidden_embs.pt')
                    torch.save(prompt_hist, f'./results/embeddings/{file_name}_train_head_{self.config.train_head}_prompt_hist.pt')
                    
                # save updated model
                trainer.model = model
                trainer.save_checkpoint(ckpt_path_abs)
                if self.config.save_for_pretraining:
                    trainer.save_checkpoint(f'/data1/bubble3jh/bp_L2P/code/train/pretrained_models/{self.config.target}-{self.config.backbone}/fold{foldIdx}.ckpt')

                # clear redundant mlflow models (save disk space)
                redundant_model_path = Path(artifact_uri)/'restored_model_checkpoint'
                if redundant_model_path.exists(): rmtree(redundant_model_path)

                metrics, _ = self.get_cv_metrics(fold_errors, dm, model, val_outputs, mode="val")
                metrics, results = self.get_cv_metrics(fold_errors, dm, model, test_outputs, mode="test", fold=foldIdx)
                if len(results) != 0:
                    folds_results.append(results)
                    
                    # --- 여기서 fold-wise SPDP + GAL 계산 및 wandb log ---
                    df = pd.DataFrame(results)
                    sbp_mae = np.mean(np.abs(df["sbp_error"]))
                    dbp_mae = np.mean(np.abs(df["dpb_error"]))
                    spdp = sbp_mae + dbp_mae

                    gal_sbp = df.groupby("group")["sbp_error"].apply(lambda x: np.mean(np.abs(x))).mean()
                    gal_dbp = df.groupby("group")["dpb_error"].apply(lambda x: np.mean(np.abs(x))).mean()
                    gal = gal_sbp + gal_dbp

                    logger.info(f"[Fold {foldIdx}] SPDP = {spdp:.2f}, GAL = {gal:.2f}")

                    if not self.config.ignore_wandb:
                        wandb.log({
                            f"Fold{foldIdx}/SPDP": spdp,
                            f"Fold{foldIdx}/GAL": gal,
                            f"Fold{foldIdx}/SBP_MAE": sbp_mae,
                            f"Fold{foldIdx}/DBP_MAE": dbp_mae,
                            f"Fold{foldIdx}/GAL_SBP": gal_sbp,
                            f"Fold{foldIdx}/GAL_DBP": gal_dbp
                        })    
                logger.info(f"\t {metrics}")
                mf.log_metrics(metrics)

        #--- compute final metric
        out_metric = {}
        fold_errors = {k:np.concatenate(v, axis=0) for k,v in fold_errors.items()}
        err_dict = {tar: fold_errors[f"test_{tar}_naive"] - fold_errors[f"test_{tar}_label"] \
                    for tar in ['sbp', 'dbp']}
        naive_metric = cal_metric(err_dict, mode='nv')
        out_metric.update(naive_metric)

        #####################################################
        #####################################################
        for mode in ['val', 'test']:
            err_dict = {tar: fold_errors[f"{mode}_{tar}_pred"] - fold_errors[f"{mode}_{tar}_label"] \
                        for tar in ['sbp', 'dbp',  'sbp_hypo', 'dbp_hypo', 'sbp_normal', 'dbp_normal',
                                    'sbp_prehyper', 'dbp_prehyper', 'sbp_hyper2', 'dbp_hyper2',
                                    ]}
            #####################################################
            #####################################################
            tmp_metric = cal_metric(err_dict, mode=mode)
            if mode == 'test':
                sbp = tmp_metric['test/sbp_mae']
                dbp = tmp_metric['test/dbp_mae']
                groups = ['hypo', 'normal', 'prehyper', 'hyper2']
                sbps = [tmp_metric[f'test/sbp_{group}_mae'] for group in groups]
                dbps = [tmp_metric[f'test/dbp_{group}_mae'] for group in groups]
                sbp_gal = sum(sbps) / 4
                dbp_gal = sum(dbps) / 4
                gal = sbp_gal + dbp_gal
                tmp_metric['sbp_gal'] = sbp_gal
                tmp_metric['dbp_gal'] = dbp_gal
                tmp_metric['gal'] = gal

                if not self.config.ignore_wandb:
                    wandb.log(tmp_metric)
                    wandb.run.summary[f'spdp'] = sbp + dbp
                    wandb.run.summary[f'gal'] = gal
                    wandb.run.summary[f'sbp_gal'] = sbp_gal
                    wandb.run.summary[f'dbp_gal'] = dbp_gal
                    wandb.run.summary[f'sbp'] = sbp
                    wandb.run.summary[f'dbp'] = dbp
                    wandb.run.summary[f'spdp_hypo'] = tmp_metric[f'test/sbp_hypo_mae'] + tmp_metric[f'test/dbp_hypo_mae']
                    wandb.run.summary[f'spdp_normal'] = tmp_metric[f'test/sbp_normal_mae'] + tmp_metric[f'test/dbp_normal_mae']
                    wandb.run.summary[f'spdp_prehyper'] = tmp_metric[f'test/sbp_prehyper_mae'] + tmp_metric[f'test/dbp_prehyper_mae']
                    wandb.run.summary[f'spdp_hyper2'] = tmp_metric[f'test/sbp_hyper2_mae'] + tmp_metric[f'test/dbp_hyper2_mae']
                    
            out_metric.update(tmp_metric)
        
        if self.config.save_result:
            flat_list = reduce(operator.concat, folds_results)
            df_output = pd.DataFrame(flat_list)
            df_output.to_csv(f"./results/errors_{self.config.transfer}_{self.config.target}_shot{self.config.shots}_train_head_{self.config.train_head}_sym_{self.config.sym_prompt}_ours.csv", index=False)

        return out_metric

    def test(self):
        results = {}
        #####################################################
        #####################################################

        fold_errors_template = {"subject_id": [], "record_id": [],
                                "sbp_naive": [], "sbp_pred": [], "sbp_label": [],
                                "dbp_naive": [], "dbp_pred": [], "dbp_label": [],
                                "sbp_hypo_pred": [], "dbp_hypo_pred": [],
                                "sbp_normal_pred": [], "dbp_normal_pred": [],
                                "sbp_prehyper_pred": [], "dbp_prehyper_pred": [],
                                "sbp_hyper2_pred": [], "dbp_hyper2_pred": [],
                                #"sbp_crisis_pred": [], "dbp_crisis_pred": [],
                                "sbp_hypo_label": [], "dbp_hypo_label": [],
                                "sbp_normal_label": [], "dbp_normal_label": [],
                                "sbp_prehyper_label": [], "dbp_prehyper_label": [],
                                "sbp_hyper2_label": [], "dbp_hyper2_label": [],
                                #"sbp_crisis_label": [], "dbp_crisis_label": [],
                                }
        #####################################################
        #####################################################
        fold_errors = {f"{mode}_{k}":[] for k,v in fold_errors_template.items() for mode in ["test"]}
        #wandb.log({f"{mode}_{k}":[] for k,v in fold_errors_template.items() for mode in ["test"]})
        #--- Data module
        dm = self._get_loader()

        #--- Load data
        if self.config.exp.subject_dict.endswith('.pkl'):
            all_split_df = joblib.load(self.config.exp.subject_dict)
        elif self.config.exp.subject_dict.endswith('fold'):
            all_split_df = [mat2df(loadmat(f"{self.config.exp.subject_dict}_{i}.mat")) for i in range(self.config.exp.N_fold)]

        #####################################################
        #####################################################
        all_split_df = remove_outlier(all_split_df)
        # if self.config.group_avg:
        all_split_df = group_annot(all_split_df)
        #####################################################
        #####################################################
        #--- Nested cv
        self.config = cal_statistics(self.config, all_split_df)

        folds_results = []
        for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(self.config.exp.N_fold)):
            #if foldIdx==1: break #################################### TODO:
            if (self.config.exp.cv=='HOO') and (foldIdx==1):  break
            train_df = pd.concat(np.array(all_split_df)[folds_train])
            val_df = pd.concat(np.array(all_split_df)[folds_val])
            test_df = pd.concat(np.array(all_split_df)[folds_test])

            dm.setup_kfold(train_df, val_df, test_df)

            # Find scaled ppg_max, ppg_min
            ppg_min = np.min(dm.train_dataloader().dataset.all_ppg)
            ppg_max = np.max(dm.train_dataloader().dataset.all_ppg)
            stats = [ppg_min, ppg_max] 
            
            #--- load trained model
            if 'param_trainer' in self.config.keys():
                trainer = MyTrainer(**dict(self.config.param_trainer))
            else:
                trainer = MyTrainer()
            ckpt_path_abs = glob(f'{self.config.param_test.model_path}{foldIdx}' + '*.ckpt')[0]

            if self.config.method.startswith("prompt"):
                #--- Init model\ ##
                data_name = self.config.exp["data_name"]
                if self.config.transfer:
                    regressor = self._get_model(fold=foldIdx)
                else:
                    regressor = self._get_model(ckpt_path_abs) # Load Model # TODO
                model_config = model_configuration[self.config.backbone][data_name] # Load pre-trained model config
                data_shape = model_config["data_dim"] 
                model = MQFP_wrapper(regressor, data_shape, model_config, self.config, stats, foldIdx)
                #model = MQFP_wrapper.load_from_checkpoint(ckpt_path_abs) # Call Prompt Model 
                
                model.load_state_dict(torch.load(ckpt_path_abs)["state_dict"])

                #model = model.load_from_checkpoint(ckpt_path_abs)

            if self.config.method == "original":
                if self.config.transfer:
                    model = self._get_model(fold=foldIdx)
                else:
                    model = self._get_model()

            model.eval()
            trainer.model = model

            #--- get test output
            val_outputs = trainer.validate(model=model, val_dataloaders=dm.val_dataloader(), verbose=False)
            test_outputs = trainer.test(model=model, test_dataloaders=dm.test_dataloader(), verbose=True)
            metrics, results = self.get_cv_metrics(fold_errors, dm, model, test_outputs, mode="test", fold=foldIdx)
            if len(results) != 0:
                folds_results.append(results)
            logger.info(f"\t {metrics}")

        #--- compute final metric
        results['fold_errors'] = fold_errors
        out_metric = {}
        fold_errors = {k:np.concatenate(v, axis=0) for k,v in fold_errors.items()}
        err_dict = {tar: fold_errors[f"test_{tar}_naive"] - fold_errors[f"test_{tar}_label"] \
                    for tar in ['sbp', 'dbp']}
        naive_metric = cal_metric(err_dict, mode='nv')
        out_metric.update(naive_metric)
        #####################################################
        #####################################################
        for mode in ['test']:
            err_dict = {tar: fold_errors[f"{mode}_{tar}_pred"] - fold_errors[f"{mode}_{tar}_label"] \
                        for tar in ['sbp', 'dbp', 'sbp_hypo', 'dbp_hypo', 'sbp_normal', 'dbp_normal',
                                    'sbp_prehyper', 'dbp_prehyper', 'sbp_hyper2', 'dbp_hyper2',
                                    #'sbp_crisis','dbp_crisis'
                                    ]}
            tmp_metric = cal_metric(err_dict, mode=mode)
            out_metric.update(tmp_metric)
        #####################################################
        #####################################################
        results['out_metric'] = out_metric
        os.makedirs(os.path.dirname(self.config.param_test.save_path), exist_ok=True)
        joblib.dump(results, self.config.param_test.save_path)

        print(out_metric)

# %%
