import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from .base_pl import Regressor
import coloredlogs, logging
from torch import Tensor
import math
from typing import Any, Dict

coloredlogs.install()
logger = logging.getLogger(__name__)  

class PositionalEncoding(nn.Module):

    def __init__(self, dimModel: int, dropout: float = 0.1, maxLen: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(maxLen).unsqueeze(1)
        divTerm = torch.exp(torch.arange(0, dimModel, 2) * (-math.log(10000.0) / dimModel))
        pe = torch.zeros(maxLen, 1, dimModel)
        pe[:, 0, 0::2] = torch.sin(position * divTerm)
        pe[:, 0, 1::2] = torch.cos(position * divTerm)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# https://github.com/nastassiavysotskaya/BP_Transformer/blob/main/train_BP_prediction_models/models.py
class BPTransformer(nn.Module):
    def __init__(self, dimModel, inputFeatureDim, demographicFeatureDim, dropout, nHeads, hiddenDim, num_encoder_layers, num_decoder_layers):
        super(BPTransformer, self).__init__()

        # INFO
        self.modelType = 'Transformer'
        self.dimModel = dimModel

        # LAYERS
        self.embedding = nn.Linear(inputFeatureDim, dimModel) #nn.linear since we have continuous, non-categorical features
        self.posEncoder = PositionalEncoding(dimModel, dropout)
        self.transformer = nn.Transformer(d_model=dimModel, nhead=nHeads, dropout=dropout, batch_first=True, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)

        self.fcForSbp = nn.Linear(dimModel + demographicFeatureDim, hiddenDim)
        self.fcForDbp = nn.Linear(dimModel + demographicFeatureDim, hiddenDim)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.main_clf = nn.Linear(hiddenDim * 2, 2)


    def forward(self, src, tgt, demographicFeatures, srcMask=None, return_penultimate=False):
        if srcMask is not None:
            import pdb; pdb.set_trace()
        src = self.embedding(src)
        src = self.posEncoder(src)

        tgt = self.embedding(tgt)
        tgt = self.posEncoder(tgt)

        if srcMask is not None:
            output = self.transformer(src=src, tgt=tgt, src_mask=srcMask)
        else:
            output = self.transformer(src=src, tgt=tgt)

        if demographicFeatures is not None:
            combinedFeatures = torch.cat((output[:, 0, :], demographicFeatures), dim=-1)
        else:
            combinedFeatures = output[:, 0, :]

        outputSbp = self.relu1(self.fcForSbp(combinedFeatures))
        outputDbp = self.relu2(self.fcForDbp(combinedFeatures))
        
        output = self.main_clf(torch.cat([outputSbp, outputDbp], dim=-1))
        if return_penultimate:
            return torch.cat([outputSbp, outputDbp], dim=-1)
        else:
            return output


class BPTransformerRegressor(Regressor):
    def __init__(self, param_model: Any, random_state: int = 0):
        super().__init__(param_model, random_state)

        self.model = BPTransformer(
            dimModel              = param_model.dim_model,
            inputFeatureDim       = param_model.input_feature_dim,
            demographicFeatureDim = param_model.demographic_feature_dim,
            dropout               = param_model.dropout,
            nHeads                = param_model.n_heads,
            hiddenDim             = param_model.hidden_dim,
            num_encoder_layers    = param_model.num_encoder_layers,
            num_decoder_layers    = param_model.num_decoder_layers,
        )

    def _shared_step(self, batch):
        x_ppg, y, group, x_abp, peakmask, vlymask = batch
        x_ppg = x_ppg['ppg'].transpose(1, 2)
        
        src = x_ppg[:, :-1, :]            # L-1 step
        tgt = x_ppg[:, -1:, :]            # decoder input

        demo = None                       
        pred_bp = self.model(src, tgt, demo)      # forward

        loss = self.criterion(pred_bp, y)                 
        return loss, pred_bp, x_abp, y
    
    def extract_penultimate_embedding(self, x_ppg):
        x_ppg = x_ppg.transpose(1, 2)
        src = x_ppg[:, :-1, :]            # L-1 step
        tgt = x_ppg[:, -1:, :]            # decoder input

        demo = None                       
        penultimate_embedding = self.model(src=src, tgt=tgt, demographicFeatures=demo, return_penultimate=True)      # forward
        return penultimate_embedding
    
    def forward(self, x):
        x = x.transpose(1, 2)
        src = x[:, :-1, :]            # L-1 step
        tgt = x[:, -1:, :]            # decoder input

        demo = None                       
        x = self.model(src=src, tgt=tgt, demographicFeatures=demo, return_penultimate=False)      # forward
        return x

    def training_step(self, batch, batch_idx):
        loss, pred_bp, t_abp, label = self._shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return {"loss": loss, "pred_bp": pred_bp,
                "true_abp": t_abp, "true_bp": label}

    def training_epoch_end(self, outputs):
        logit = torch.cat([o["pred_bp"] for o in outputs], 0)
        label = torch.cat([o["true_bp"] for o in outputs], 0)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="train")

    def validation_step(self, batch, batch_idx):
        loss, pred_bp, t_abp, label = self._shared_step(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {"loss": loss, "pred_bp": pred_bp,
                "true_abp": t_abp, "true_bp": label}

    def validation_epoch_end(self, outputs):
        logit = torch.cat([o["pred_bp"] for o in outputs], 0)
        label = torch.cat([o["true_bp"] for o in outputs], 0)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="val")

    def test_step(self, batch, batch_idx):
        loss, pred_bp, t_abp, label = self._shared_step(batch)
        self.log('test_loss', loss, prog_bar=True)
        return {"loss": loss, "pred_bp": pred_bp,
                "true_abp": t_abp, "true_bp": label}

    def test_epoch_end(self, outputs):
        logit = torch.cat([o["pred_bp"] for o in outputs], 0)
        label = torch.cat([o["true_bp"] for o in outputs], 0)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="test")