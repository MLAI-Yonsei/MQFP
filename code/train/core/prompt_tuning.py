import pytorch_lightning as pl
import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
import csv
import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from pyts.decomposition import SingularSpectrumAnalysis
import pywt
from core.utils import perform_pca, project_to_pca_plane, loc_z, perform_pca_w_fft, freq_norm, freq_denorm, dbg, register_nan_for_tensor
import wandb
import math

def hook_fn(module, input, output):
    global hidden_output
    hidden_output = output

# def normalizer(x, x_prompted):
#     x_max = x.max(dim=-1, keepdim=True)[0]; x_min = x.min(dim=-1, keepdim=True)[0] # 256, 1, 1
#     x_prompted_max = x_prompted.max(dim=-1, keepdim=True)[0]; x_prompted_min = x_prompted.min(dim=-1, keepdim=True)[0]
#     scale = (x_max - x_min) / (x_prompted_max - x_prompted_min)
#     kk = scale*(x_prompted - x_prompted_min) + x_min
#     return kk

def normalizer(x, x_prompted):
    x_mean = x.mean(dim=-1, keepdim=True)  
    x_std = x.std(dim=-1, keepdim=True) + 1e-6  
    normalized_x_prompted = (x_prompted - x_mean) / x_std
    return normalized_x_prompted

def normalize_keys(keys):
<<<<<<< HEAD
    # Min-Max Scaling
=======
    # Min-Max Scaling을 사용한 정규화
>>>>>>> origin/main
    min_val = keys.min(dim=-1, keepdim=True)[0]
    max_val = keys.max(dim=-1, keepdim=True)[0]
    normalized_keys = (keys - min_val) / (max_val - min_val)
    return normalized_keys

def global_normalizer(prompt, x_min, x_max):
    x_prompted_max = prompt.max(dim=-1, keepdim=True)[0]; x_prompted_min = prompt.min(dim=-1, keepdim=True)[0]
    scale = (x_max - x_min) / (x_prompted_max - x_prompted_min)
    norm_prompt = scale*(prompt - x_prompted_min) + x_min
    return norm_prompt

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        return x
    
class PromptCNN(nn.Module):
    def __init__(self):
        super(PromptCNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.to(self.conv1d.weight.device)
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.squeeze(1)
        return x
    
class SimpleLinear(nn.Module):
    def __init__(self, in_size):
        super(SimpleLinear, self).__init__()
        self.linear = nn.Linear(in_size, in_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        return x

class PPGEmbeddingGenerator(nn.Module):  
    def __init__(self, group_embedding_dim=64, num_groups=4, trunc_length=None, use_group=False):
        super(PPGEmbeddingGenerator, self).__init__()  # Initialize the parent class
        if use_group:
            self.use_group = use_group
            self.group_embedding_dim = group_embedding_dim
            self.group_lookup_table = nn.Embedding(num_groups, group_embedding_dim) if num_groups is not None else None
        
        self.trunc_length=trunc_length
        
    # def pca_transform(self, data, num_components=64):
    #     reshaped_data = data.view(data.size(0), -1).cpu().numpy()
    #     pca = PCA(n_components=min(num_components, reshaped_data.shape[1]))
    #     transformed_data = pca.fit_transform(reshaped_data)
    #     return torch.tensor(transformed_data, dtype=torch.float32, device=data.device)

    def pca_transform(self, data, pca_matrix=None, pca_mean=None):
        transformed_data = project_to_pca_plane(data, pca_matrix, pca_mean)
        return torch.tensor(transformed_data, dtype=torch.float32, device=data.device)
    
    def ssa_transform(self, data, window_size=125, groups=20):
        reshaped_data = data.view(data.size(0), -1).cpu().numpy()
        ssa = SingularSpectrumAnalysis(window_size=min(window_size, reshaped_data.shape[1]//2), groups=min(groups, reshaped_data.shape[1]//5))
        transformed_data = ssa.fit_transform(reshaped_data)
        return torch.tensor(transformed_data, dtype=torch.float32, device=data.device)

    def wavelet_transform(self, data, wavelet_name='db1', level=5):
        transformed_data = []
        for batch in data:
            coeffs = pywt.wavedec(batch.cpu().numpy().flatten(), wavelet_name, level=min(level, int(np.log2(batch.numel()))-1))
            flat_coeffs = np.concatenate(coeffs)
            # transformed_data.append(flat_coeffs[:num_components])
            transformed_data.append(flat_coeffs)
        transformed_data = np.array(transformed_data)
        return torch.tensor(transformed_data, dtype=torch.float32, device=data.device)

    def fft_transform(self, data):
        if self.trunc_length is None:
            num_components = (data.size(2) // 2) + 1 # FFT has symentic matrix
        else:
            num_components = self.trunc_length
        
        reshaped_data = data.view(data.size(0), -1).cpu().numpy()
        transformed_data = np.fft.fft(reshaped_data, axis=1)
        return torch.tensor(np.real(transformed_data[:, :num_components]), dtype=torch.float32, device=data.device)

    def gen_ppg_emb(self, ppg, groups, pca_matrix, pca_mean, multi_query=False):
        # Ensure the input data is in the correct shape
        assert len(ppg.shape) == 3 and ppg.shape[1] == 1, "PPG data should be in shape [batch_size, 1, seq_len]"
        device = ppg.device  # Get the device of the input PPG data

        # Generate Group embedding
        # group_emb = self.group_lookup_table(groups.to(device))  # Move groups to the same device as ppg

        if not multi_query:
            # Generate FFT embedding
            fft_emb = self.fft_transform(ppg)

            # Generate PCA after FFT embedding
            pca_of_fft_emb = self.pca_transform(fft_emb, pca_matrix, pca_mean)

            # Generate Wavelet embedding
            wavelet_emb = self.wavelet_transform(ppg)
            
            return pca_of_fft_emb, fft_emb, wavelet_emb #, pt_emb

        elif multi_query:
             # Generate PCA embedding
            pca_emb = self.pca_transform(ppg, pca_matrix, pca_mean)
            
            # Generate FFT embedding
            fft_emb = self.fft_transform(ppg)

            # Generate Wavelet embedding
            wavelet_emb = self.wavelet_transform(ppg)
            return pca_emb, fft_emb, wavelet_emb #, pt_emb
        
class L2Prompt_stepbystep(nn.Module):
    def __init__(self, config, model_config, x_min, x_max):
        super().__init__()
        self.config = config
        self.num_pool = config.num_pool
        self.model_config = model_config
        self.trunc_dim = config.trunc_dim

        # Query
        if config.pass_pca:
            self.fft_proj = nn.Linear(config.trunc_dim, config.query_dim, bias=False)
            self.norm = nn.InstanceNorm1d(num_features=1, affine=True)
            # self.norm = nn.BatchNorm1d(num_features=1)
        else:
            self.pca_proj = nn.Linear(config.pca_dim, config.query_dim, bias=False)

        self.ppg_embedding_generator = PPGEmbeddingGenerator(self.config.query_dim, trunc_length=self.trunc_dim)
        
        # Key
        self.keys = nn.Parameter(torch.randn(self.num_pool, 1, self.config.query_dim))
        nn.init.uniform_(self.keys,-1,1)

        # Prompts (Value)
        if config.add_freq:
            self.prompts = nn.Parameter(torch.randn(self.num_pool, 1, self.trunc_dim*2 if config.train_imag else self.trunc_dim))
        else:        
            self.prompts = nn.Parameter(torch.randn(self.num_pool, 1, self.model_config["data_dim"]))
        nn.init.uniform_(self.prompts,-1,1)
    
    # def forward(self, x, group_labels, pca_matrix, pca_mean):
    #     bs = x['ppg'].shape[0]
    #     dim = x['ppg'].shape[-1]

    #     # fft2pca_emb: FFT=>PCA (Batch, PCA_DIM)
    #     fft2pca_emb, fft_emb, _ = self.ppg_embedding_generator.gen_ppg_emb(x['ppg'], group_labels, pca_matrix, pca_mean, multi_query=False)

    #     if len(fft2pca_emb.shape) == 1:
    #         fft2pca_emb = fft2pca_emb.unsqueeze(0)

    #     if not self.config.pass_pca:
    #         query = self.pca_proj(fft2pca_emb)  # (B, D_q)
    #         query = query.unsqueeze(1)          # (B, 1, D_q)

    #     elif self.config.pass_pca:
    #         query = fft_emb.unsqueeze(1)
    #         query = self.norm(query)
    #         query = self.fft_proj(query)
    #         # query = query.unsqueeze(1)

    #     d_k = query.size(-1)
    #     qk = torch.einsum('bid,pid->bip', query, self.keys) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    #     gumbel_samples = F.gumbel_softmax(qk, tau=1.0, hard=True)
    #     self.top1_indices = gumbel_samples.argmax(dim=-1).to(torch.int64)
    #     top1_prompts = torch.einsum('bip,pid->bid', gumbel_samples, self.prompts)
    def forward(self, x, group_labels, pca_matrix, pca_mean, step=0):
        bs = x['ppg'].shape[0]
        # ---------------------------------------------------
        fft2pca_emb, fft_emb, _ = self.ppg_embedding_generator.gen_ppg_emb(
            x['ppg'], group_labels, pca_matrix, pca_mean, multi_query=False)
        # dbg("fft2pca_emb", fft2pca_emb, step)

        # ---------------------------------------------------  pca_proj
        # dbg("pca_proj.weight", self.pca_proj.weight, step)
        # if self.pca_proj.weight.grad is not None:
            # dbg("pca_proj.weight.grad", self.pca_proj.weight.grad, step)

        if not self.config.pass_pca:
            query = self.pca_proj(fft2pca_emb)
            # dbg("query_raw", query, step)

            query = query.unsqueeze(1)
            # query = F.layer_norm(query, query.shape[-1:])
            # dbg("query_norm", query, step)
        else:
            query = fft_emb.unsqueeze(1)
            query = self.norm(query)
            query = self.fft_proj(query)
            # dbg("query_fftproj", query, step)
            
        # ---------------------------------------------------  keys
        # dbg("keys", self.keys, step)

        d_k = query.size(-1)
        query_norm = F.layer_norm(query, query.shape[-1:])
        
        # register_nan_for_tensor(query_norm, "query_norm")     # ← ①
        d_k = query_norm.size(-1)

        qk = torch.einsum('bid,pid->bip', query_norm, self.keys) / math.sqrt(d_k)
        # register_nan_for_tensor(qk, "qk")                     # ← ②

        # gumbel_samples = F.gumbel_softmax(qk, tau=1.0, hard=True)
        gumbel_samples = F.gumbel_softmax(qk, tau=1.0, hard=True)

        top1_prompts = torch.einsum('bip,pid->bid', gumbel_samples, self.prompts)

        if self.config.add_freq:
            # fft + propmt
            ppg_fft = torch.fft.fft(x['ppg'], dim=-1)
            ppg_fft_real, freq_real_min, freq_real_max = freq_norm(ppg_fft.real)

            ppg_fft_imag, freq_imag_min, freq_imag_max = freq_norm(ppg_fft.imag)
<<<<<<< HEAD
=======
            # 1번째부터 trunc_dim+1 번째까지 값을 넣고 나머지에 zero padding
>>>>>>> origin/main
            truncated_prompt_real = torch.zeros_like(x['ppg'], dtype=torch.float)
            truncated_prompt_imag = torch.zeros_like(x['ppg'], dtype=torch.float)
            truncated_prompt_real[:,:,1:self.trunc_dim+1] = self.config.global_coeff*top1_prompts[:,:,:self.trunc_dim]
            if self.config.sym_prompt:
                # truncated_prompt_real[:,:,-self.trunc_dim:] = self.config.global_coeff*top1_prompts[:,:,:self.trunc_dim]
                flipped_prompts = torch.flip(top1_prompts[:,:,:self.trunc_dim], dims=[-1])
                truncated_prompt_real[:,:,-self.trunc_dim:] = self.config.global_coeff * flipped_prompts
            
            prompt_add_fft_real = freq_denorm(ppg_fft_real + truncated_prompt_real, freq_real_min, freq_real_max)

            if self.config.train_imag:
                truncated_prompt_imag[:,:,1:self.trunc_dim+1] = self.config.global_coeff*top1_prompts[:,:,self.trunc_dim:]
                if self.config.sym_prompt:
                    # truncated_prompt_imag[:,:,-self.trunc_dim:] = self.config.global_coeff*top1_prompts[:,:,self.trunc_dim:]
                    flipped_prompts = torch.flip(top1_prompts[:,:,self.trunc_dim:], dims=[-1])
                    truncated_prompt_imag[:,:,-self.trunc_dim:] = -1*self.config.global_coeff * flipped_prompts
                    
                prompt_add_fft_imag = freq_denorm(ppg_fft_imag + truncated_prompt_imag, freq_imag_min, freq_imag_max)
            else:
                prompt_add_fft_imag = freq_denorm(ppg_fft_imag, freq_imag_min, freq_imag_max)
            prompt_add_fft = torch.complex(prompt_add_fft_real, prompt_add_fft_imag)

            # prompted_signal = Inverse FFT(prompt_add_fft)
            prompted_signal = torch.fft.ifft(prompt_add_fft, dim=-1).real

            # # save datas
            # saved_data = {
            #     'original_ppg': x['ppg'],
            #     'prompt_add_fft_real': prompt_add_fft_real,
            #     'original_fft_real': prompt_add_fft.real,
            #     'prompted_signal': prompted_signal
            # }

<<<<<<< HEAD
=======
            # # .pt 파일로 저장
>>>>>>> origin/main
            # torch.save(saved_data, "./saved_data.pt")
            
        else:
            # Pure add (ppg + propmt)
            prompted_signal = x['ppg'] + self.config.global_coeff*top1_prompts

        # Calculate entropy penalty to ensure diverse prompt selection
        gumbel_samples = gumbel_samples.squeeze().sum(dim=0)
        probabilities = gumbel_samples.float() / bs
        entropy = -torch.sum(probabilities * torch.log(probabilities+1e-10))
        
        if self.prompts.grad is not None:
            wandb.log({f'Prompts/gradient': wandb.Histogram(self.prompts.grad.cpu().numpy())})
            wandb.log({f'key/gradient': wandb.Histogram(self.keys.grad.cpu().numpy())})

            if self.config.add_freq:
                wandb.log({f'Prompts/weight4real': wandb.Histogram(self.prompts.data[:,:,:self.trunc_dim].detach().cpu().numpy())})
                wandb.log({f'Prompts/weight4imag': wandb.Histogram(self.prompts.data[:,:,self.trunc_dim:].detach().cpu().numpy())})
                wandb.log({f'FFT/real': wandb.Histogram(prompt_add_fft_real.detach().cpu().numpy())})
                wandb.log({f'FFT/imag': wandb.Histogram(prompt_add_fft_imag.detach().cpu().numpy())})
                wandb.log({f'FFT/diff': wandb.Histogram((prompted_signal - x['ppg']).detach().cpu().numpy())})

        sim_loss = 0
        return prompted_signal, sim_loss, entropy

class L2Prompt(nn.Module):
    def __init__(self, config, model_config, x_min, x_max):
        super().__init__()
        self.config = config
        self.x_min = x_min
        self.x_max = x_max
        self.model_config = model_config
        self.trunc_dim = config.trunc_dim
        self.num_pool = config.num_pool
        self.penalty = config.penalty
        self.ppg_embedding_generator = PPGEmbeddingGenerator(config.use_group, self.config.query_dim)
            
        # Projection Matrix for feature querys
        self.pca_proj = nn.Linear(config.pca_dim, config.query_dim, bias=False)
        self.fft_proj = nn.Linear(model_config["data_dim"]//2 + 1, config.query_dim, bias=False)
        self.wavelet_proj = nn.Linear(model_config["wavelet_dim"], config.query_dim, bias=False)
        
        # Initialize learnable parameters for keys and prompts
        self.num_kq = 4 if config.use_pt_emb else 3
        self.keys = nn.Parameter(torch.randn(self.num_pool, self.num_kq, self.config.query_dim))
        nn.init.uniform_(self.keys,-1,1)

        self.prompts = nn.Parameter(torch.randn(self.num_pool, 1, self.model_config["data_dim"]))
        if config.add_freq:
            self.prompts = nn.Parameter(torch.randn(self.num_pool, 1, self.trunc_dim*2 if config.train_imag else self.trunc_dim))
        nn.init.uniform_(self.prompts,-1,1)
    
        # Initialize learnable weights
        self.weight_per_prompt = config.weight_per_prompt
        if self.weight_per_prompt:
            self.learnable_weights = nn.Parameter(self._initialize_weights(self.num_kq, self.num_pool))
        else:
            self.learnable_weights = nn.Parameter(self._initialize_weights(self.num_kq))
        
        # if self.config.prompt_weights == 'attention':
        #     self.attention = torch.nn.MultiheadAttention(embed_dim=self.model_config["data_dim"], num_heads=2)
    
    def _initialize_weights(self, size, num_pool=None):
        if num_pool:
            weights = torch.empty((size, num_pool))
            torch.nn.init.xavier_uniform_(weights)
        else:
            weights = torch.empty((size,1))
            torch.nn.init.xavier_uniform_(weights)
            weights = weights.squeeze()
        return weights
    
    def forward(self, x, group_labels, pca_matrix, pca_mean):
        bz = x['ppg'].shape[0]
        x_ppg = x['ppg']
        
        # Generate PPG embeddings
        pca_emb, fft_emb, wavelet_emb = self.ppg_embedding_generator.gen_ppg_emb(x_ppg, group_labels, pca_matrix, pca_mean, multi_query=True)
        
        if len(pca_emb.shape) == 1:
            pca_emb = pca_emb.unsqueeze(0)
        if len(fft_emb.shape) == 1:
            fft_emb = fft_emb.unsqueeze(0)
        if len(wavelet_emb.shape) == 1:
            wavelet_emb = wavelet_emb.unsqueeze(0)
        # if self.config.use_pt_emb:
        #     if len(pt_emb.shape) == 1:
        #         pt_emb = pt_emb.unsqueeze(0)

        # Project each feature to same dimensions
        pca_emb = self.pca_proj(pca_emb)
        fft_emb = self.fft_proj(fft_emb)
        wavelet_emb = self.wavelet_proj(wavelet_emb)
        emb_list = [pca_emb, fft_emb, wavelet_emb]

        # Optionally add pt_emb if use_pt_emb is True
        # if self.config.use_pt_emb:
        #     pt_emb = self.pt_proj(pt_emb)
        #     emb_list.append(pt_emb)

        queries = torch.stack(emb_list, dim=1)
        
<<<<<<< HEAD
        d_k = queries.size(-1)  # query's dimension
=======
        d_k = queries.size(-1)  # query의 마지막 차원의 크기
>>>>>>> origin/main
        cos_sim = torch.einsum('bqd,nqd->bqn', queries, self.keys) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        gumbel_sample = F.gumbel_softmax(cos_sim, tau=1.0, hard=True)

        top1_prompts = torch.einsum('bki,nd->bkd', gumbel_sample, self.prompts.squeeze(1))

        if self.config.add_freq:
            x_fft = torch.fft.fft(x_ppg, dim=-1)

        # top1_indices = gumbel_sample.argmax(dim=-1).clone().detach().to(torch.int64)
        self.top1_indices = gumbel_sample.argmax(dim=-1).to(torch.int64)

        if self.config.prompt_weights == 'cos_sim':
            # Compute matching scores and apply softmax to get weights
            matching_scores = cos_sim.gather(-1, self.top1_indices.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, 3)
            weights = F.softmax(matching_scores, dim=-1)  # Shape: (batch_size, 3)
        elif self.config.prompt_weights == 'learnable':
            if not self.weight_per_prompt:
                weights = self.learnable_weights.expand([bz, self.learnable_weights.shape[0]])
                weights = F.softmax(weights, dim=-1)
            else:
                # Use learnable weights
                weights = torch.stack([
                    self.learnable_weights[i].gather(0, self.top1_indices[:, i]) for i in range(3)
                ], dim=1)  # Shape: (batch_size, 3)
                weights = F.softmax(weights, dim=-1)  # Shape: (batch_size, 3)
        elif self.config.prompt_weights == 'attention':
            # Use attention to merge original signal and top1_prompts
            if self.config.add_freq:
                values, _, _ = freq_norm(x_fft.real)
                weights = torch.einsum('bcd,bkd->bk', values[:,:,:self.trunc_dim], top1_prompts)
            else:
                weights = torch.einsum('bcd,bkd->bk', x_ppg, top1_prompts)
            weights = F.softmax(weights, dim=-1)

        # Compute weighted sum of prompts
        final_prompt = (weights.unsqueeze(-1) * top1_prompts).sum(dim=1, keepdim=True)  # Shape: (batch_size, 1, prompt_dim)
        
        if self.config.add_freq:
            # fft + propmt
            ppg_fft_real, freq_real_min, freq_real_max = freq_norm(x_fft.real)
            ppg_fft_imag, freq_imag_min, freq_imag_max = freq_norm(x_fft.imag)
<<<<<<< HEAD
            # 1st to trunc_dim+1th value to padding to zero
=======
            # 1번째부터 trunc_dim+1 번째까지 값을 넣고 나머지에 zero padding
>>>>>>> origin/main
            truncated_prompt_real = torch.zeros_like(x_fft, dtype=torch.float)
            truncated_prompt_imag = torch.zeros_like(x_fft, dtype=torch.float)

            truncated_prompt_real[:,:,1:self.trunc_dim+1] = self.config.global_coeff*final_prompt[:,:,:self.trunc_dim]

            if self.config.sym_prompt:
                # truncated_prompt_real[:,:,-self.trunc_dim:] = self.config.global_coeff*final_prompt[:,:,:self.trunc_dim]
                flipped_prompts = torch.flip(final_prompt[:,:,:self.trunc_dim], dims=[-1])
                truncated_prompt_real[:,:,-self.trunc_dim:] = self.config.global_coeff * flipped_prompts
                
            prompt_add_fft_real = freq_denorm(ppg_fft_real + truncated_prompt_real, freq_real_min, freq_real_max)

            if self.config.train_imag:
                truncated_prompt_imag[:,:,1:self.trunc_dim+1] = self.config.global_coeff*final_prompt[:,:,self.trunc_dim:]
                if self.config.sym_prompt:
                    # truncated_prompt_imag[:,:,-self.trunc_dim:] = self.config.global_coeff*final_prompt[:,:,self.trunc_dim:]
                    flipped_prompts = torch.flip(final_prompt[:,:,self.trunc_dim:], dims=[-1])
                    truncated_prompt_imag[:,:,-self.trunc_dim:] = -1*self.config.global_coeff * flipped_prompts
                    
                prompt_add_fft_imag = freq_denorm(ppg_fft_imag + truncated_prompt_imag, freq_imag_min, freq_imag_max)
            else:
                prompt_add_fft_imag = freq_denorm(ppg_fft_imag, freq_imag_min, freq_imag_max)
            prompt_add_fft = torch.complex(prompt_add_fft_real, prompt_add_fft_imag)

            # prompted_signal = Inverse FFT(prompt_add_fft)
            prompted_signal = torch.fft.ifft(prompt_add_fft, dim=-1).real
        else:
            prompted_signal = self.config.lam * x_ppg + (1-self.config.lam) * self.config.global_coeff*final_prompt
        
        # Calculate pull_constraint loss (similarity loss) using cos_sim
        # sim_pull = cos_sim.gather(-1, self.top1_indices.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, 4)
        # sim_pull = cos_sim.gather(-1, self.top1_indices).squeeze(-1)
        # sim_loss = torch.clamp(1 - sim_pull.mean(), min=0)
        sim_loss = 0
        # Negative to maximize similarity

        # top_prompt_idx = self.top1_indices.detach().cpu().numpy()
        # wandb.log({"top_prompt_idx": wandb.Table(data=top_prompt_idx, columns=["PCA", "FFT", "Wave"])})

        # Calculate entropy penalty to ensure diverse prompt selection
        gumbel_samples = gumbel_sample.sum(dim=1).sum(dim=0)
        probabilities = gumbel_samples.float() / (bz*self.num_kq)
        entropy_penalty = -torch.sum(probabilities * torch.log(probabilities+1e-10))
        
        if not self.config.ignore_wandb:
            if self.prompts.grad is not None:
                wandb.log({f'Propmts/gradient': wandb.Histogram(self.prompts.grad.cpu().numpy())})
                wandb.log({f'key/gradient': wandb.Histogram(self.keys.grad.cpu().numpy())})
        
        return prompted_signal, sim_loss, entropy_penalty

class MQFP_wrapper(pl.LightningModule):
    def __init__(self, model, data_shape, model_config, config, stats, fold):
        super().__init__()
        self.fold = fold
        self.config = config
        self.base_model = model
        self.data_shape = data_shape
        self.model_config = model_config
        self.ppg_min = stats[0]
        self.ppg_max = stats[1]

        if not self.config.instance and not self.config.stepbystep:
            self.prompt_learner_glo =L2Prompt(self.config, self.model_config, self.ppg_min, self.ppg_max)
        elif self.config.stepbystep and not self.config.instance:
            self.prompt_learner_glo =L2Prompt_stepbystep(self.config, self.model_config, self.ppg_min, self.ppg_max)
            self.trunc_dim = self.prompt_learner_glo.trunc_dim

        #Loss Function
        if self.config.group_avg:
            self.criterion = nn.MSELoss(reduction="none")
        else:
            self.criterion = nn.MSELoss()
        print('init model')
        
        self.pca_matrix = None
        self.pca_train_mean = 0

        self.raw_input = []
        self.prompted_input = []
        self.hidden_embs = []
        self.prompt_hist = []

    def hook_fn(self, module, input, output):
        self.hidden_output = output

    def embedding_loss(self, h, group):
        self.source_emb = torch.load(f'./embeds_mean/{self.config.backbone}/{self.config.transfer}_mean_embeddings.pt', map_location=self.base_model.device)
        h_source = self.source_emb[group]
        diff_loss = torch.norm(h-h_source, dim=1).mean()
        return diff_loss

    def _shared_step(self, batch, mode):
        x_ppg, y, group, x_abp, peakmask, vlymask = batch
        if (self.pca_matrix == None) & (self.step_mode=="val"):
            merged, sim_loss, entropy_penalty = self.prompt_learner_glo(x_ppg, group, self.sanity_pca_matrix, self.sanity_val_mean)
        else:
            merged, sim_loss, entropy_penalty = self.prompt_learner_glo(x_ppg, group, self.pca_matrix, self.pca_train_mean)
        
        if self.config.normalize:
            merged = normalizer(x_ppg["ppg"], merged)
        if self.config.clip:
            merged = torch.clamp(merged, min=self.ppg_min,max=self.ppg_max)
        
        hidden_emb = self.base_model.extract_penultimate_embedding(merged)

<<<<<<< HEAD
=======
        # hidden_emb + prompt 를 resnet penltimate layer에 삽입 (교체)
        # pred = self.base_model.main_clf(hidden_emb + penulit_emb_prompt)

        # torch.save(merged, "merged_1.pt")
        # pred = self.base_model(merged)

>>>>>>> origin/main
        if self.config.add_prompts == "every" or self.config.add_prompts == 'final':
            pred = self.base_model.model.forward_w_add_prompts(merged)
        else:
            pred = self.base_model(merged)

        if mode=='test' and self.config.get_emb:
            self.prompt_hist.append(self.prompt_learner_glo.top1_indices)
            self.raw_input.append(x_ppg['ppg'].squeeze(dim=1))
            self.prompted_input.append(merged.squeeze(dim=1))
            self.hidden_embs.append(hidden_emb)

        if self.config.group_avg:
            raise ValueError("We do not use group loss")
            losses = self.criterion(pred, y)
            loss = self.grouping(losses, group)
            if self.config.method == "prompt_global":
                loss = loss + self.config.qk_sim_coeff*sim_loss 
                if self.config.penalty:
                    loss = loss + self.config.penalty_scaler*entropy_penalty
                return loss, pred, x_abp, y, group
            return loss, pred, x_abp, y, group

        else:
            loss = self.criterion(pred, y)
<<<<<<< HEAD
            if self.config.use_emb_diff:
                emb_diff_loss = self.embedding_loss(hidden_emb, group)
            else:
                emb_diff_loss = 0
=======
            emb_diff_loss = self.embedding_loss(hidden_emb, group)
>>>>>>> origin/main
            if not self.config.ignore_wandb:
                wandb.log(
                    {f'Fold{self.fold}/{mode}_reg_loss':loss,
                        f'Fold{self.fold}/{mode}_qk_sim_loss':sim_loss,
                        f'Fold{self.fold}/{mode}_entropy':entropy_penalty,
                        f'Fold{self.fold}/{mode}_total_loss': loss + self.config.qk_sim_coeff*sim_loss + self.config.penalty_scaler*entropy_penalty + self.config.diff_loss_weight * emb_diff_loss,
                        f'Fold{self.fold}/{mode}_emb_diff_loss': emb_diff_loss,
                        'epoch': self.current_epoch}
                        )
            if self.config.method == "prompt_global":
                loss = loss + self.config.qk_sim_coeff*sim_loss #- entropy
                if self.config.penalty:
                    loss = loss + self.config.penalty_scaler*entropy_penalty
                if self.config.use_emb_diff:
                    loss = loss + self.config.diff_loss_weight * emb_diff_loss
            
            return loss, pred, x_abp, y, group
        
    def grouping(self, losses, group):
        group_type = torch.arange(0,4).cuda()
        group_map = (group_type.view(-1,1)==group).float()
        group_count = group_map.sum(1)
        group_loss_map = losses.squeeze(0) * group_map.unsqueeze(2) # (4,bs,2)
        group_loss = group_loss_map.sum(1)                         # (4,2)

        # Average only across the existing group
        mask = group_count != 0
        avg_per_group = torch.zeros_like(group_loss)
        avg_per_group[mask, :] = group_loss[mask, :] / group_count[mask].unsqueeze(1)
        exist_group = mask.sum()
        avg_group = avg_per_group.sum(0)/exist_group
        loss = avg_group.sum()/2
        return loss

    def training_step(self, batch, batch_idx):
        self.step_mode = 'train'
        if (self.pca_matrix==None):
            assert len(batch[0]['ppg']==self.config.param_model.batch_size)
            if self.config.stepbystep:
                self.pca_matrix, self.pca_train_mean = perform_pca_w_fft(batch[0]['ppg'], n_components=self.config.pca_dim, trunc_leng=self.trunc_dim)
            else:
                self.pca_matrix, self.pca_train_mean = perform_pca(batch[0]['ppg'], n_components=self.config.pca_dim)
            
        loss, pred_bp, t_abp, label, group = self._shared_step(batch, mode = 'train')
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group} 
    
    def training_epoch_end(self, train_step_outputs):
        logit = torch.cat([v["pred_bp"] for v in train_step_outputs], dim=0)
        label = torch.cat([v["true_bp"] for v in train_step_outputs], dim=0)
        group = torch.cat([v["group"] for v in train_step_outputs], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach(), group)
        self._log_metric(metrics, mode="train")

    def validation_step(self, batch, batch_idx):
        self.step_mode = 'val'
        if (self.pca_matrix == None):
            if self.config.stepbystep:
                self.sanity_pca_matrix = torch.randn((self.trunc_dim, self.config.pca_dim)).cuda() # FFT dim size 313
            else:
                self.sanity_pca_matrix = torch.randn((batch[0]['ppg'].shape[-1], self.config.pca_dim)).cuda()
            
            self.sanity_val_mean = 0 # torch.mean(batch[0]['ppg'], dim=0)
        loss, pred_bp, t_abp, label, group  = self._shared_step(batch, mode='val')
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group}  

    def validation_epoch_end(self, val_step_end_out):
        logit = torch.cat([v["pred_bp"] for v in val_step_end_out], dim=0)
        label = torch.cat([v["true_bp"] for v in val_step_end_out], dim=0)
        group = torch.cat([v["group"] for v in val_step_end_out], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach(), group)
        self._log_metric(metrics, mode="val")
        return val_step_end_out

    def test_step(self, batch, batch_idx):
        self.step_mode = 'test'
        loss, pred_bp, t_abp, label, group = self._shared_step(batch, mode='test')
        self.log('test_loss', loss, prog_bar=True)
        return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group}  

    def test_epoch_end(self, test_step_end_out):
        logit = torch.cat([v["pred_bp"] for v in test_step_end_out], dim=0)
        label = torch.cat([v["true_bp"] for v in test_step_end_out], dim=0)
        group = torch.cat([v["group"] for v in test_step_end_out], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach(), group)
        self._log_metric(metrics, mode="test")
        return test_step_end_out
    
    def _cal_metric(self, logit: torch.tensor, label: torch.tensor, group=None):
        prev_mse = (logit-label)**2
        prev_mae = torch.abs(logit-label)
        prev_me = logit-label
        mse = torch.mean(prev_mse)
        mae = torch.mean(prev_mae)
        me = torch.mean(prev_me)
        std = torch.std(torch.mean(logit-label, dim=1))
        group_mse = self.grouping(prev_mse, group)
        group_mae = self.grouping(prev_mae, group)
        group_me = self.grouping(prev_me, group)
        return {"mse":mse, "mae":mae, "std": std, "me": me, "group_mse":group_mse, "group_mae":group_mae, "group_me":group_me} 
    
    def _log_metric(self, metrics, mode):
        for k,v in metrics.items():
            self.log(f"{mode}_{k}", v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.prompt_learner_glo.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.wd},
            {'params': self.base_model.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.wd}
        ])
        # optimizer = torch.optim.Adam([
        #     {'params': self.prompt_learner_glo.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.wd},
        #     {'params': self.base_model.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.wd}
        # ])
        return optimizer