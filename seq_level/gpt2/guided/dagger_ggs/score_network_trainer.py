from typing import Optional

from copy import deepcopy
from pprint import pformat

from scipy.stats import kendalltau


import math
import shutil
import numpy as np
import logging

from pytorch_lightning import (LightningModule, 
                               LightningDataModule,
                               Trainer)

from pytorch_lightning.plugins import DDPPlugin, DDPSpawnPlugin


from torch.optim import Adadelta, Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import seq_level.gpt2.guided.dagger_ggs.score_network as score_network

from seq_level.gpt2.guided.dagger_ggs.ggs_efficient_utils import (InstanceType, 
                                                                  get_dataloader, 
                                                                 )
import seq_level.gpt2.guided.utils as ggs_utils


class ScorerAnalysis:
    def __init__(self, prefix):
        self.prefix = prefix
        self.reset()

    def __call__(self, idx, type, true_dist, pred_dist, kl_dist=None, tv_dist=None):
        batch_size = true_dist.size(0)
        self.num_docs += batch_size

        l2dist = (((true_dist - pred_dist)**2)**0.5)
        self.cuml_l2dist += l2dist.sum().item()

        self.true_dists += true_dist.tolist()
        self.pred_dists += pred_dist.tolist()
        self.l2dists += l2dist.tolist()

        scorer_fit = torch.abs((true_dist - pred_dist)/true_dist)\
                            .sum().item()
        
        self.cuml_scorer_fit_all += scorer_fit

        if idx not in self.scorer_diff_fit_dict:
            self.scorer_diff_fit_dict[idx] = \
                {'non_pert': None, 'perturbed': [],}
        
        self.cuml_pred_dist += pred_dist.sum().item()
        self.cuml_true_dist += true_dist.sum().item()
        if type == InstanceType.PERTURBED:
            self.num_docs_pert += batch_size
            self.cuml_scorer_fit_pert += scorer_fit
            self.cuml_pert_l2dist += l2dist.sum().item()
            self.cuml_pred_dist_pert += pred_dist.sum().item()
            self.cuml_true_dist_pert += true_dist.sum().item()

            self.true_dist_pert += true_dist.tolist()
            self.pred_dist_pert += pred_dist.tolist()
            self.l2dists_pert += l2dist.tolist()
            self.scorer_diff_fit_dict[idx]['perturbed'].append((true_dist, 
                                                                pred_dist))
        else:
            self.num_docs_non_pert += batch_size
            self.cuml_scorer_fit_non_pert += scorer_fit
            self.cuml_non_pert_l2dist += l2dist.sum().item()
            self.cuml_pred_dist_non_pert += pred_dist.sum().item()
            self.cuml_true_dist_non_pert += true_dist.sum().item()

            self.true_dist_non_pert += true_dist.tolist()
            self.pred_dist_non_pert += pred_dist.tolist()
            self.l2dists_non_pert += l2dist.tolist()
            self.scorer_diff_fit_dict[idx]['non_pert'] = (true_dist, 
                                                          pred_dist)

    def reset(self):
        self.scorer_diff_fit_dict = {}
        self.cuml_l2dist = 0.
        self.cuml_pert_l2dist = 0.
        self.cuml_non_pert_l2dist = 0.

        self.cuml_scorer_fit_all = 0.
        self.cuml_scorer_fit_pert = 0.
        self.cuml_scorer_fit_non_pert = 0.

        self.num_docs = 1
        self.num_docs_pert = 1
        self.num_docs_non_pert = 1

        self.l2dists = []
        self.true_dists = []
        self.pred_dists = []

        self.true_dist_pert = []
        self.true_dist_non_pert = []
        self.pred_dist_pert = []
        self.pred_dist_non_pert = []
        self.l2dists_pert = []
        self.l2dists_non_pert = []

        self.disagreements = 0.
        self.mean_disagreement = 0.
        self.disagreement_count = 1
        
        self.cuml_pred_dist = 0.
        self.cuml_true_dist = 0.
        
        self.cuml_pred_dist_pert = 0.
        self.cuml_pred_dist_non_pert = 0.
        self.cuml_true_dist_pert = 0.
        self.cuml_true_dist_non_pert = 0.

        self.kl_dist = None
        self.tv_dist = None

    def get_metrics(self):
        for idx, diff_dict in self.scorer_diff_fit_dict.items():
            non_pert_true_score, non_pert_pred_score = diff_dict['non_pert']

            pertured_scores = diff_dict['perturbed']

            for perturb_true_score, perturb_pred_score in pertured_scores:
                pred_diff = perturb_pred_score - non_pert_pred_score
                true_diff = perturb_true_score - non_pert_true_score

                if torch.any(true_diff == 0):
                    perturb_true_score_list = str(perturb_true_score.tolist())
                    non_pert_true_score_list = str(non_pert_true_score.tolist())
                    logging.debug(f"For idx: {idx}," + 
                        f" perturbed score: {perturb_true_score_list}" + 
                        f" non_pert score: {non_pert_true_score_list}" + 
                         " are equal.")
                    continue

                self.disagreements += (pred_diff/true_diff < 0).sum().item()
                self.mean_disagreement += torch.abs((pred_diff - true_diff)/true_diff).sum().item()
                self.disagreement_count += 1

        
        l2dists = np.array(self.l2dists)
        l2dists_pert = np.array(self.l2dists_pert)
        l2dists_non_pert = np.array(self.l2dists_non_pert)
        return {
          f"{self.prefix}/corr_all": kendalltau(self.true_dists, self.pred_dists)[0],
          f"{self.prefix}/corr_pert": kendalltau(self.true_dist_pert, self.pred_dist_pert)[0],
          f"{self.prefix}/corr_non_pert": kendalltau(self.true_dist_non_pert,
                                                     self.pred_dist_non_pert)[0],

          f"{self.prefix}/l2dist_mean": self.cuml_l2dist / self.num_docs,
          f"{self.prefix}/l2dist_mean_pert": self.cuml_pert_l2dist / self.num_docs_pert,
          f"{self.prefix}/l2dist_mean_non_pert": self.cuml_non_pert_l2dist / self.num_docs_non_pert, 

          f"{self.prefix}/l2dist_u1": np.mean(l2dists < 1) ,
          f"{self.prefix}/l2dist_pert_u1": np.mean(l2dists_pert < 1),
          f"{self.prefix}/l2dist_non_pert_u1": np.mean(l2dists_non_pert < 1),

          f"{self.prefix}/l2dist_u10": np.mean(l2dists < 10) ,
          f"{self.prefix}/l2dist_pert_u10": np.mean(l2dists_pert < 10),
          f"{self.prefix}/l2dist_non_pert_u10": np.mean(l2dists_non_pert < 10), 

          f"{self.prefix}/l2dist_u20": np.mean(l2dists < 20) ,
          f"{self.prefix}/l2dist_pert_u20": np.mean(l2dists_pert < 20),
          f"{self.prefix}/l2dist_non_pert_u20": np.mean(l2dists_non_pert < 20), 

          f"{self.prefix}/l2dist_u30": np.mean(l2dists < 30) ,
          f"{self.prefix}/l2dist_pert_u30": np.mean(l2dists_pert < 30),
          f"{self.prefix}/l2dist_non_pert_u30": np.mean(l2dists_non_pert < 30), 

          f"{self.prefix}/scorer_fit_all": self.cuml_scorer_fit_all / self.num_docs,
          f"{self.prefix}/scorer_fit_pert": self.cuml_scorer_fit_pert / self.num_docs_pert,
          f"{self.prefix}/scorer_fit_non_pert": self.cuml_scorer_fit_non_pert / self.num_docs_non_pert,

          f"{self.prefix}/disagreements": self.disagreements / self.num_docs,
          f"{self.prefix}/mean_disagreements": self.mean_disagreement / self.num_docs,

          f"{self.prefix}/true_dist_mean": self.cuml_true_dist/self.num_docs, 
          f"{self.prefix}/true_dist_pert_mean": self.cuml_true_dist_pert/self.num_docs_pert, 
          f"{self.prefix}/true_dist_non_pert_mean": self.cuml_true_dist_non_pert/self.num_docs_non_pert,

          f"{self.prefix}/pred_dist_mean":self.cuml_pred_dist/self.num_docs, 
          f"{self.prefix}/pred_dist_pert_mean":self.cuml_pred_dist_pert/self.num_docs_pert,
          f"{self.prefix}/pred_dist_non_pert_mean":self.cuml_pred_dist_non_pert/self.num_docs_non_pert, 
        }

class ScoringNetworkTrainingDataset(Dataset):
    def __init__(self, buffer, split, args, device=0):
        self.args = args
        if split == 'train':
            iterator = buffer.train_queue
        elif split == 'valid':
            iterator = buffer.valid_queue
        else:
            raise ValueError(f"Illegal Split: {split}")

        self.iterator = iterator
        self.buffer = buffer
        self.idx2model_info = buffer._generate_model_info(iterator, 
                                min_context_len=args.context_length + 2)
        self.idx2model_keys = [x for x in self.idx2model_info.keys()]
        self.device = device

    def __len__(self):
        return len(self.idx2model_keys)

    def __getitem__(self, idx):
        data_key = self.idx2model_keys[idx]
        return self.idx2model_info[data_key] #self.buffer._get_model_at_idx(self.idx2model_info, 
                          #        self.args, data_key, self.device)

class ScoreNetworkDataModule(LightningDataModule):
    def __init__(self, args, buffer):
      super().__init__()
      self.args = args
      self.buffer = buffer

    def prepare_data(self):
      world_size = torch.cuda.device_count()
      for idx, (k,v) in enumerate(self.buffer.db.items()):
          self.buffer.db[k] = v.to(f'cuda:{idx % world_size}')
      pass

    def setup(self, stage: Optional[str] = None):
      self.train_data = ScoringNetworkTrainingDataset(self.buffer, 
                                                      'train', self.args)
      self.val_data = ScoringNetworkTrainingDataset(self.buffer, 
                                                    'valid', self.args)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=1, 
                collate_fn=lambda x: x[0], num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, 
                collate_fn=lambda x: x[0], num_workers=0, pin_memory=True)

        

    def transfer_batch_to_device(self, data, device, dataloader_idx):
        idx2model = {}
        batch = self.buffer.db[data['batch_key']]
        model = self.buffer.db[data['model_key']]

        batch = batch.to(device)
        model = model.to(device)

        idx2model['idx'] = data['idx']
        idx2model['batch'] = batch
        idx2model['model'] = model

        inp, target = batch[:, :-1], batch[:, 1:]

        sequences_key_np, distances_np, _, _, _, _ = data['non_pert']

        sequence_np = self.buffer.db[sequences_key_np]
        sequence_np = sequence_np.to(device)
        distances_np = distances_np.to(device)
        idx2model['non_pert'] = (sequence_np, distances_np)

        idx2model['pert'] = []
        model_ = deepcopy(model)

        model_.eval()
        with torch.set_grad_enabled(True):
            model_with_grad, _ = ggs_utils.mle_grad(model_, inp, target, 
                                    self.args.pad_token_id, self.args.max_grad_norm)

        for (sequences_key_p, distances_p, rng_state, 
                    log_rho, noise_mag,  perturb_type) in data['pert']:
            sequence_p = self.buffer.db[sequences_key_p]
            sequence_p = sequence_p.to(device)
            distances_p = distances_p.to(device)
            if self.args.zero_dist_only and \
                not perturb_type != ggs_utils.PerturbationType.ONLY_NOISE:
                    continue

            if self.args.mle_dist_only and \
                not perturb_type != ggs_utils.PerturbationType.MLE_GRAD_W_NOISE:
                    continue

            model_p, log_rho_, noise_mag_, rng_ = ggs_utils.perturb_single(model, 
                model_with_grad, self.args.ggs_noise, noise_scale=self.args.noise_scale,
                                perturb_type=perturb_type, rng_state=rng_state)

            entry = (model_p, sequence_p, distances_p, log_rho_, noise_mag_)
            idx2model['pert'].append(entry)
        return idx2model

class ScoreNetworkTrainingModule(LightningModule):
    def __init__(self, args, score_network):
      super().__init__()
      self.args = args
      self.score_network = score_network
      self.train_scorer_analysis = ScorerAnalysis('train')
      self.valid_scorer_analysis = ScorerAnalysis('valid')
    
    def forward(self, data):
        dists_and_preds = []
        batch = data['batch']
        model_np = data['model']

        sequences_np, distances_np = data['non_pert']
        model_np = model_np
        sequences_np = sequences_np
        distances_np = distances_np
        batch = batch

        model_np.eval()
        pred_dist_np = self.score_network(model_np, batch, 
                                  predictions=sequences_np)
        model_np.train()

        dists_and_preds.append((distances_np, pred_dist_np, 
                                None, InstanceType.NON_PERTURBED))
        
        pertubed_data = []
        if not self.args.only_train_on_non_pert_data:
          pertubed_data = data['pert']
  
        for model_p, sequences_p, distances_p, log_rho, noise_mag in pertubed_data:
            model_p = model_p
            sequences_p = sequences_p
            distances_p = distances_p
            
            model_p.eval()
            pred_dist_p = self.score_network(model_p, batch, 
                                     predictions=sequences_p)
            model_p.train()
            dists_and_preds.append((distances_p, pred_dist_p, log_rho, InstanceType.PERTURBED))
        return dists_and_preds

    def training_step(self, data, batch_idx):
        dists_and_preds = self.forward(data)
        batch_loss = self.get_loss(dists_and_preds, 
                      self.args.scorer_loss_func, 
                      self.args.ggs_beta)
        
        for distances, pred_distances, _, type in dists_and_preds:
            pred_distances = pred_distances.detach()
            self.train_scorer_analysis(data['idx'], type, 
                            distances, pred_distances)
        return {'loss': batch_loss.mean(), 
                'dists_and_preds': dists_and_preds}

    def validation_step(self, data, batch_idx):
        dists_and_preds = self.forward(data)
        batch_loss = self.get_loss(dists_and_preds, 
                      self.args.scorer_loss_func, 
                      self.args.ggs_beta)

        for distances, pred_distances, _, type in dists_and_preds:
            pred_distances = pred_distances.detach()
            self.valid_scorer_analysis(data['idx'], type, 
                                distances, pred_distances)
        self.log("loss", batch_loss.mean())
        return {'loss': batch_loss.mean(), 
                'dists_and_preds': dists_and_preds}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.score_network.parameters(), lr=self.args.scorer_lr)
        return [optimizer], [StepLR(optimizer, step_size=1, gamma=0.5)]

    def train_epoch_end(self, outputs):
        train_info_dict = self.train_scorer_analysis.get_metrics()
        if self.global_rank == 0:
            print()
            print(pformat(train_info_dict))
        self.log_dict(train_info_dict)

    def validation_epoch_end(self, outputs):
        val_loss = np.mean([x['loss'].item() for x in outputs])
        valid_info_dict = self.valid_scorer_analysis.get_metrics()
        valid_info_dict['loss'] = val_loss
        if self.global_rank == 0:
            print()
            print(pformat(valid_info_dict))
        self.log_dict(valid_info_dict)

    def get_loss(self, data, loss_func_type='mse', ggs_beta=1.0):
        loss = 0.
        if loss_func_type == 'mse':
            for (dist, pred_dist, _, _) in data:
                loss += F.mse_loss(
                            pred_dist,
                            dist.view(-1, 1),
                            reduction='none',)
            loss /= len(data)
        elif loss_func_type == 'mse-diff':
            non_perturb_data = data[0]
            assert InstanceType.NON_PERTURBED == non_perturb_data[-1]

            a = (non_perturb_data[0].view(-1, 1) - non_perturb_data[1])**2
            loss += a
            for perturb_data in  data[1:]:
                assert InstanceType.PERTURBED == perturb_data[-1]
                b = (perturb_data[0].view(-1, 1) - perturb_data[1])**2
                loss += b
                loss += ((perturb_data[0] - non_perturb_data[0]).view(-1, 1) - (perturb_data[1] - non_perturb_data[1]))**2
                loss /= 3
            loss /= len(data[1:])
        elif loss_func_type == 'kl' or loss_func_type == 'tv':
            def compute_weight(dist, perturb_dist, log_rhos, beta):
                ws = beta * (dist - perturb_dist).clamp(max=1e16)
                ws = ws - log_rhos.to(ws.device)
                log_ws = torch.log_softmax(ws, dim=-1)
                return log_ws

            non_perturb_data = data[0]
            assert InstanceType.NON_PERTURBED == non_perturb_data[-1]
            curr_true_dist, curr_pred_dist =  non_perturb_data[:2]
            curr_true_dist = curr_true_dist.view(-1, 1)
            curr_pred_dist = curr_pred_dist.view(-1, 1)
            perturb_true_dist = []
            perturb_pred_dist = []
            log_rhos = []
            for perturb_data in  data[1:]:
                perturb_true_dist.append(perturb_data[0])
                perturb_pred_dist.append(perturb_data[1].squeeze(-1))
                log_rhos.append(perturb_data[2].item())

            perturb_true_dist = torch.stack(perturb_true_dist, dim=-1)
            perturb_pred_dist = torch.stack(perturb_pred_dist, dim=-1)
            log_rhos = torch.tensor(log_rhos)
            true_weights = compute_weight(curr_true_dist, perturb_true_dist, log_rhos, ggs_beta)
            pred_weights = compute_weight(curr_pred_dist, perturb_pred_dist, log_rhos, ggs_beta)
            if loss_func_type == 'kl':
                loss = F.kl_div(pred_weights, true_weights, 
                                reduction='none', log_target=True) \
                            .sum(dim=-1, keepdim=True)
            elif loss_func_type == 'tv':
                loss = 0.5 * torch.abs(torch.exp(pred_weights) - torch.exp(true_weights)) \
                            .sum(dim=-1, keepdim=True)
        return loss

def train_score_network_lightning(buffer, score_network, tokenizer, 
            args, train_score_network_iteration=0, epochs=100,):
    datamodule = ScoreNetworkDataModule(args, buffer)
    model = ScoreNetworkTrainingModule(args, score_network)
    trainer = Trainer(
                gpus=-1 if args.multigpu else 1, 
                strategy=DDPSpawnPlugin(find_unused_parameters=False),
                max_epochs=epochs,
                num_sanity_val_steps=0)

    trainer.fit(model=model, 
                datamodule=datamodule)

def validate_score_network(valid_iter, score_network, tokenizer, device, args):
    cuml_valid_loss = 0.
    num_docs = 1
    cuml_kl_dist = 0.
    cuml_tv_dist = 0.
    if not args.multigpu or device == 0:
        valid_scorer_analysis = ScorerAnalysis('valid')

    score_network.eval()
    for step, data in enumerate(valid_iter):
        losses, distances_and_pred_distances = \
            get_loss(data, score_network, tokenizer, device, args)

        batch_loss = losses.sum().item()
        batch_size = losses.size(0)
        if batch_loss < 0:
            continue

        cuml_valid_loss += batch_loss
        num_docs += batch_size
        if not args.multigpu or device == 0:
            kl_dist = calculate_scorer_loss(distances_and_pred_distances, 'kl')
            tv_dist = calculate_scorer_loss(distances_and_pred_distances, 'tv')
            cuml_kl_dist += kl_dist.sum().item()
            cuml_tv_dist += tv_dist.sum().item()
            for distances, pred_distances, _, type in distances_and_pred_distances:
                pred_distances = pred_distances.squeeze(1).detach()
                valid_scorer_analysis(data['idx'], type, distances, pred_distances)

        if not args.multigpu or device == 0:
            print('Validation:: Step: %5d, Loss: %6.3f'
                  % (step, cuml_valid_loss/num_docs), end='\r')
    score_network.train()

    valid_info_dict = None 
    if not args.multigpu or device == 0:
        print()
        valid_info_dict = valid_scorer_analysis.get_metrics()
        valid_info_dict['valid/kl_dist'] = cuml_kl_dist/num_docs
        valid_info_dict['valid/tv_dist'] = cuml_tv_dist/num_docs

    return cuml_valid_loss/num_docs, valid_info_dict


def train_score_network_multigpu(buffers, score_network, tokenizer, 
            args, train_score_network_iteration=0, epochs=100, world_size=1):
    world_size = torch.cuda.device_count()
    ctx = mp.get_context('spawn')
    queue = ctx.Queue(10000)
    processes = []

    for idx, (k,v) in enumerate(buffers.db.items()):

        buffers.db[k] = v.to(f'cuda:{idx % 4}')

    for rank in range(world_size):
        p = mp.Process(target=train_score_network_helper_v3, 
                args=(rank, score_network, buffers, tokenizer, args,
                    train_score_network_iteration, epochs, world_size, queue))
        processes.append(p)
        p.start()

    while True:
        try:
            data = queue.get(timeout=100)
            utils.log_tensorboard(*data)
            del data
        except Exception as e:
            print(e)
            break

    for p in processes:
        p.join()


def train_score_network_helper_v3(device, score_network, buffers, tokenizer, 
    args, train_score_network_iteration=0, epochs=100, world_size=1, queue=None):

    multigpu_setup(device, world_size)
    train_dataloader, valid_dataloader = buffers.get_dataloaders(args, device, world_size)
    score_network = score_network.to(device)
    score_network = DDP(score_network, device_ids=[device], 
                        output_device=device, find_unused_parameters=False)
            
    train_score_network(device, score_network, tokenizer, 
                        train_dataloader, valid_dataloader, args, 
                        train_score_network_iteration, epochs=epochs, queue=queue)
    cleanup()


def train_score_network(device, score_network, tokenizer, train_dataloader, valid_dataloader, 
                args, train_score_network_iteration=0, epochs=100, queue=None):
    """ This method takes in the scoring data (B) and learns a parameterized scoring model (S) to 
        mimic the original cost function (C) by minimizing the L2 loss.
        $C$ is defined as
            C(\theta) = 1/|B| \sum_{i=1}^{B} c(y_i, F(x_i, \theta))

        The scoring function optimizes the following equation. 
            min_{W,b} \sum_{(x_i, y_i, yo_i, yp_i, \Delta_i) in B} || S(x_i, \theta, \Delta; W, b) - (C(\theta) - C(\theta + \Delta))||^2
            where S(x_i, \theta, \Delta; W, b) = W^T[R(x;\theta) - R(x;\theta+\Delta)] + b
    """
    min_valid_loss = math.inf
    best_score_network = None
    patience_counter = 0

    if not args.multigpu or device == 0:
        print('=' * 100)
        print('Start training the score network.\n')

    phi_optimizer = Adam(score_network.parameters(), lr=args.scorer_lr)
    scheduler = ReduceLROnPlateau(phi_optimizer, 'min', patience=3, verbose=True)

    for epoch in range(epochs):
        score_network.train()
        if not args.multigpu or device == 0:
            train_scorer_analysis = ScorerAnalysis('train')
        cuml_kl_dist = 0.
        cuml_tv_dist = 0.
        cuml_train_loss = 0.
        num_docs = 1
        train_dataloader.sampler.set_epoch(epoch)

        with timer_context('train_score_network_time') as ctxt_timer:
            start_time = timer()
            for step, data in enumerate(train_dataloader):
                phi_optimizer.zero_grad()

                losses, distances_and_pred_distances = \
                    get_loss(data, score_network, tokenizer, device, args)

                batch_size = losses.size(0)
                loss = losses.sum()/batch_size

                if loss < 0 or torch.isnan(loss):
                    continue
                
                if not args.multigpu or device == 0:
                    kl_dist = calculate_scorer_loss(distances_and_pred_distances, 'kl')
                    tv_dist = calculate_scorer_loss(distances_and_pred_distances, 'tv')
                    cuml_kl_dist += kl_dist.sum().item()
                    cuml_tv_dist += tv_dist.sum().item()
                    for distances, pred_distances, _, type in distances_and_pred_distances:
                        pred_distances = pred_distances.squeeze(1).detach()
                        train_scorer_analysis(data['idx'], type, distances, pred_distances)

                cuml_train_loss += losses.sum().item()
                num_docs += batch_size
                loss.backward()
                phi_optimizer.step()

                if (not args.multigpu or device == 0):
                    print('Training:: Epoch: %3d :: Step: %5d/%-5d, Loss: %6.3f' % (epoch, step, len(train_dataloader), cuml_train_loss / num_docs),
                        end='\r')

            end_time = timer()
            if not args.multigpu or device == 0:
                print()
                train_info_dict = train_scorer_analysis.get_metrics()
                train_info_dict['train/kl_dist'] = cuml_kl_dist/num_docs
                train_info_dict['train/tv_dist'] = cuml_tv_dist/num_docs
                print(pformat(train_info_dict))

            ctxt_timer.timeit(start_time, end_time)

        with timer_context('validate_score_network_time') as ctxt_timer:
            start_time = timer()
            train_loss = cuml_train_loss / num_docs
            valid_loss, valid_info_dict = validate_score_network(
                valid_dataloader, score_network, tokenizer, device, args)

            if min_valid_loss < valid_loss:
                patience_counter += 1
            else:
                patience_counter = 0
                min_valid_loss = valid_loss
                best_score_network = deepcopy(score_network)

                if (not args.multigpu or device == 0) and args.save_score_network:
                    if args.multigpu:
                        best_score_network = best_score_network.module
                    
                    score_network_epoch_filepath = os.path.join(args.save_base_dir,
                                                        f'score_network_epoch_{epoch}.pkl')
                    score_network_best_filepath = os.path.join(args.save_base_dir,
                                                        f'score_network_best.pkl')
                    print(f"Saving score network::{score_network_epoch_filepath}")
                    print(f"Saving score network::{score_network_best_filepath}")
                    torch.save({
                        'model_state_dict': best_score_network.state_dict(),
                        'epochs': epoch,
                        'dataset_size': len(train_dataloader),
                    }, score_network_epoch_filepath)
                    shutil.copy(score_network_epoch_filepath, score_network_best_filepath)

            if not args.multigpu or device == 0:
                print(pformat(valid_info_dict))
                print('Epoch: %d :: Train Loss: %.3f, ' % (epoch, train_loss) +
                            'Best Valid Loss: %.3f, Valid Loss: %.3f, Epochs Since Last Best: %d '
                            % (min_valid_loss, valid_loss, patience_counter))
                print(f"Train score network epoch {epoch} done!")
                print(f"Avg Epoch Time: " +
                    f"Train: {timer_context('train_score_network_time').avg_time():.3f} " +
                    f"Valid: {timer_context('validate_score_network_time').avg_time():.3f}.")

                prefix = f"tsn-{train_score_network_iteration}/"
                valid_metrics = {
                    prefix + "train_loss": train_loss,
                    prefix + "valid_loss": valid_loss,
                    prefix + "min_valid_loss": min_valid_loss,
                    prefix + "epochs_since_best": patience_counter,
                    prefix + "current_epoch": epoch,
                }

                for key, val in valid_info_dict.items():
                    valid_metrics[prefix + key] = val

                for key, val in train_info_dict.items():
                    valid_metrics[prefix + key] = val

                if args.multigpu and queue is not None:
                    queue.put((valid_metrics, epoch))
                else:
                    utils.log_tensorboard(valid_metrics, epoch)

            scheduler.step(valid_loss)
            # scheduler.step()

            if patience_counter > args.train_score_patience:
                print(f"Stopping Early at epoch: {epoch} with best validation loss: {min_valid_loss}")
                break

            end_time = timer()
            ctxt_timer.timeit(start_time, end_time)

    if not args.multigpu or device == 0:
        print('Done training the score network.\n')
        print('=' * 100)
        score_network = best_score_network

def add_args(parser):
    parser.add_argument(
        "--score-network-epochs", type=int, default=20,
    )
    parser.add_argument(
        "--retrain-score-network-epochs", type=int, default=30,
    )

    parser.add_argument(
        "--retrain-score-network-every", type=int, default=500
    )
    parser.add_argument(
        "--use-saved-score-network", action='store_true',
    )
    parser.add_argument(
        "--score-network-file", type=str,
    )
    parser.add_argument(
        "--save-score-network", action='store_true',
    )
    parser.add_argument(
        "--train-score-patience", type=int, default=30,
    )
    parser.add_argument(
        "--scorer-lr", type=float, default=1e-4,
    )

    parser.add_argument(
        "--scorer-loss-func", type=str, default="mse",
    )

    parser.add_argument(
        "--initialize-score-network", action='store_true',
    )

    parser = score_network.add_args(parser)
    return parser