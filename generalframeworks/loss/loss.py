import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from generalframeworks.utils import simplex
from math import pi
from generalframeworks.networks.ddp_model import concat_all_gather
from generalframeworks.utils import watch_dist, watch, watch_mmt_prt
import torch.distributed as dist
from torch.cuda.amp.autocast_mode import autocast

class Attention_Threshold_Loss(nn.Module):
    def __init__(self, strong_threshold):
        super(Attention_Threshold_Loss, self).__init__()
        self.strong_threshold = strong_threshold

    def forward(self, pred: torch.Tensor, pseudo_label: torch.Tensor, logits: torch.Tensor):
        batch_size = pred.shape[0]
        valid_mask = (pseudo_label >= 0).float() # only count valid pixels (class)
        weighting = logits.view(batch_size, -1).ge(self.strong_threshold).sum(-1) / (valid_mask.view(batch_size, -1).sum(-1)) # May be nan if the whole target is masked in cutout
        #self.tmp_valid_num = logits.ge(self.strong_threshold).view(logits.shape[0], -1).float().sum(-1).mean(0)
        # weight represent the proportion of valid pixels in this batch
        loss = F.cross_entropy(pred, pseudo_label, reduction='none', ignore_index=-1) # pixel-wise
        weighted_loss = torch.mean(torch.masked_select(weighting[:, None, None] * loss, loss > 0))
        # weight torch.size([4]) -> weight[:, None, None] torch.size([4, 1, 1]) for broadcast to multiply the weight to the corresponding class
        # torch.masked_select to select loss > 0 only leaved 
        
        return weighted_loss

class Prcl_Loss(nn.Module):
    def __init__(self, num_queries, num_negatives, temp=0.5, mean=False, strong_threshold=0.97):
        super(Prcl_Loss, self).__init__()
        self.temp = temp
        self.mean = mean
        self.num_queries = num_queries
        self.num_negatives = num_negatives
        self.strong_threshold = strong_threshold
    def forward(self, mu, sigma, label, mask, prob, epoch, iter_i):
        
        mu_prt = concat_all_gather(mu) # For protoype computing on all cards (w/o gradients)
        sigma_prt = concat_all_gather(sigma)
        batch_size, num_feat, mu_w, mu_h = mu.shape
        num_segments = label.shape[1] #21
        valid_pixel_all = label * mask
        valid_pixel_all_prt = concat_all_gather(valid_pixel_all) # For protoype computing on all cards 

        # Permute representation for indexing" [batch, rep_h, rep_w, feat_num]
        
        mu = mu.permute(0, 2, 3, 1)
        sigma = sigma.permute(0, 2, 3, 1)
        mu_prt = mu_prt.permute(0, 2, 3, 1)
        sigma_prt = sigma_prt.permute(0, 2, 3, 1)

        mu_all_list = []
        sigma_all_list = []
        mu_hard_list = []
        sigma_hard_list = []
        num_list = []
        proto_mu_list = []
        proto_sigma_list = []

        for i in range(num_segments): #21
            valid_pixel = valid_pixel_all[:, i]
            valid_pixel_gather = valid_pixel_all_prt[:, i]
            if valid_pixel.sum() == 0:
                continue
            prob_seg = prob[:, i, :, :]
            rep_mask_hard = (prob_seg < self.strong_threshold) * valid_pixel.bool() # Only on single card
            # Prototype computing on all cards
            with torch.no_grad():
                proto_sigma_ = 1 / torch.sum((1 / sigma_prt[valid_pixel_gather.bool()]), dim=0, keepdim=True)   
                proto_mu_ = torch.sum((proto_sigma_ / sigma_prt[valid_pixel_gather.bool()]) \
                    * mu_prt[valid_pixel_gather.bool()], dim=0, keepdim=True)
                proto_mu_list.append(proto_mu_)
                proto_sigma_list.append(proto_sigma_)

            mu_all_list.append(mu[valid_pixel.bool()])
            sigma_all_list.append(sigma[valid_pixel.bool()])
            mu_hard_list.append(mu[rep_mask_hard])
            sigma_hard_list.append(sigma[rep_mask_hard])
            num_list.append(int(valid_pixel.sum().item()))
        
        # Compute Probabilistic Representation Contrastive Loss
        if (len(num_list) <= 1) : # in some rare cases, a small mini-batch only contain 1 or no semantic class
            return torch.tensor(0.0) #+ 0 * mu.sum() + 0 * sigma.sum() # A trick for avoiding data leakage in DDP training
        else:
            prcl_loss = torch.tensor(0.0)
            proto_mu = torch.cat(proto_mu_list) # [c]
            proto_sigma = torch.cat(proto_sigma_list)
            valid_num = len(num_list)
            seg_len = torch.arange(valid_num)

            for i in range(valid_num):
                if len(mu_hard_list[i]) > 0:
                    # Random Sampling anchor representations
                    sample_idx = torch.randint(len(mu_hard_list[i]), size=(self.num_queries, ))
                    anchor_mu = mu_hard_list[i][sample_idx]
                    anchor_sigma = sigma_hard_list[i][sample_idx]
                else:
                    continue
                with torch.no_grad():
                    # Select negatives
                    id_mask = torch.cat(([seg_len[i: ], seg_len[: i]]))
                    proto_sim = mutual_likelihood_score(proto_mu[id_mask[0].unsqueeze(0)],
                                                        proto_mu[id_mask[1: ]],
                                                        proto_sigma[id_mask[0].unsqueeze(0)],
                                                        proto_sigma[id_mask[1: ]])
                    proto_prob = torch.softmax(proto_sim / self.temp, dim=0)
                    negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
                    samp_class = negative_dist.sample(sample_shape=[self.num_queries, self.num_negatives])
                    samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)
                    negative_num_list = num_list[i+1: ] + num_list[: i]
                    negative_index = negative_index_sampler(samp_num, negative_num_list)
                    negative_mu_all = torch.cat(mu_all_list[i+1: ] + mu_all_list[: i])
                    negative_sigma_all = torch.cat(sigma_all_list[i+1: ] + sigma_all_list[: i])
                    negative_mu = negative_mu_all[negative_index].reshape(self.num_queries, self.num_negatives, num_feat)
                    negative_sigma = negative_sigma_all[negative_index].reshape(self.num_queries, self.num_negatives, num_feat)
                    positive_mu = proto_mu[i].unsqueeze(0).unsqueeze(0).repeat(self.num_queries, 1, 1)
                    positive_sigma = proto_sigma[i].unsqueeze(0).unsqueeze(0).repeat(self.num_queries, 1, 1)
                    all_mu = torch.cat((positive_mu, negative_mu), dim=1)
                    all_sigma = torch.cat((positive_sigma, negative_sigma), dim=1)
                    # if (iter_i % 10 == 0) and (dist.get_rank() == 0):
                    #     watch_dist(proto_sim, './prcl_right_withexp_watch/watch_dist.txt', epoch)
                    #     watch(positive_sigma, './prcl_right_withexp_watch/watch_proto.txt', i, iter_i, epoch)
                    #     watch(negative_sigma, './prcl_right_withexp_watch/watch_single.txt', i, iter_i, epoch)
                
                logits = mutual_likelihood_score(anchor_mu.unsqueeze(1), all_mu, anchor_sigma.unsqueeze(1), all_sigma)
                prcl_loss = prcl_loss + F.cross_entropy(logits / self.temp, torch.zeros(self.num_queries).long().cuda())
                
            return prcl_loss / valid_num

#### Utils ####

def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(seg_num_list[: j]),
                                                high=sum(seg_num_list[: j+1]),
                                                size=int(samp_num[i, j])).tolist()
    
    return negative_index

def positive_index_sampler(num_queries, num_positives, seg_num):
    
    positive_index = []
    for i in range(num_queries):
        for j in range(num_positives):
            positive_index += [np.random.randint(low=0, high=seg_num)]

    return positive_index


##### Kullback-Leibler divergence #####

class KL_Divergence_2D(nn.Module):

    def __init__(self, reduce=False, eps=1e-10):
        super().__init__()
        self.reduce =reduce
        self.eps = eps

    def forward(self, p_prob: torch.Tensor, y_prob: torch.Tensor) -> torch.Tensor:
        assert simplex(p_prob, 1), '{} must be probability'.format(p_prob)
        assert simplex(y_prob, 1), '{} must be probability'.format(y_prob)

        logp = (p_prob + self.eps).log()
        logy = (y_prob + self.eps).log()
        
        ylogy = (y_prob * logy).sum(dim=1)
        ylogp = (y_prob * logp).sum(dim=1)
        if self.reduce:
            return (ylogy - ylogp).mean()
        else:
            return ylogy - ylogp

class MLS(nn.Module):
    def __init__(self):
        super(MLS, self).__init__()

    def forward(self, mu_0, mu_1, sigma_0, sigma_1, gamma):
        mu_0 = F.normalize(mu_0, dim=-1)
        mu_1 = F.normalize(mu_1, dim=-1)
        up = (mu_0 - mu_1) ** 2
        down = (sigma_0 + sigma_1) ** gamma
        mls = -0.5 * (up / down + torch.log(down)).mean(-1)
        
        return mls

def mutual_likelihood_score(mu_0, mu_1, sigma_0, sigma_1):
    '''
    Compute the MLS
    param: mu_0, mu_1 [256, 513, 256]  [256, 1, 256] 
           sigma_0, sigma_1 [256, 513, 256] [256, 1, 256]
    '''
    mu_0 = F.normalize(mu_0, dim=-1)
    mu_1 = F.normalize(mu_1, dim=-1)
    up = (mu_0 - mu_1) ** 2
    down = sigma_0 + sigma_1
    # print('inside', (-0.5 * (up / down + 0.5 * torch.log(down))))
    mls = -0.5 * (up / down + torch.log(down)).mean(-1)
    

    return mls


def l2_distance(mu_0, mu_1):
    mu_0 = F.normalize(mu_0, dim=-1)
    mu_1 = F.normalize(mu_1, dim=-1)
    l2d  = -((mu_0 - mu_1) ** 2).mean(-1)
    return l2d

def rampscheduler(epoch, begin_epoch, max_epochs, max_val: float, mult: float):
    if epoch < begin_epoch:
        return 0.
    elif epoch >= max_epochs:
        return max_val
    return max_val * np.exp(mult * (1. - float(epoch - begin_epoch) / (max_epochs - begin_epoch)) ** 2)

def watch_proto(file_root: str, proto: torch.Tensor):
    with open(file_root, 'a') as f:
        np.set_printoptions(edgeitems=3, precision=5, linewidth=3600)
        print(proto.squeeze(0).detach().cpu().numpy(), file=f)