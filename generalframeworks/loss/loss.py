# Author: Haoyu Xie
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from generalframeworks.utils import simplex

##### Used in Reco unsupervised learning #####

def attention_threshold_loss(pred: torch.Tensor, pseudo_label: torch.Tensor, logits: torch.Tensor, strong_threshold):
    batch_size = pred.shape[0]
    valid_mask = (pseudo_label >= 0).float() # only count valid pixels (class)
    weighting = logits.view(batch_size, -1).ge(strong_threshold).sum(-1) / valid_mask.view(batch_size, -1).sum(-1)
    # weight represent the proportion of valid pixels in this batch
    loss = F.cross_entropy(pred, pseudo_label, reduction='none', ignore_index=-1) # pixel-wise
    weighted_loss = torch.mean(torch.masked_select(weighting[:, None, None] * loss, loss > 0))
    # weight torch.size([4]) -> weight[:, None, None] torch.size([4, 1, 1]) for broadcast to multiply the weight to the corresponding class
    # torch.masked_select to select loss > 0 only leaved
    
    return weighted_loss

##### Reco Loss #####

def compute_reco_loss(rep: torch.Tensor, label, mask, prob, strong_threshold=1.0, temp=0.5, num_queries=256, num_negatives=256):
    batch_size, num_feat, rep_w, rep_h = rep.shape
    num_segments = label.shape[1] # num_class
    device = rep.device

    # Compute valid binary mask for each pixel
    valid_pixel = label * mask
    
    # Permute representation for indexing: [batch, rep_h, rep_w, feature_channel]
    rep = rep.permute(0, 2, 3, 1)
    # Permute prototype (class mean representation) for each class across all valid pixels
    seg_feat_all_list = []
    seg_feat_hard_list = []
    seg_num_list = []
    seg_proto_list = [] # class mean proto
    for i in range(num_segments): #class
        valid_pixel_seg = valid_pixel[:, i] # select binary mask for i-th class torch.size([b, w, h])
        if valid_pixel_seg.sum() == 0: # not all classes would be available in a mini-batch
            continue
        prob_seg = prob[:, i, :, :] # class prob torch.size([batch, w, h])
        rep_mask_hard = (prob_seg < strong_threshold) * valid_pixel_seg.bool() # select hard queries(prod > threshold & valid mask)
        seg_proto_list.append(torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True)) # compute mean on batch-dim
        seg_feat_all_list.append(rep[valid_pixel_seg.bool()]) # leave valid feature
        seg_feat_hard_list.append(rep[rep_mask_hard]) #leave valid and confident feat
        seg_num_list.append(int(valid_pixel_seg.sum().item())) # valid pixel num on each class-dim

    #Compute regional contrastive loss
    if len(seg_num_list) < 1: # in some rare cases, a small mini-batch only contain 1 or no semantic class
        return torch.tensor(0.0)
    else:
        reco_loss = torch.tensor(0.0)
        seg_proto = torch.cat(seg_proto_list) # torch.size([c, h, w])
        valid_seg = len(seg_num_list)
        seg_len = torch.arange(valid_seg)

        for i in range(valid_seg):
            # Sample hard queries
            if len(seg_feat_hard_list[i]) > 0:
                seg_hard_idx = torch.randint(len(seg_feat_hard_list), size=(num_queries, ))
                # There is bug in original reco project. There is a sutuation, where the the num of hard feature is too small, the following condition may  fail being met. 
                if seg_feat_hard_list[i].shape[0] < len(seg_feat_hard_list):
                    continue
                else:
                    anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
                    anchor_feat = anchor_feat_hard
            else:
                continue

            # Apply negative key sampling (with no gradients)
            with torch.no_grad():
                # Generate index mask for the current query class; e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
                seg_mask = torch.cat(([seg_len[i: ], seg_len[: i]]))
                # Computing similarity for each negative segment prototype (semantic class relation graph)
                proto_sim = torch.cosine_similarity(seg_proto[seg_mask[0]].unsqueeze(0), seg_proto[seg_mask[1: ]], dim=1)
                proto_prob = torch.softmax(proto_sim / temp, dim=0)
                # Samplin negative keys based on the generated distribution [num_queries * num_negatives]
                negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
                samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
                samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)
                # Sampling negatie indices from each negative calss
                negative_num_list = seg_num_list[i+1: ] + seg_num_list[: i]
                negative_index = negative_index_sampler(samp_num, negative_num_list)
                # index negative keys (from other classes)
                negative_feat_all = torch.cat(seg_feat_all_list[i+1: ] + seg_feat_all_list[: i])
                negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)
                # Combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
                positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)
            
            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
            reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().to(device))
        
        return reco_loss /valid_seg

def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(seg_num_list[: j]),
                                                high=sum(seg_num_list[: j+1]),
                                                size=int(samp_num[i, j])).tolist()
    
    return negative_index

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




