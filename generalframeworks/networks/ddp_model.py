import copy
import torch
import torch.nn as nn
from generalframeworks.networks.deeplabv3.deeplabv3 import DeepLabv3Plus_nr, DeepLabv3Plus_with_un, DeepLabv3Plus, DeepLabv3Plus_light, DeepLabv3Plus_multi_rep
from generalframeworks.networks.ema import EMA, OOD_EMA
import torch.nn.functional as F
from generalframeworks.dataset_helpers.VOC import batch_transform, generate_cut_gather, generate_cut
import torch.distributed as dist
from generalframeworks.networks.uncer_head import Uncertainty_head, Uncertainty_head_conv
from generalframeworks.networks.predictor import Predictor
from torch.cuda.amp.autocast_mode import autocast


class Model(nn.Module):
    '''
    Build a model for DDP with: a DeepLabV3_Plus, and a ema
    '''

    def __init__(self, base_encoder, num_classes=21, output_dim=256, ema_alpha=0.99, config=None) -> None:
        super(Model, self).__init__()

        self.model = DeepLabv3Plus(base_encoder, num_classes=num_classes, output_dim=output_dim).cuda(dist.get_rank())
        self.step = 0
        self.ema_model = copy.deepcopy(self.model)
        self.alpha = ema_alpha
        print('EMA model has been prepared. Alpha = {}'.format(self.alpha))
        self.config = config

    def ema_update(self):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1

    def forward(self, train_l_image, train_u_image):
        ##### generate pseudo label #####
        with torch.no_grad():
            pred_u, _ = self.ema_model(train_u_image)
            pred_u_large_raw = F.interpolate(pred_u, size=train_u_image.shape[2:], mode='bilinear', align_corners=True)
            pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)

            # Randomly scale images
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_image, pseudo_labels,
                                                                                       pseudo_logits,
                                                                                       crop_size=[321, 321],
                                                                                       scale_size=(0.5, 1.5),
                                                                                       augmentation=False)
            # Apply mixing strategy
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = generate_cut_gather(train_u_aug_image,
                                                                                    train_u_aug_label,
                                                                                    train_u_aug_logits,
                                                                                    mode=self.config['Dataset'][
                                                                                        'mix_mode'])
            # Apply augmnetation : color jitter + flip + gaussian blur
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_aug_image,
                                                                                       train_u_aug_label,
                                                                                       train_u_aug_logits,
                                                                                       crop_size=[321, 321],
                                                                                       scale_size=(1.0, 1.0),
                                                                                       augmentation=True)

        pred_l, rep_l = self.model(train_l_image)
        pred_l_large = F.interpolate(pred_l, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)

        pred_u, rep_u = self.model(train_u_aug_image)
        pred_u_large = F.interpolate(pred_u, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)


        rep_all = torch.cat((rep_l, rep_u))
        pred_all = torch.cat((pred_l, pred_u))

        return pred_l_large, pred_u_large, train_u_aug_label, train_u_aug_logits, rep_all, pred_all, pred_u_large_raw

class Model_multi_rep(nn.Module):
    '''
    Build a model for DDP with: a DeepLabV3_Plus, and a ema
    '''

    def __init__(self, base_encoder, num_classes=21, output_dim=256, ema_alpha=0.99, config=None) -> None:
        super(Model_multi_rep, self).__init__()

        self.model = DeepLabv3Plus_multi_rep(base_encoder, num_classes=num_classes, output_dim=output_dim).cuda(dist.get_rank())
        self.step = 0
        self.ema_model = copy.deepcopy(self.model)
        self.alpha = ema_alpha
        print('EMA model has been prepared. Alpha = {}'.format(self.alpha))
        self.config = config

    def ema_update(self):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1

    def forward(self, train_l_image, train_u_image):
        ##### generate pseudo label #####
        with torch.no_grad():
            pred_u, _, _ = self.ema_model(train_u_image)
            pred_u_large_raw = F.interpolate(pred_u, size=train_u_image.shape[2:], mode='bilinear', align_corners=True)
            pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)

            # Randomly scale images
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_image, pseudo_labels,
                                                                                       pseudo_logits,
                                                                                       crop_size=[321, 321],
                                                                                       scale_size=(0.5, 1.5),
                                                                                       augmentation=False)
            # Apply mixing strategy
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = generate_cut_gather(train_u_aug_image,
                                                                                    train_u_aug_label,
                                                                                    train_u_aug_logits,
                                                                                    mode=self.config['Dataset'][
                                                                                        'mix_mode'])
            # Apply augmnetation : color jitter + flip + gaussian blur
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_aug_image,
                                                                                       train_u_aug_label,
                                                                                       train_u_aug_logits,
                                                                                       crop_size=[321, 321],
                                                                                       scale_size=(1.0, 1.0),
                                                                                       augmentation=True)

        pred_l, rep_l, rep_l_1 = self.model(train_l_image)
        pred_l_large = F.interpolate(pred_l, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)

        pred_u, rep_u, rep_u_1 = self.model(train_u_aug_image)
        pred_u_large = F.interpolate(pred_u, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)


        rep_all = torch.cat((rep_l, rep_u))
        rep_all_1 = torch.cat((rep_l_1, rep_u_1))
        pred_all = torch.cat((pred_l, pred_u))

        return pred_l_large, pred_u_large, train_u_aug_label, train_u_aug_logits, rep_all, rep_all_1, pred_all, pred_u_large_raw




class Model_with_ema(nn.Module):
    '''
    Build a model for DDP with: a DeepLabV3_Plus without representation, and a ema
    '''

    def __init__(self, base_encoder, num_classes=21, output_dim=256, ema_alpha=0.99, config=None) -> None:
        super(Model_with_ema, self).__init__()

        self.model = DeepLabv3Plus_nr(base_encoder, num_classes=num_classes, output_dim=output_dim).cuda(dist.get_rank())
        self.step = 0
        self.ema_model = copy.deepcopy(self.model)
        self.alpha = ema_alpha
        print('EMA model has been prepared. Alpha = {}'.format(self.alpha))
        self.config = config

    def ema_update(self):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1

    def forward(self, train_l_image, train_u_image):
        ##### generate pseudo label #####
        with torch.no_grad():
            pred_u = self.ema_model(train_u_image)
            pred_u_large_raw = F.interpolate(pred_u, size=train_u_image.shape[2:], mode='bilinear', align_corners=True)
            pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)

            # Randomly scale images
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_image, pseudo_labels,
                                                                                       pseudo_logits,
                                                                                       crop_size=self.config['Dataset']['crop_size'],
                                                                                       scale_size=self.config['Dataset']['scale_size'],
                                                                                       augmentation=False)
            # Apply mixing strategy
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = generate_cut_gather(train_u_aug_image,
                                                                                    train_u_aug_label,
                                                                                    train_u_aug_logits,
                                                                                    mode=self.config['Dataset'][
                                                                                        'mix_mode'])
            # Apply augmnetation : color jitter + flip + gaussian blur
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_aug_image,
                                                                                       train_u_aug_label,
                                                                                       train_u_aug_logits,
                                                                                       crop_size=self.config['Dataset']['crop_size'],
                                                                                       scale_size=(1.0, 1.0),
                                                                                       augmentation=True)

        pred_l = self.model(train_l_image)
        pred_l_large = F.interpolate(pred_l, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)

        pred_u = self.model(train_u_aug_image)
        pred_u_large = F.interpolate(pred_u, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)

        pred_all = torch.cat((pred_l, pred_u))

        return pred_l_large, pred_u_large, train_u_aug_label, train_u_aug_logits, pred_all, pred_u_large_raw

class Model_with_un(nn.Module):
    '''
    Build a model for DDP with: a DeepLabV3_Plus, a ema, and a mlp
    '''

    def __init__(self, base_encoder, num_classes=21, output_dim=256, ema_alpha=0.99, config=None) -> None:
        super(Model_with_un, self).__init__()
        self.model = DeepLabv3Plus_with_un(base_encoder, num_classes=num_classes, output_dim=output_dim)
        ##### Init EMA #####
        self.step = 0
        self.ema_model = copy.deepcopy(self.model)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.alpha = ema_alpha
        print('EMA model has been prepared. Alpha = {}'.format(self.alpha))

        ##### Init Uncertainty Head #####
        self.uncer_head = Uncertainty_head()

        self.config = config

    def ema_update(self):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1

    def forward(self, train_l_image, train_u_image):
        ##### generate pseudo label #####
        with torch.no_grad():
            pred_u, _, _ = self.ema_model(train_u_image)
            pred_u_large_raw = F.interpolate(pred_u, size=train_u_image.shape[2:], mode='bilinear', align_corners=True)
            pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)

            # Randomly scale images
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_image, pseudo_labels,
                                                                                       pseudo_logits,
                                                                                       crop_size=self.config['Dataset']['crop_size'],
                                                                                       scale_size=self.config['Dataset']['scale_size'],
                                                                                       augmentation=False)
            # Apply mixing strategy
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = generate_cut_gather(train_u_aug_image,
                                                                                    train_u_aug_label,
                                                                                    train_u_aug_logits,
                                                                                    mode=self.config['Dataset'][
                                                                                        'mix_mode'])
            # Apply augmnetation : color jitter + flip + gaussian blur
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_aug_image,
                                                                                       train_u_aug_label,
                                                                                       train_u_aug_logits,
                                                                                       crop_size=self.config['Dataset']['crop_size'],
                                                                                       scale_size=(1.0, 1.0),
                                                                                       augmentation=True)


        pred_l, rep_l, raw_feat_l = self.model(train_l_image)
        pred_l_large = F.interpolate(pred_l, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)

        pred_u, rep_u, raw_feat_u = self.model(train_u_aug_image)
        pred_u_large = F.interpolate(pred_u, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)

        rep_all = torch.cat((rep_l, rep_u))
        pred_all = torch.cat((pred_l, pred_u))

        log_uncer_all = self.uncer_head(torch.cat((raw_feat_l, raw_feat_u), dim=0))
        # uncer_all = torch.exp(log_uncer_all)
        uncer_all = log_uncer_all

        return pred_l_large, pred_u_large, train_u_aug_label, train_u_aug_logits, rep_all, pred_all, pred_u_large_raw, uncer_all

class Model_with_un_conv(nn.Module):
    '''
    Build a model for DDP with: a DeepLabV3_Plus, a ema, and a mlp
    '''

    def __init__(self, base_encoder, num_classes=21, output_dim=256, ema_alpha=0.99, config=None) -> None:
        super(Model_with_un_conv, self).__init__()
        self.model = DeepLabv3Plus_with_un(base_encoder, num_classes=num_classes, output_dim=output_dim)
        ##### Init EMA #####
        self.step = 0
        self.ema_model = copy.deepcopy(self.model)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.alpha = ema_alpha
        print('EMA model has been prepared. Alpha = {}'.format(self.alpha))

        ##### Init Uncertainty Head #####
        self.uncer_head = Uncertainty_head_conv()

        self.config = config

    def ema_update(self):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1

    def forward(self, train_l_image, train_u_image):
        ##### generate pseudo label #####
        with torch.no_grad():
            pred_u, _, _ = self.ema_model(train_u_image)
            pred_u_large_raw = F.interpolate(pred_u, size=train_u_image.shape[2:], mode='bilinear', align_corners=True)
            pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)

            # Randomly scale images
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_image, pseudo_labels,
                                                                                       pseudo_logits,
                                                                                       crop_size=self.config['Dataset']['crop_size'],
                                                                                       scale_size=self.config['Dataset']['scale_size'],
                                                                                       augmentation=False)
            # Apply mixing strategy
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = generate_cut_gather(train_u_aug_image,
                                                                                    train_u_aug_label,
                                                                                    train_u_aug_logits,
                                                                                    mode=self.config['Dataset'][
                                                                                        'mix_mode'])
            # Apply augmnetation : color jitter + flip + gaussian blur
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_aug_image,
                                                                                       train_u_aug_label,
                                                                                       train_u_aug_logits,
                                                                                       crop_size=self.config['Dataset']['crop_size'],
                                                                                       scale_size=(1.0, 1.0),
                                                                                       augmentation=True)


        pred_l, rep_l, raw_feat_l = self.model(train_l_image)
        pred_l_large = F.interpolate(pred_l, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)

        pred_u, rep_u, raw_feat_u = self.model(train_u_aug_image)
        pred_u_large = F.interpolate(pred_u, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)

        rep_all = torch.cat((rep_l, rep_u))
        pred_all = torch.cat((pred_l, pred_u))

        log_uncer_all = self.uncer_head(torch.cat((raw_feat_l, raw_feat_u), dim=0))
        # uncer_all = torch.exp(log_uncer_all)
        uncer_all = log_uncer_all

        return pred_l_large, pred_u_large, train_u_aug_label, train_u_aug_logits, rep_all, pred_all, pred_u_large_raw, uncer_all

class Model_light(nn.Module):
    '''
    Build a model for DDP with: a DeepLabV3_Plus, a ema, and a mlp
    '''
    def __init__(self, base_encoder, num_classes=21, output_dim=256, ema_alpha=0.99, config=None) -> None:
        super(Model_light, self).__init__()
        self.model = DeepLabv3Plus_light(base_encoder, output_dim=output_dim, apex_sync_bn=False)
        self.predictor = Predictor(out_feat=num_classes)
        ##### Init EMA #####
        self.step = 0
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model_pred = copy.deepcopy(self.predictor)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        for p in self.ema_model_pred.parameters():
            p.requires_grad = False
        self.alpha = ema_alpha
        print('EMA model has been prepared. Alpha = {}'.format(self.alpha))

        ##### Init Uncertainty Head #####
        self.uncer_head = Uncertainty_head(apex_sync_bn=False)

        self.config = config

    @torch.no_grad()
    def update(self):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        for ema_param, param in zip(self.ema_model_pred.parameters(), self.predictor.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1
        
    def forward(self, train_l_image, train_u_image):
        ##### generate pseudo label #####
        ##### batch shuffle #####
        # train_u_image, idx_unshuffle = self._batch_shuffle_ddp(train_u_image)
        with torch.no_grad():
            _, raw_feat = self.ema_model(train_u_image)
            pred_u = self.ema_model_pred(raw_feat)
            pred_u_large_raw = F.interpolate(pred_u, size=train_u_image.shape[2:], mode='bilinear', align_corners=True)
            pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)

            # Randomly scale images
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_image, pseudo_labels,
                                                                                       pseudo_logits,
                                                                                       crop_size=self.config['Dataset']['crop_size'],
                                                                                       scale_size=(0.5, 1.5),
                                                                                       augmentation=False)
            # Apply mixing strategy
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = generate_cut(train_u_aug_image,
                                                                                    train_u_aug_label,
                                                                                    train_u_aug_logits,
                                                                                    mode=self.config['Dataset'][
                                                                                        'mix_mode'])
            # Apply augmnetation : color jitter + flip + gaussian blur
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_aug_image,
                                                                                       train_u_aug_label,
                                                                                       train_u_aug_logits,
                                                                                       crop_size=self.config['Dataset']['crop_size'],
                                                                                       scale_size=(1.0, 1.0),
                                                                                       augmentation=True)


        rep_l, raw_feat_l = self.model(train_l_image)
        pred_l = self.predictor(raw_feat_l)
        pred_l_large = F.interpolate(pred_l, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)

        rep_u, raw_feat_u = self.model(train_u_aug_image)
        pred_u = self.predictor(raw_feat_u)
        pred_u_large = F.interpolate(pred_u, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)

        rep_all = torch.cat((rep_l, rep_u))
        pred_all = torch.cat((pred_l, pred_u))

        uncer_all = self.uncer_head(torch.cat((raw_feat_l, raw_feat_u), dim=0))

        return pred_l_large, pred_u_large, train_u_aug_label, train_u_aug_logits, rep_all, pred_all, pred_u_large_raw, uncer_all

class Model_with_un_gather(nn.Module):
    '''
    Build a model for DDP with: a DeepLabV3_Plus, a ema, and a mlp
    '''

    def __init__(self, base_encoder, num_classes=21, output_dim=256, ema_alpha=0.99, config=None) -> None:
        super(Model_with_un_gather, self).__init__()
        self.model = DeepLabv3Plus_with_un(base_encoder, num_classes=num_classes, output_dim=output_dim)
        ##### Init EMA #####
        self.step = 0
        self.ema_model = copy.deepcopy(self.model)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.alpha = ema_alpha
        print('EMA model has been prepared. Alpha = {}'.format(self.alpha))

        ##### Init Uncertainty Head #####
        self.uncer_head = Uncertainty_head()

        self.config = config

    def ema_update(self):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1

    def forward(self, train_l_image, train_u_image):
        ##### generate pseudo label #####
        with torch.no_grad():
            pred_u, _, _ = self.ema_model(train_u_image)
            pred_u_large_raw = F.interpolate(pred_u, size=train_u_image.shape[2:], mode='bilinear', align_corners=True)
            pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)

            # Randomly scale images
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_image, pseudo_labels,
                                                                                       pseudo_logits,
                                                                                       crop_size=self.config['Dataset']['crop_size'],
                                                                                       scale_size=self.config['Dataset']['scale_size'],
                                                                                       augmentation=False)
            # Apply mixing strategy
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = generate_cut_gather(train_u_aug_image,
                                                                                    train_u_aug_label,
                                                                                    train_u_aug_logits,
                                                                                    mode=self.config['Dataset'][
                                                                                        'mix_mode'])
            # Apply augmnetation : color jitter + flip + gaussian blur
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_aug_image,
                                                                                       train_u_aug_label,
                                                                                       train_u_aug_logits,
                                                                                       crop_size=self.config['Dataset']['crop_size'],
                                                                                       scale_size=(1.0, 1.0),
                                                                                       augmentation=True)


        pred_l, rep_l, raw_feat_l = self.model(train_l_image)
        pred_l_large = F.interpolate(pred_l, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)

        pred_u, rep_u, raw_feat_u = self.model(train_u_aug_image)
        pred_u_large = F.interpolate(pred_u, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)

        rep_all = torch.cat((rep_l, rep_u))
        pred_all = torch.cat((pred_l, pred_u))

        log_uncer_all = self.uncer_head(torch.cat((raw_feat_l, raw_feat_u), dim=0))
        uncer_all = torch.exp(log_uncer_all)

        return pred_l_large, pred_u_large, train_u_aug_label, train_u_aug_logits, rep_all, pred_all, pred_u_large_raw, uncer_all
    
class Model_fp16(nn.Module):
    '''
    Build a model for DDP with: a DeepLabV3_Plus, and a ema
    '''

    def __init__(self, base_encoder, num_classes=21, output_dim=256, ema_alpha=0.99, config=None) -> None:
        super(Model_fp16, self).__init__()
        self.model = DeepLabv3Plus_with_un(base_encoder, num_classes=num_classes, output_dim=output_dim)
        ##### Init EMA #####
        self.step = 0
        self.ema_model = copy.deepcopy(self.model)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.alpha = ema_alpha
        print('EMA model has been prepared. Alpha = {}'.format(self.alpha))

        ##### Init Uncertainty Head #####
        self.uncer_head = Uncertainty_head()

        self.config = config

    def ema_update(self):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1
    @autocast()
    def forward(self, train_l_image, train_u_image):
        ##### generate pseudo label #####
        with torch.no_grad():
            pred_u, _, _ = self.ema_model(train_u_image)
            pred_u_large_raw = F.interpolate(pred_u, size=train_u_image.shape[2:], mode='bilinear', align_corners=True)
            pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)

            # Randomly scale images
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_image, pseudo_labels,
                                                                                       pseudo_logits,
                                                                                       crop_size=self.config['Dataset']['crop_size'],
                                                                                       scale_size=self.config['Dataset']['scale_size'],
                                                                                       augmentation=False)
            # Apply mixing strategy
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = generate_cut(train_u_aug_image,
                                                                                    train_u_aug_label,
                                                                                    train_u_aug_logits,
                                                                                    mode=self.config['Dataset'][
                                                                                        'mix_mode'])
            # Apply augmnetation : color jitter + flip + gaussian blur
            train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_aug_image,
                                                                                       train_u_aug_label,
                                                                                       train_u_aug_logits,
                                                                                       crop_size=self.config['Dataset']['crop_size'],
                                                                                       scale_size=(1.0, 1.0),
                                                                                       augmentation=True)


        pred_l, rep_l, raw_feat_l = self.model(train_l_image)
        pred_l_large = F.interpolate(pred_l, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)

        pred_u, rep_u, raw_feat_u = self.model(train_u_aug_image)
        pred_u_large = F.interpolate(pred_u, size=train_l_image.shape[2:], mode='bilinear', align_corners=True)

        rep_all = torch.cat((rep_l, rep_u))
        pred_all = torch.cat((pred_l, pred_u))

        log_uncer_all = self.uncer_head(torch.cat((raw_feat_l, raw_feat_u), dim=0))
        # uncer_all = torch.exp(log_uncer_all)
        uncer_all = log_uncer_all

        return pred_l_large, pred_u_large, train_u_aug_label, train_u_aug_logits, rep_all, pred_all, pred_u_large_raw, uncer_all



    # utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    Warning: torch.distributed.all_ather has no gradient.
    """
    tensor_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensor_gather, tensor, async_op=False)
    output = torch.cat(tensor_gather, dim=0)

    return output