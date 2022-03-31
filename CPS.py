'''
Thansk for Chen et al. and their paper: https://arxiv.org/abs/2106.01226
Author: Changqi Wang
'''
import os
from pathlib import Path
from imageio import save
import torch.nn.functional as F
from torch import nn
import torch
from random import Random
import warnings
from typing import List
import yaml
from generalframeworks.utils import dict_merge, fix_all_seed, yaml_parser, now_time, class2one_hot, label_onehot, tqdm_
from generalframeworks.dataset_helpers.dataset import BaseDataSet
from torch.utils.data import DataLoader, sampler
from torchvision import transforms
from pprint import pprint
from generalframeworks.dataset_helpers.make_list import make_ACDC_list
from generalframeworks.dataset_helpers.ACDC_helper import ACDC_Dataset
from generalframeworks.networks import network_factory
import generalframeworks.scheduler.my_lr_scheduler as my_lr_scheduler
from generalframeworks.networks.ema import EMA
import numpy as np
from generalframeworks.meter.meter import AverageValueMeter, Meter
from generalframeworks.augmentation.transform import batch_transform, generate_cut
from generalframeworks.loss.loss import attention_threshold_loss, compute_reco_loss
from generalframeworks.utils import iterator_, RampScheduler

from tensorboardX import SummaryWriter


##### Config Preparation #####
warnings.filterwarnings('ignore')
parser_args = yaml_parser()

#pprint('--> Input args:')
with open('./config/CPS_config.yaml', 'r') as f:
    config = yaml.load(f.read())
config = dict_merge(config, parser_args, True)
#pprint(config)
print('Hello, it is {} now, let\'s train cps!'.format(now_time()))
save_dir = config['Training_Setting']['save_dir'] + '/' + config['Dataset']['root_dir'].split('/')[-1] + '/'+ \
           str(config['Labeled_Dataset']['num_patient']) + '/' + str(config['Seed'])
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(save_dir + '/config.yaml', 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)

##### Tensorboard #####
writer = SummaryWriter(save_dir)

##### Init Random Seed #####
fix_all_seed(int(config['Seed']))                   

if __name__ == '__main__':
    ##### Dataset Preparation #####
    make_ACDC_list(dataset_dir=config['Dataset']['root_dir'], save_dir=save_dir, labeled_num=config['Labeled_Dataset']['num_patient']) # Make there dataset lists
    train_l_dataset = ACDC_Dataset(root_dir=config['Dataset']['root_dir'], save_dir=save_dir, mode='label', meta_label=False)
    train_u_dataset = ACDC_Dataset(root_dir=config['Dataset']['root_dir'], save_dir=save_dir, mode='unlabel', meta_label=False)  
    val_dataset = ACDC_Dataset(root_dir=config['Dataset']['root_dir'], save_dir=save_dir, mode='val', meta_label=False) 
    train_l_loader = DataLoader(train_l_dataset,
                                batch_size=config['Labeled_Dataset']['batch_size'], 
                                sampler=sampler.RandomSampler(data_source=train_l_dataset),
                                drop_last=True)
    train_u_loader = DataLoader(train_u_dataset,
                                batch_size=config['Unlabeled_Dataset']['batch_size'],
                                sampler=sampler.RandomSampler(data_source=train_u_dataset),
                                drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    ##### Device #####
    device = torch.device(config['Training_Setting']['device'] if torch.cuda.is_available() else "cpu")
    print('Training on ' + config['Training_Setting']['device'] if torch.cuda.is_available() else "cpu")

    ##### Model Initization #####
    model_l = network_factory(name=config['Network']['name'], num_class=config['Network']['num_class']).to(device)
    model_r = network_factory(name=config['Network']['name'], num_class=config['Network']['num_class']).to(device)
    optimizer_l = torch.optim.SGD(model_l.parameters(), lr=float(config['Optim']['lr']), weight_decay=float(config['Optim']['weight_decay']),
                                                        momentum=0.9, nesterov=True)
    optimizer_r = torch.optim.SGD(model_r.parameters(), lr=float(config['Optim']['lr']), weight_decay=float(config['Optim']['weight_decay']),
                                                        momentum=0.9, nesterov=True)
    #scheduler: torch.optim.lr_scheduler = getattr(my_lr_scheduler, config['Lr_Scheduler']['name']) if config['Lr_Scheduler']['name'] in ['PolyLR'] \
                                        #else getattr(my_lr_scheduler, config['Lr_Scheduler']['name'])(optimizer, **config['Lr_Scheduler'])
    scheduler_l = my_lr_scheduler.PolyLR(optimizer_l, config['Training_Setting']['epoch'], power=config['Lr_Scheduler']['lr_power'])
    scheduler_r = my_lr_scheduler.PolyLR(optimizer_r, config['Training_Setting']['epoch'], power=config['Lr_Scheduler']['lr_power'])
    ##### Metrics Initization #####
    max_epoch = config['Training_Setting']['epoch']
    iter_max = len(train_u_loader)
    # iter_max = len(train_u_loader)
    lab_dice = AverageValueMeter(num_class=config['Network']['num_class'])
    unlab_dice = AverageValueMeter(num_class=config['Network']['num_class'])
    val_dice_l = AverageValueMeter(num_class=config['Network']['num_class'])
    val_dice_r = AverageValueMeter(num_class=config['Network']['num_class'])
    '''metrics = {'train_dice': np.zeros([max_epoch, config['Network']['num_class']]),
                'train_unlab_dice': np.zeros([max_epoch, config['Network']['num_class']]),
                'val_dice': np.zeros([max_epoch, config['Network']['num_class']]),
                'val_batch_dice': np.zeros([max_epoch, config['Network']['num_class']])}'''
    avg_cost = np.zeros((max_epoch, 13))
    loss = {'sup_loss': np.zeros(max_epoch),
                'unsup_loss': np.zeros(max_epoch),
                'cps_loss': np.zeros(max_epoch)}
    
    # criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    # scheduler
    cps_scheduler = RampScheduler(config['Cps_Scheduler']['begin_epoch'], config['Cps_Scheduler']['max_epoch'], config['Cps_Scheduler']['max_value'], \
                                             config['Cps_Scheduler']['ramp_mult'])
    ##### Training #####
    for epoch in range(max_epoch):
        cost = np.zeros(4)
        dice_lab = []
        dice_unlab = []
        train_lab_iter = iterator_(train_l_loader)
        train_unlab_iter = iterator_(train_u_loader)
        # train_lab_iter = iterator_(train_l_loader)
        # train_unlab_iter = iterator_(train_u_loader)
        
        model_l.train()
        model_r.train()
        lab_dice.reset()
        unlab_dice.reset()
        val_dice_l.reset()
        val_dice_r.reset()
        train_iter_tqdm = tqdm_(range(iter_max))
        for i in train_iter_tqdm:
            current_idx = epoch * iter_max + i
            lab_image, lab_label = train_lab_iter.__next__()
            lab_image, lab_label = lab_image.to(device), lab_label.to(device) #torch.Size([4, 1, 256, 256]) torch.Size([4, 256, 256])

            unlab_image, unlab_label = train_unlab_iter.__next__()
            unlab_image, unlab_label = unlab_image.to(device), unlab_label.to(device) #torch.Size([4, 1, 256, 256])

            #unlab_label_oh = class2one_hot(unlab_label, num_class=config['Network']['num_class'])#torch.Size([4, 4, 256, 256])
       
            optimizer_l.zero_grad()
            optimizer_r.zero_grad()

            # Generate pseudo_labels for unlabeled data
            pred_u_l, _ = model_l(unlab_image) #torch.Size([4, 4, 64, 64]) torch.Size([4, 256, 64, 64])
            pred_u_r, _ = model_r(unlab_image) #torch.Size([4, 4, 64, 64]) torch.Size([4, 256, 64, 64])
            pred_u_l_large_raw = F.interpolate(pred_u_l, size=unlab_label.shape[1:], mode='bilinear', align_corners=True)#torch.Size([4, 4, 256, 256])
            pred_u_r_large_raw = F.interpolate(pred_u_r, size=unlab_label.shape[1:], mode='bilinear', align_corners=True)#torch.Size([4, 4, 256, 256])
            # Generate pseudo logits and labels. Attention: l is for r!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
            pseudo_logits_unlab_l, pseudo_labels_unlab_l = torch.max(pred_u_l_large_raw, dim=1) #torch.Size([4, 256, 256]) torch.Size([4, 256, 256])
            pseudo_logits_unlab_r, pseudo_labels_unlab_r = torch.max(pred_u_r_large_raw, dim=1) #torch.Size([4, 256, 256]) torch.Size([4, 256, 256])
            pseudo_labels_unlab_l.long()
            pseudo_labels_unlab_r.long()

            # Generate pseudo_labels for labeled data
            pred_lab_l, _ = model_l(lab_image) #torch.Size([4, 4, 64, 64]) torch.Size([4, 256, 64, 64])
            pred_lab_r, _ = model_r(lab_image) #torch.Size([4, 4, 64, 64]) torch.Size([4, 256, 64, 64])
            pred_lab_l_large_raw = F.interpolate(pred_lab_l, size=lab_label.shape[1:], mode='bilinear', align_corners=True)#torch.Size([4, 4, 256, 256])
            pred_lab_r_large_raw = F.interpolate(pred_lab_r, size=lab_label.shape[1:], mode='bilinear', align_corners=True)#torch.Size([4, 4, 256, 256])

            # Generate pseudo logits and labels. Attention: l is for r!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
            pseudo_logits_lab_l, pseudo_labels_lab_l = torch.max(pred_lab_l_large_raw, dim=1) #torch.Size([4, 256, 256]) torch.Size([4, 256, 256])
            pseudo_logits_lab_r, pseudo_labels_lab_r = torch.max(pred_lab_r_large_raw, dim=1) #torch.Size([4, 256, 256]) torch.Size([4, 256, 256])
            pseudo_labels_lab_l.long()
            pseudo_labels_lab_r.long()

            # Random scale and  and crop, We ommit the imagenet normalization.!!!!! and we change the crop_size here instead of (512, 512)
            #???????????????????????????????????????????????
            # image_u_aug, label_u_l_aug, logits_u_l_aug = batch_transform(unlab_image, pseudo_labels_unlab_l, pseudo_logits_unlab_l,\
            #     crop_size=(256, 256), scale_size=(0.5, 1.5), apply_augmentation=False)
            # _, label_u_r_aug, logits_u_r_aug = batch_transform(unlab_image, pseudo_labels_unlab_r, pseudo_logits_unlab_r,\
            #     crop_size=(256, 256), scale_size=(0.5, 1.5), apply_augmentation=False)
            
            # Apply mixing strategy: cutout, cutmix or classmix
            # image_u_aug, label_u_l_aug, logits_u_l_aug = generate_cut(image_u_aug, label_u_l_aug, logits_u_l_aug,\
            #     mode=config['Unlabeled_Dataset']['aug_mode'])
            # _, label_u_r_aug, logits_u_r_aug = generate_cut(image_u_aug, label_u_r_aug, logits_u_r_aug,\
            #     mode=config['Unlabeled_Dataset']['aug_mode'])
            #torch.Size([4, 1, 256, 256]) torch.Size([4, 256, 256]) torch.Size([4, 4, 256, 256])
            
            # Generate labeled and unlabeled data loss, why the pred here is torch.Size([4, 4, 64, 64])???

            ##### Supervised learning Loss #####
            sup_loss = criterion(pred_lab_l_large_raw, lab_label) + criterion(pred_lab_r_large_raw, lab_label) # label must be class (index)

            ##### Cps Loss #####
            if config['Cps_Loss']['is_available']:
                cps_loss_lab = criterion(pred_lab_l_large_raw, pseudo_labels_lab_r) + criterion(pred_lab_r_large_raw, pseudo_labels_lab_l)
                cps_loss_unlab = criterion(pred_u_l_large_raw, pseudo_labels_unlab_r) + criterion(pred_u_r_large_raw, pseudo_labels_unlab_l)
                cps_loss = cps_loss_lab + cps_loss_unlab
                cps_loss = cps_loss * cps_scheduler.value
            else:
                cps_loss = torch.tensor(0.0)
            
            loss = sup_loss + cps_loss
            loss.backward()
            optimizer_l.step()
            optimizer_r.step()

            lab_dice.add(pred_lab_l_large_raw, lab_label)
            unlab_dice.add(pred_u_l_large_raw, unlab_label)
            cost[0] = sup_loss.item()
            cost[1] = cps_loss_lab.item()
            cost[2] = cps_loss_unlab.item()
            cost[3] = cps_loss.item()
            avg_cost[epoch, :4] += cost / iter_max
            # Progress Bar
            train_iter_tqdm.set_description('Training' + ':' + 'Sup:{:.3f}|cps_loss_lab:{:.3f}|cps_loss_unlab:{:.3f}|cps_loss_total:{:.3f}'.format(cost[0], cost[1], cost[2], cost[3]))
            train_iter_tqdm.set_postfix({'Lab_DSC': '{:.3f}'.format(lab_dice.value(i)), 'Unlab_DSC': '{:.3f}'.format(unlab_dice.value(i))})
        # Log dice in one epoch
        lab_dct = lab_dice.summary()
        for i, key in enumerate(lab_dct):
            avg_cost[epoch, 4 + i] = lab_dct[key] #[4, 5, 6]
        avg_cost[epoch, 7] = avg_cost[epoch, 4: 7].mean()
        
        ##### Evaluation #####
        with torch.no_grad():
            model_l.eval()
            model_r.eval()
            val_iter = iter(val_loader)
            val_dice_l.reset()
            max_iter = len(val_loader)
            iter_val_tqdm = tqdm_(range(max_iter))
            for i in iter_val_tqdm:
                val_image, val_label = val_iter.next()
                val_image, val_label = val_image.to(device), val_label.to(device)

                pred_l, _ = model_l(val_image)
                pred_l = F.interpolate(pred_l, size=val_label.shape[1:], mode='bilinear', align_corners=True)
                loss_l = F.cross_entropy(pred_l, val_label)
                # Log
                val_dice_l.add(pred_l, val_label)
                avg_cost[epoch, 8] += loss_l.item() / max_iter
                dice_tmp_l = [val_dice_l.value(i, mode='all')[c] for c in range(1, 4)]
                dice_tmp_l.append(np.array(dice_tmp_l).mean())
                iter_val_tqdm.set_description('Validing' + ':' + 'DSC_1:{:.3f}|DSC_2:{:.3f}|DSC_3:{:.3f}|DSC_AVG:{:.3f}'.format(*dice_tmp_l))
            val_dct = val_dice_l.summary()
            for i, key in enumerate(val_dct):
                avg_cost[epoch, 9 + i] = val_dct[key]
            avg_cost[epoch, 12] = avg_cost[epoch, 9: 12].mean()
        scheduler_l.step()
        scheduler_r.step()
        cps_scheduler.step()
        print('\n  EPOCH | TRAIN  |SUP_LOSS|LAB_CPS_LOSS|UNLAB_CPS_LOSS|CPS_LOSS| DISC_1 | DISC_2 | DISC_3 |DSIC_AVG| \n   {:03d}  |        | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n        |  Test  |  Loss  | DISC_1 | DISC_2 | DISC_3 |DISC_AVG|TOP_DISC|\n        |        | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'\
            .format(epoch, avg_cost[epoch][0], avg_cost[epoch][1], avg_cost[epoch][2], avg_cost[epoch][3], avg_cost[epoch][4], avg_cost[epoch][5], avg_cost[epoch][6], avg_cost[epoch][7], avg_cost[epoch][8],\
                 avg_cost[epoch][9], avg_cost[epoch][10], avg_cost[epoch][11], avg_cost[epoch][12],avg_cost[:, 12].max()))
        writer.add_scalar('Train_Loss/supervised_loss', avg_cost[epoch][0], epoch)
        writer.add_scalar('Train_Loss/CPS_loss', avg_cost[epoch][3], epoch)
        dict = {f"DSC{n - 8}": avg_cost[epoch][n] for n in range(9, 13)}
        writer.add_scalars('Valid_Loss/', dict, epoch)
        if avg_cost[epoch][12] >= avg_cost[:, 12].max():
            best_score = avg_cost[epoch][9: ]
            torch.save(model_l.state_dict(), save_dir + '/model.pth')
    np.savetxt(save_dir + '/logging_avg_cost.npy', avg_cost, fmt='%.4f')
    np.savetxt(save_dir + '/logging_best_score_{:3f}.npy'.format(best_score[3]), best_score, fmt='%.4f')


            
