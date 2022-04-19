'''
Thanks for Liu et al. and their paper: http://arxiv.org/abs/2104.04465
Author: Haoyu Xie

22.3.25 updated by Changqi Wang:
Add tensorboard 

'''
import os
from pathlib import Path
from imageio import save
import torch.nn.functional as F
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
from generalframeworks.dataset_helpers.VOC_helper import BuildDataloader
from generalframeworks.networks import network_factory
import generalframeworks.scheduler.my_lr_scheduler as my_lr_scheduler
from generalframeworks.networks.ema import EMA
import numpy as np
from generalframeworks.meter.meter import AverageValueMeter, Meter, ConfMatrix
from generalframeworks.augmentation.transform import batch_transform, generate_cut
from generalframeworks.loss.loss import attention_threshold_loss, compute_reco_loss
from generalframeworks.utils import iterator_

from tensorboardX import SummaryWriter


##### Config Preparation #####
warnings.filterwarnings('ignore')
parser_args = yaml_parser()

#pprint('--> Input args:')
with open('./config/MT_ReCo_config.yaml', 'r') as f:
    config = yaml.load(f.read())
config = dict_merge(config, parser_args, True)
#pprint(config)
print('Hello, it is {} now'.format(now_time()))
save_dir = config['Training_Setting']['save_dir'] + '/' + config['Dataset']['root_dir'].split('/')[-1] + '/'+ \
           str(config['Dataset']['labeled_num']) + '/' + str(config['Seed'])
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
    pascal_dataloader = BuildDataloader(config['Dataset']['name'], config['Dataset']['root_dir'],\
                                         config['Dataset']['batch_size'], config['Dataset']['labeled_num'])
    train_l_loader, train_u_loader, val_loader = pascal_dataloader.build(supervised=False)
    print('This is ' + config['Dataset']['name'] + ' dataset!')
    print('Number of labeled data:{}'.format(config['Dataset']['labeled_num']))

    ##### Device #####
    device = torch.device(config['Training_Setting']['device'] if torch.cuda.is_available() else "cpu")
    print('Training on ' + config['Training_Setting']['device'] if torch.cuda.is_available() else "cpu")

    ##### Model Initization #####
    model = network_factory(name=config['Network']['name'], num_class=config['Network']['num_class']).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(config['Optim']['lr']), weight_decay=float(config['Optim']['weight_decay']),
                                                        momentum=0.9, nesterov=True)
    #scheduler: torch.optim.lr_scheduler = getattr(my_lr_scheduler, config['Lr_Scheduler']['name']) if config['Lr_Scheduler']['name'] in ['PolyLR'] \
                                        #else getattr(my_lr_scheduler, config['Lr_Scheduler']['name'])(optimizer, **config['Lr_Scheduler'])
    scheduler = my_lr_scheduler.PolyLR(optimizer, config['Training_Setting']['epoch'], power=0.9)
    ema = EMA(model, alpha=config['EMA']['alpha'])

    ##### Metrics Initization #####
    max_epoch = config['Training_Setting']['epoch']
    # iter_max = len(train_l_loader)
    iter_max = len(train_u_loader)
    lab_mIOU = ConfMatrix(num_classes=config['Network']['num_class'])
    unlab_mIOU = ConfMatrix(num_classes=config['Network']['num_class'])
    val_mIOU = ConfMatrix(num_classes=config['Network']['num_class'])
    '''metrics = {'train_dice': np.zeros([max_epoch, config['Network']['num_class']]),
                'train_unlab_dice': np.zeros([max_epoch, config['Network']['num_class']]),
                'val_dice': np.zeros([max_epoch, config['Network']['num_class']]),
                'val_batch_dice': np.zeros([max_epoch, config['Network']['num_class']])}'''
    avg_cost = np.zeros((max_epoch, 12))
    loss = {'sup_loss': np.zeros(max_epoch),
                'unsup_loss': np.zeros(max_epoch),
                'reco_loss': np.zeros(max_epoch)}
    ##### Training #####
    for epoch in range(max_epoch):
        cost = np.zeros(3)
        # train_lab_iter = iter(train_l_loader)
        # train_unlab_iter = iter(train_u_loader)
        train_lab_iter = iterator_(train_l_loader)
        train_unlab_iter = iterator_(train_u_loader)
        
        model.train()
        ema.model.train()
        lab_mIOU.reset()
        unlab_mIOU.reset()
        val_mIOU.reset()
        train_iter_tqdm = tqdm_(range(iter_max))
        for i in train_iter_tqdm:
            lab_image, lab_label = train_lab_iter.__next__()
            lab_image, lab_label = lab_image.to(device), lab_label.to(device) #torch.Size([4, 1, 256, 256]) torch.Size([4, 256, 256])

            unlab_image, unlab_label = train_unlab_iter.__next__()
            unlab_image, unlab_label = unlab_image.to(device), unlab_label.to(device) #torch.Size([4, 1, 256, 256])

            #unlab_label_oh = class2one_hot(unlab_label, num_class=config['Network']['num_class'])#torch.Size([4, 4, 256, 256])
       
            optimizer.zero_grad()

            # Generate pseudo_labels
            with torch.no_grad():
                pred_u, _ = ema.model(unlab_image) #torch.Size([4, 4, 64, 64]) torch.Size([4, 256, 64, 64])
                pred_u_large_raw = F.interpolate(pred_u, size=unlab_label.shape[1:], mode='bilinear', align_corners=True)#torch.Size([4, 4, 256, 256])#
                pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1) #torch.Size([4, 256, 256]) torch.Size([4, 256, 256])
                
                # Random scale and  and crop, We ommit the imagenet normalization.!!!!! and we change the crop_size here instead of (512, 512)
                image_u_aug, label_u_aug, logits_u_aug = batch_transform(unlab_image, pseudo_labels, pseudo_logits,\
                    crop_size=(256, 256), scale_size=(0.5, 1.5), apply_augmentation=False)
                
                # Apply mixing strategy: cutout, cutmix or classmix
                image_u_aug, label_u_aug, logits_u_aug = generate_cut(image_u_aug, label_u_aug, logits_u_aug,\
                    mode=config['Unlabeled_Dataset']['aug_mode'])
                #torch.Size([4, 1, 256, 256]) torch.Size([4, 256, 256]) torch.Size([4, 4, 256, 256])
            
            # Generate labeled and unlabeled data loss, why the pred here is torch.Size([4, 4, 64, 64])???
            pred_l, rep_l = model(lab_image)
            pred_l_large = F.interpolate(pred_l, size=lab_label.shape[1:], mode='bilinear', align_corners=True)#torch.Size([4, 4, 256, 256])

            pred_un, rep_un = model(image_u_aug)
            pred_un_large = F.interpolate(pred_un, size=unlab_label.shape[1:], mode='bilinear', align_corners=True)

            rep_all = torch.cat((rep_l, rep_un)) #torch.Size([8, 4, 64, 64])
            pred_all = torch.cat((pred_l, pred_un)) #torch.Size([8, 4, 64, 64])

            ##### Supervised learning Loss #####
            sup_loss = F.cross_entropy(pred_l_large, lab_label, ignore_index=-1) # label must be class (index)

            ##### Unsupervised learning Loss #####
            unsup_loss = attention_threshold_loss(pred_un_large, label_u_aug, logits_u_aug, strong_threshold=config['Reco_Loss']['strong_threshold'])

            ##### Reco Loss #####
            if config['Reco_Loss']['is_available']:
                with torch.no_grad():
                    mask_u_aug = logits_u_aug.ge(config['Reco_Loss']['weak_threshold']).float() # confidence from logits torch.Size([4, 256, 256])
                    mask_all = torch.cat(((lab_label.unsqueeze(1) >= 0).float(), mask_u_aug.unsqueeze(1))) #torch.Size([8, 1, 256, 256])
                    mask_all = F.interpolate(mask_all, size=pred_all.shape[2:], mode='nearest')

                    lab_l = F.interpolate(label_onehot(lab_label, num_class=config['Network']['num_class']), size=pred_all.shape[2: ], mode='nearest')
                    lab_un = F.interpolate(label_onehot(label_u_aug, num_class=config['Network']['num_class']), size=pred_all.shape[2: ], mode='nearest')
                    label_all = torch.cat((lab_l, lab_un))

                    prob_l = torch.softmax(pred_l, dim=1)
                    prob_un = torch.softmax(pred_un, dim=1)
                    prob_all = torch.cat((prob_l, prob_un)) #torch.Size([8, 4, 64, 64])

                reco_loss = compute_reco_loss(rep_all, label_all, mask_all, prob_all, config['Reco_Loss']['strong_threshold'],
                                              config['Reco_Loss']['temp'], config['Reco_Loss']['num_queries'],
                                              config['Reco_Loss']['num_nagetives'])
            else:
                reco_loss = torch.tensor(0.0)
            
            loss = sup_loss + unsup_loss + reco_loss
            loss.backward()
            optimizer.step()
            ema.update(model)

            lab_mIOU.add(pred_l_large, lab_label)
            unlab_mIOU.add(pred_u_large_raw, unlab_label)
            cost[0] = sup_loss.item()
            cost[1] = unsup_loss.item()
            cost[2] = reco_loss.item()
            avg_cost[epoch, :3] += cost / iter_max
            # Progress Bar
            train_iter_tqdm.set_description('Training' + ':' + 'Sup:{:.3f}|Unsup:{:.3f}|Reco:{:.3f}|Total:{:.3f}'.format(cost[0], cost[1], cost[2], (cost[0]+cost[1]+cost[2])/3))
            train_iter_tqdm.set_postfix({'Lab_mIOU': '{:.3f}'.format(lab_mIOU.value()[0]), 'Unlab_mIOU': '{:.3f}'.format(unlab_mIOU.value()[0])})
        # Log dice in one epoch
        lab_miou, lab_acc = lab_mIOU.value()
        avg_cost[epoch, 3] = lab_miou # mean mIOU for every class
        avg_cost[epoch, 4] = lab_acc # mean acc for every class
        unlab_miou, unlab_acc = unlab_mIOU.value()
        avg_cost[epoch, 5] = unlab_miou # mean mIOU for every class
        avg_cost[epoch, 6] = unlab_acc # mean acc for every class
        ##### Evaluation #####
        with torch.no_grad():
            model.eval()
            ema.model.eval()
            val_iter = iter(val_loader)
            max_iter = len(val_loader)
            iter_val_tqdm = tqdm_(range(max_iter))
            for i in iter_val_tqdm:
                val_image, val_label = val_iter.next()
                val_image, val_label = val_image.to(device), val_label.to(device)

                pred, _ = ema.model(val_image)
                pred = F.interpolate(pred, size=val_label.shape[1:], mode='bilinear', align_corners=True)
                loss = F.cross_entropy(pred, val_label, ignore_index=-1)
                # Log
                val_mIOU.add(pred, val_label)
                avg_cost[epoch, 7] += loss.item() / max_iter
                mIOU_tmp, Acc_tmp = val_mIOU.value(mode='mean')
                iter_val_tqdm.set_description('Validing' + ':' + 'mIOU:{:.3f}|Acc:{:.3f}'.format(mIOU_tmp, Acc_tmp))
            val_miou, val_acc = val_mIOU.value()
            avg_cost[epoch, 8] = val_miou
            avg_cost[epoch, 9] = val_acc
        scheduler.step()
        print('\n  EPOCH | TRAIN  |SUP_LOSS|UNSUP_LOSS|RECO_LOSS|Lab_mIOU|Lab_Acc|Unlab_mIOU|Unlab_Acc| \n   {:03d}  |        | {:.4f} |  {:.4f}  | {:.4f}  | {:.4f} | {:.4f}|  {:.4f}  |  {:.4f} | \n        |  Test  |  Loss  |  mIOU  |  Acc.  |Best mIOU|\n        |        | {:.4f} | {:.4f} | {:.4f} | {:.4f}  |'\
            .format(epoch, avg_cost[epoch][0], avg_cost[epoch][1], avg_cost[epoch][2], avg_cost[epoch][3], avg_cost[epoch][4], avg_cost[epoch][5], avg_cost[epoch][6], avg_cost[epoch][7], avg_cost[epoch][8], avg_cost[epoch][9], avg_cost[:, 8].max()))
        writer.add_scalar('Train_Loss/supervised_loss', avg_cost[epoch][0], epoch)
        writer.add_scalar('Train_Loss/CPS_loss', avg_cost[epoch][1], epoch)
        dict_train = {f"loss{n - 2}": avg_cost[epoch][n] for n in range(3, 7)}
        writer.add_scalars('Train_mIOU/', dict_train, epoch)
        dict_val = {f"loss{n - 7}": avg_cost[epoch][n] for n in range(8, 10)}
        writer.add_scalars('Valid_mIOU/', dict_val, epoch)
        if avg_cost[epoch][5] >= avg_cost[:, 8].max():
            best_score = avg_cost[epoch][8: ]
            torch.save(ema.model.state_dict(), save_dir + '/model.pth')
    np.savetxt(save_dir + '/logging_avg_cost.npy', avg_cost, fmt='%.4f')
    np.savetxt(save_dir + '/logging_best_score_{:3f}.npy'.format(best_score[2]), best_score, fmt='%.4f')


            
