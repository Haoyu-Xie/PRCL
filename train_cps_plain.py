# Author: Haoyu Xie
import os
from pathlib import Path
import sched
from statistics import mode
from unicodedata import name
from imageio import save
import torch.nn.functional as F
import torch
from random import Random
import warnings
from typing import List
import yaml
from generalframeworks.utils import dict_merge, fix_all_seed, yaml_parser, now_time, class2one_hot, label_onehot, tqdm_, iterator_
from generalframeworks.dataset_helpers.dataset import BaseDataSet
from torch.utils.data import DataLoader, sampler
from torchvision import transforms
from pprint import pprint
from generalframeworks.dataset_helpers.make_list import make_ACDC_list, make_ACDC_list_llu
from generalframeworks.dataset_helpers.ACDC_helper import ACDC_Dataset
from generalframeworks.networks import network_factory
import generalframeworks.scheduler.my_lr_scheduler as my_lr_scheduler
from generalframeworks.scheduler.rampscheduler import RampScheduler
from generalframeworks.networks.ema import EMA
import numpy as np
from generalframeworks.meter.meter import AverageValueMeter, Meter
from generalframeworks.augmentation.transform import batch_transform, generate_cut
from generalframeworks.loss.loss import KL_Divergence_2D
from generalframeworks.module.vat import VATGenerator


##### Config Preparation #####
warnings.filterwarnings('ignore')
parser_args = yaml_parser()

#pprint('--> Input args:')
with open('./config/cps_config.yaml', 'r') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
config = dict_merge(config, parser_args, True)
#pprint(config)
print('Hello, it is {} now'.format(now_time()))
print('Cross pseudo-label supervised training is preparing...')
save_dir = config['Training_Setting']['save_dir'] + '/' + config['Dataset']['root_dir'].split('/')[-1] + '/'+ \
           str(config['Labeled_Dataset']['num_patient']) + '/' + str(config['Seed'])
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(save_dir + '/config.yaml', 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)
##### Init Random Seed #####
fix_all_seed(int(config['Seed']))                   

if __name__ == '__main__':
    ##### Dataset Preparation #####
    make_ACDC_list_llu(dataset_dir=config['Dataset']['root_dir'], save_dir=save_dir, labeled_num=config['Labeled_Dataset']['num_patient']) # Make there dataset lists
    train_l_dataset_0 = ACDC_Dataset(root_dir=config['Dataset']['root_dir'], save_dir=save_dir, mode='label_0', meta_label=False)
    train_l_dataset_1 = ACDC_Dataset(root_dir=config['Dataset']['root_dir'], save_dir=save_dir, mode='label_1', meta_label=False)
    train_u_dataset = ACDC_Dataset(root_dir=config['Dataset']['root_dir'], save_dir=save_dir,mode='unlabel', meta_label=False)  
    val_dataset = ACDC_Dataset(root_dir=config['Dataset']['root_dir'], save_dir=save_dir, mode='val', meta_label=False) 
    train_l_loader_0 = DataLoader(train_l_dataset_0,
                                batch_size=config['Labeled_Dataset']['batch_size'], 
                                sampler=sampler.RandomSampler(data_source=train_l_dataset_0),
                                drop_last=True)
    train_l_loader_1 = DataLoader(train_l_dataset_1,
                                batch_size=config['Labeled_Dataset']['batch_size'], 
                                sampler=sampler.RandomSampler(data_source=train_l_dataset_1),
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
    model_0 = network_factory(name=config['Network']['name'], num_class=config['Network']['num_class']).to(device)
    model_1 = network_factory(name=config['Network']['name'], num_class=config['Network']['num_class']).to(device)

    optimizer_0 = torch.optim.SGD(model_0.parameters(), lr=float(config['Optim']['lr']), weight_decay=float(config['Optim']['weight_decay']),
                                                        momentum=0.9)
    optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=float(config['Optim']['lr']), weight_decay=float(config['Optim']['weight_decay']),
                                                        momentum=0.9)
    scheduler_0 = my_lr_scheduler.PolyLR(optimizer_0, config['Training_Setting']['epoch'], power=0.9)
    scheduler_1 = my_lr_scheduler.PolyLR(optimizer_1, config['Training_Setting']['epoch'], power=0.9)
    cps_scheduler = RampScheduler(**config['Cps_Loss'])
    vat_scheduler = RampScheduler(**config['Vat_scheduler'])

    ##### Metrics Initization #####
    max_epoch = config['Training_Setting']['epoch']
    iter_max = len(train_u_loader)
    lab_dice_0 = AverageValueMeter(num_class=config['Network']['num_class'])
    lab_dice_1 = AverageValueMeter(num_class=config['Network']['num_class'])
    unlab_dice_0 = AverageValueMeter(num_class=config['Network']['num_class'])
    unlab_dice_1 = AverageValueMeter(num_class=config['Network']['num_class'])
    val_dice_0 = AverageValueMeter(num_class=config['Network']['num_class'])
    val_dice_1 = AverageValueMeter(num_class=config['Network']['num_class'])      
    avg_cost = np.zeros((max_epoch, 22))
    
    ##### Training #####
    for epoch in range(max_epoch):
        cost = np.zeros(4) # loss_sup_0, loss_sup_1, cps_loss_0, cps_loss_1
        train_lab_iter_0 = iterator_(train_l_loader_0)
        train_lab_iter_1 = iterator_(train_l_loader_1)
        train_unlab_iter = iter(train_u_loader)
        
        model_0.train()
        model_1.train()
        lab_dice_0.reset()
        lab_dice_1.reset()
        unlab_dice_0.reset()
        unlab_dice_1.reset()
        val_dice_0.reset()
        val_dice_1.reset()
        train_iter_tqdm = tqdm_(range(iter_max))
        for i in train_iter_tqdm:
            lab_image_0, lab_label_0 = train_lab_iter_0.__next__()
            lab_image_0, lab_label_0 = lab_image_0.to(device), lab_label_0.to(device) #torch.Size([4, 1, 256, 256]) torch.Size([4, 256, 256])
            lab_image_1, lab_label_1 = train_lab_iter_1.__next__()
            lab_image_1, lab_label_1 = lab_image_1.to(device), lab_label_1.to(device) #torch.Size([4, 1, 256, 256]) torch.Size([4, 256, 256])

            unlab_image, unlab_label = train_unlab_iter.next()
            unlab_image, unlab_label = unlab_image.to(device), unlab_label.to(device) #torch.Size([4, 1, 256, 256])
       
            optimizer_0.zero_grad()
            optimizer_1.zero_grad()
            # Generate pseudo labels
            
            unlab_logits_0 = model_0(unlab_image)
            lab_logits_0 = model_0(lab_image_0)

            unlab_logits_1 = model_1(unlab_image)
            lab_logits_1 = model_1(lab_image_1)

            pred_0 = torch.softmax(torch.cat([lab_logits_0, unlab_logits_0], dim=0), dim=1)
            pred_1 = torch.softmax(torch.cat([lab_logits_1, unlab_logits_1], dim=0), dim=1)
            # Generate pseudo labels
            _, max_0 = torch.max(pred_0, dim=1)
            _, max_1 = torch.max(pred_1, dim=1)
            
            # Cross supervise
            cps_loss_0 = F.cross_entropy(pred_0, max_1, reduction='mean', ignore_index=-1)
            cps_loss_1 = F.cross_entropy(pred_1, max_0, reduction='mean', ignore_index=-1)
            # Supervised learning 
            sup_loss_0 = F.cross_entropy(lab_logits_0, lab_label_0, reduction='mean', ignore_index=-1)
            sup_loss_1 = F.cross_entropy(lab_logits_1, lab_label_1, reduction='mean', ignore_index=-1)

            # Virtual Adversarial Training
            if config['Vat_Loss']['is_available']:
                image_adv_0, noise_0 = VATGenerator(model_0)(unlab_image)
                adv_pred_0 = torch.softmax(model_0(image_adv_0)[0], dim=1)
                real_pred_0 = torch.softmax(model_0(unlab_image)[0].detach(), dim=1)
                vat_loss_0 = KL_Divergence_2D(reduce=True)(adv_pred_0, real_pred_0)

                image_adv_1, noise_1 = VATGenerator(model_1)(unlab_image)
                adv_pred_1 = torch.softmax(model_1(image_adv_1)[0], dim=1)
                real_pred_1 = torch.softmax(model_0(unlab_image)[0].detach(), dim=1)
                vat_loss_1 = KL_Divergence_2D(reduce=True)(adv_pred_1, real_pred_1)

                vat_loss = vat_loss_1 + vat_loss_1
            else:
                vat_loss = torch.Tensor(0.0)

            # Total loss    
            #loss = cps_scheduler.value * (cps_loss_0 + cps_loss_1) + sup_loss_0 + sup_loss_1 + vat_scheduler.value * vat_loss
            loss = sup_loss_0 + sup_loss_1
            cost[0], cost[1], cost[2], cost[3] = sup_loss_0.item(), sup_loss_1.item(), cps_loss_0.item(), cps_loss_1.item()
            avg_cost[epoch, :4] += cost / iter_max 
            lab_dice_0.add(lab_logits_0, lab_label_0)
            lab_dice_1.add(lab_logits_1, lab_label_1)
            unlab_dice_0.add(unlab_logits_0, unlab_label)
            unlab_dice_1.add(unlab_logits_1, unlab_label)
            
    
            loss.backward()
            # Progress bar
            train_iter_tqdm.set_description('Training' + ':' + 'Sup_0:{:.3f}|Sup_1:{:.3f}|Cps_0:{:.3f}|Cps_1:{:.3f}|Total:{:.3f}'.format(cost[0], cost[1], cost[2], cost[3], (cost[0]+cost[1]+cost[2]+cost[3])/4))
            train_iter_tqdm.set_postfix({'|Lab_DSC_0|': '{:.3f}'.format(lab_dice_0.value(i)), 
                                        '|Lab_DSC_1|': '{:.3f}'.format(lab_dice_1.value(i)),
                                        '|Unlab_DSC_0|': '{:.3f}'.format(unlab_dice_0.value(i)),
                                        '|Unlab_DSC_1|': '{:.3f}'.format(unlab_dice_1.value(i))})
            lab_dice_0_dct = lab_dice_0.summary()
            for i, key in enumerate(lab_dice_0_dct):
                avg_cost[epoch, 4 + i] = lab_dice_0_dct[key] / 3 # model_0 DSC_1, DSC_2, DSC_3
            avg_cost[epoch, 7] = avg_cost[4: 7].mean()
            lab_dice_1_dct = lab_dice_1.summary()
            for i, key in enumerate(lab_dice_1_dct):
                avg_cost[epoch, 8 + i] = lab_dice_1_dct[key] / 3 # model_1 DSC_1, DSC_2, DSC_3
            avg_cost[epoch, 11] = avg_cost[8: 11].mean()
        
        ##### Evaluation #####
        with torch.no_grad():
            model_0.eval()
            model_1.eval()
            val_iter = iter(val_loader)
            val_dice_0.reset()
            val_dice_1.reset()
            max_iter = len(val_loader)
            iter_val_tqdm = tqdm_(range(max_iter))
            for i in iter_val_tqdm:
                val_image, val_label = val_iter.next()
                val_image, val_label = val_image.to(device), val_label.to(device)

                pred_0 = model_0(val_image)
                loss_0 = F.cross_entropy(pred_0, val_label)
                pred_1 = model_1(val_image)
                loss_1 = F.cross_entropy(pred_1, val_label)
                # Log
                val_dice_0.add(pred_0, val_label)
                val_dice_1.add(pred_1, val_label)
                avg_cost[epoch, 12] += loss_0.item() / max_iter
                avg_cost[epoch, 13] += loss_1.item() / max_iter
                dice_tmp_0 = [val_dice_0.value(i, mode='all')[c] for c in range(1, 4)]
                dice_tmp_1 = [val_dice_1.value(i, mode='all')[c] for c in range(1, 4)]
                dice_tmp_0.append(np.array(dice_tmp_0).mean())
                dice_tmp_1.append(np.array(dice_tmp_1).mean())

                iter_val_tqdm.set_description('Validing' + ':' + '|Model_0|DSC_1:{:.3f}|DSC_2:{:.3f}|DSC_3:{:.3f}|DSC_AVG:{:.3f}'.format(*dice_tmp_0))
                iter_val_tqdm.set_description('Validing' + ':' + '|Model_1|DSC_1:{:.3f}|DSC_2:{:.3f}|DSC_3:{:.3f}|DSC_AVG:{:.3f}'.format(*dice_tmp_1))
            val_dct_0 = val_dice_0.summary()
            val_dct_1 = val_dice_1.summary()
            for i, key in enumerate(val_dct_0):
                avg_cost[epoch, 14 + i] = val_dct_0[key]
            avg_cost[epoch, 17] = avg_cost[epoch, 14: 17].mean()
            for i, key in enumerate(val_dct_1):
                avg_cost[epoch, 18 + i] = val_dct_1[key]
            avg_cost[epoch, 21] = avg_cost[epoch, 18: 21].mean()
        scheduler_0.step()
        scheduler_1.step()
        cps_scheduler.step()
        vat_scheduler.step()
        print('|  EPOCH | TRAIN  |SUP_LOSS|        |CPS_LOSS|        |')
        print('|  {:03d}   |        | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'\
            .format(epoch, avg_cost[epoch][0], avg_cost[epoch][1], avg_cost[epoch][2], avg_cost[epoch][3]))
        print('|  MOD_0 | DISC_1 | DISC_2 | DISC_3 |DSIC_AVG|')
        print('|        | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'\
            .format(avg_cost[epoch][4], avg_cost[epoch][5], avg_cost[epoch][6], avg_cost[epoch][7]))
        print('|  MOD_1 | DISC_1 | DISC_2 | DISC_3 |DSIC_AVG|')
        print('|        | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'\
            .format(avg_cost[epoch][8], avg_cost[epoch][9], avg_cost[epoch][10], avg_cost[epoch][11]))
        print('|  TEST  |  MOD_0 |  LOSS  |  MOD_1 |  LOSS  |')
        print('|        |        | {:.4f} |        | {:.4f} |'\
            .format(avg_cost[epoch][12], avg_cost[epoch][13]))
        print('|  MOD_0 | DISC_1 | DISC_2 | DISC_3 |DSIC_AVG|TOP_DISC|')
        print('|        | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'\
            .format(avg_cost[epoch][14], avg_cost[epoch][15], avg_cost[epoch][16], avg_cost[epoch][17], avg_cost[:, 17].max()))
        print('|  MOD_1 | DISC_1 | DISC_2 | DISC_3 |DSIC_AVG|TOP_DISC|')
        print('|        | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'\
            .format(avg_cost[epoch][18], avg_cost[epoch][19], avg_cost[epoch][20], avg_cost[epoch][21], avg_cost[:, 21].max()))
        if avg_cost[epoch][17] >= avg_cost[:, 17].max():
            best_score_0 = avg_cost[epoch][14: 18]
            torch.save(model_0.state_dict(), save_dir + '/model_0.pth')
        if avg_cost[epoch][21] >= avg_cost[:, 21].max():
            best_score_1 = avg_cost[epoch][18: 21]
            torch.save(model_1.state_dict(), save_dir + '/model_0.pth')
    np.savetxt(save_dir + '/logging_avg_cost.npy', avg_cost, fmt='%.4f')
    np.savetxt(save_dir + '/logging_best_score.npy', best_score_0, fmt='%.4f')
    np.savetxt(save_dir + '/logging_best_score.npy', best_score_1, fmt='%.4f')


            
