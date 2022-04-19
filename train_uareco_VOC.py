
import shutup
shutup.please()
import os
from random import Random, random
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import yaml
import random
from generalframeworks.dataset_helpers.VOC_helper import BuildDataLoader, batch_transform
from generalframeworks.networks.deeplabv3.deeplabv3 import DeepLabv3Plus, DeepLabv3Plus_MC
from generalframeworks.networks.ema import EMA
from generalframeworks.scheduler.my_lr_scheduler import PolyLR
from generalframeworks.meter.meter import ConfMatrix
from generalframeworks.loss.loss import Reco_Loss, Uncertainty_Aware_Reco, Attention_Threshold_Loss
from generalframeworks.augmentation.transform import generate_cut
from generalframeworks.utils import label_onehot, fix_all_seed, tqdm_, apply_dropout
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tensorboardX import SummaryWriter
from generalframeworks.scheduler.rampscheduler import RampScheduler

def main(rank, config):
    ##### Distributed Training Preparation #####
    dist.init_process_group(backend='nccl', world_size=config['Distributed']['world_size'], rank=rank)
    torch.cuda.set_device(rank)
    torch.autograd.set_detect_anomaly(True)
    print("Hello from rank {}\n".format(rank))
    fix_all_seed(config['Seed'])

    ##### Load the dataset #####
    data_loader = BuildDataLoader(batch_size=config['Dataset']['batch_size'], num_labels=config['Dataset']['num_labels'], distributed=False)
    train_l_loader, train_u_loader, test_loader = data_loader.build(supervised=False)

    ##### Model Initialization #####
    model = DeepLabv3Plus_MC(models.resnet101(pretrained=True), num_classes=data_loader.num_segments, output_dim=256)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model = DDP(model)
    ema = EMA(model, alpha=config['EMA']['alpha'])
    ema.model = DDP(ema.model.cuda())
    optimizer = optim.SGD(model.parameters(), lr=float(config['Optim']['lr']), weight_decay=float(config['Optim']['weight_decay']),
                          momentum=0.9, nesterov=True)
    total_epoch = int(config['Training_Setting']['epoch'])
    lr_scheduler = PolyLR(optimizer, total_epoch)
    warm_scheduler = RampScheduler(begin_epoch=0, max_epoch=20, max_value=1, ramp_mult=-5)
    train_epoch = len(train_l_loader)
    test_epoch = len(test_loader)

    ##### Loss Initialization #####
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1).cuda()
    att_th_loss = Attention_Threshold_Loss(strong_threshold=config['Reco_Loss']['strong_threshold']).cuda()
    uareco_loss = Uncertainty_Aware_Reco(strong_threshold=config['Reco_Loss']['strong_threshold'], temp=config['Reco_Loss']['temp'],
                          num_queries=config['Reco_Loss']['num_queries'], num_negatives=config['Reco_Loss']['num_negatives'], scheduler=warm_scheduler).cuda()
    
    if dist.get_rank() == 0:
        save_dir = config['Training_Setting']['save_dir'] + '/' + config['Dataset']['name'] + '/' + str(config['Dataset']['num_labels'])\
              + '/' + str(config['Seed'])
        avg_cost = np.zeros((total_epoch, 10))
        iteration = 0
        best_iu = 0
        valid_pixel_num = np.zeros((total_epoch, data_loader.num_segments))
        writer = SummaryWriter(save_dir + '/logs')

    ##### Training #####
    for index in range(total_epoch):
        training_l_iter = iter(train_l_loader)
        training_u_iter = iter(train_u_loader)
        model.train()
        ema.model.train()
        if dist.get_rank() == 0:
            cost = np.zeros(3)
            l_conf_mat = ConfMatrix(data_loader.num_segments)
            u_conf_mat = ConfMatrix(data_loader.num_segments)
            train_epoch_tqdm = tqdm_(range(train_epoch))
        else:
            train_epoch_tqdm = range(train_epoch)

        for i in train_epoch_tqdm:
            train_l_image, train_l_label = training_l_iter.next()
            train_l_image, train_l_label = train_l_image.cuda(), train_l_label.cuda()

            train_u_image, train_u_label = training_u_iter.next()
            train_u_image, train_u_label = train_u_image.cuda(), train_u_label.cuda()

            optimizer.zero_grad()

            # Generate pseudo-labels
            with torch.no_grad():
                pred_u, _ = ema.model(train_u_image)
                pred_u_large_raw = F.interpolate(pred_u, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)
                pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)

                # Randomly scale images
                train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_image, pseudo_labels, pseudo_logits,
                                                                                           crop_size=data_loader.crop_size,
                                                                                           scale_size=data_loader.scale_size,
                                                                                           augmentation=False)
                # Apply mixing strategy
                train_u_aug_image, train_u_aug_label, train_u_aug_logits = generate_cut(train_u_aug_image, train_u_aug_label, train_u_aug_logits,
                                                                                        mode=config['Dataset']['mix_mode'])
                # Apply augmnetation : color jitter + flip + gaussian blur
                train_u_aug_image, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_aug_image, train_u_aug_label, train_u_aug_logits,
                                                                                           crop_size=data_loader.crop_size,
                                                                                           scale_size=(1.0, 1.0),
                                                                                           augmentation=True)
            
            # Generate labeled and unlabeled loss
            pred_l, rep_l = model(train_l_image)
            pred_l_large = F.interpolate(pred_l, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)
            '''
            with torch.no_grad():
                preds_l = torch.zeros_like(pred_l)
                for _ in range(config['Network']['MC_T']):
                    pred_l, _= model(train_l_image)
                    preds_l += torch.softmax(pred_l, dim=1)
                preds_l /= config['Network']['MC_T']
                uncertainty_l = -1.0 * torch.sum(preds_l * torch.log(preds_l + 1e-8), dim=1, keepdim=True)'''
            uncertainty_l = torch.ones(pred_l.shape[0], 1, pred_l.shape[2], pred_l.shape[3]).cuda()

            pred_u, rep_u = model(train_u_aug_image)
            pred_u_large = F.interpolate(pred_u, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)  
            # Uncertainty Generation
            with torch.no_grad():
                model.eval()
                model.apply(apply_dropout)
                preds_u = torch.zeros_like(pred_u)
                for _ in range(config['Network']['MC_T']):
                    pred_u, _= model(train_u_aug_image)
                    preds_u += torch.softmax(pred_u, dim=1)
                preds_u /= config['Network']['MC_T']
                uncertainty_u = -1.0 * torch.sum(preds_u * torch.log(preds_u + 1e-8), dim=1, keepdim=True).cuda()

            model.train()
            rep_all = torch.cat((rep_l, rep_u))
            pred_all = torch.cat((pred_l, pred_u))

            sup_loss = ce_loss(pred_l_large, train_l_label)
            unsup_loss = att_th_loss(pred_u_large, train_u_aug_label, train_u_aug_logits)

            if config['Reco_Loss']['is_available']:
                with torch.no_grad():
                    train_u_aug_mask = train_u_aug_logits.ge(config['Reco_Loss']['weak_threshold']).float()
                    mask_all = torch.cat(((train_l_label.unsqueeze(1) >= 0).float(), train_u_aug_mask.unsqueeze(1)))
                    mask_all = F.interpolate(mask_all, size=pred_all.shape[2:], mode='nearest')

                    label_l = F.interpolate(label_onehot(train_l_label, data_loader.num_segments), size=pred_all.shape[2:], mode='nearest')
                    label_u = F.interpolate(label_onehot(train_u_aug_label, data_loader.num_segments), size=pred_all.shape[2:], mode='nearest')
                    label_all = torch.cat((label_l, label_u))

                    prob_l = torch.softmax(pred_l, dim=1)
                    prob_u = torch.softmax(pred_u, dim=1)
                    prob_all = torch.cat((prob_l, prob_u))

                contrastive_loss = uareco_loss(rep_all, label_all, mask_all, prob_all, uncertainty_l, uncertainty_u)
            else:
                contrastive_loss = torch.tensor(0.0)

            loss = sup_loss + unsup_loss + contrastive_loss
            loss.backward()
            optimizer.step()
            ema.update(model)
            
            if dist.get_rank() == 0:
                l_conf_mat.update(pred_l_large.argmax(1).flatten(), train_l_label.flatten())
                u_conf_mat.update(pred_u_large_raw.argmax(1).flatten(), train_u_label.flatten())
                cost[0] = sup_loss.item()
                cost[1] = unsup_loss.item()
                cost[2] = contrastive_loss.item()
                avg_cost[index, :3] += cost / train_epoch
                tmp_valid_num = uareco_loss.tmp_valid_pixel.view(uareco_loss.tmp_valid_pixel.shape[0], uareco_loss.tmp_valid_pixel.shape[1], -1).sum(-1).mean(0)
                valid_pixel_num[index] += tmp_valid_num.cpu().numpy() / train_epoch

                iteration += 1
                # Progress Bar
                train_epoch_tqdm.set_description('Training: Sup_loss:{:.3f}|Unsup_loss:{:.3f}|Reco_loss:{:.3f}|Total_loss'\
                    .format(cost[0], cost[1], cost[2], (cost[0] + cost[1] + cost[2])))
                #train_epoch_tqdm.set_postfix({'Lab_mIoU': l_conf_mat.get_metrics()[0], 'UnLab_mIoU': u_conf_mat.get_metrics()[0]})
        lr_scheduler.step()
        warm_scheduler.step()
        if dist.get_rank() == 0:
            avg_cost[index, 3:5] = l_conf_mat.get_metrics()
            avg_cost[index, 5:7] = u_conf_mat.get_metrics()
            sup_loss_dict = {'sup_loss': avg_cost[index, 0]}
            unsup_loss_dict = {'unsup_loss': avg_cost[index, 1]}
            contrastive_loss_dict = {'uareco_loss': avg_cost[index, 2]}
            total_loss_dict = {'total_loss': (avg_cost[index, 0] + avg_cost[index, 1] + avg_cost[index, 2]) / 3}
            valid_pixel_num_dict = {'{}'.format(c): valid_pixel_num[index][c] for c in range(data_loader.num_segments)}
            valid_pixel_num_dict['total_num'] = valid_pixel_num[index].sum() 
            writer.add_scalars('Train_loss', sup_loss_dict, index)
            writer.add_scalars('Train_loss', unsup_loss_dict, index)
            writer.add_scalars('Train_loss', contrastive_loss_dict, index)
            writer.add_scalars('Train_loss', total_loss_dict, index)
            writer.add_scalars('Valid_pixel_num', valid_pixel_num_dict, index)
            
            ##### Evaluate on validation set #####
            with torch.no_grad():
                ema.model.eval()
                test_iter = iter(test_loader)
                conf_mat = ConfMatrix(data_loader.num_segments)
                for i in range(test_epoch):
                    test_image, test_label = test_iter.next()
                    test_image, test_label = test_image.cuda(), test_label.cuda()

                    pred, rep = ema.model(test_image)
                    pred = F.interpolate(pred, size=test_label.shape[1:], mode='bilinear', align_corners=True)
                    loss = F.cross_entropy(pred, test_label, ignore_index=-1)

                    conf_mat.update(pred.argmax(1).flatten(), test_label.flatten())
                    avg_cost[index, 7] += loss.item() / test_epoch

                avg_cost[index, 8:] = conf_mat.get_metrics()
                dict_val = {'val_loss': avg_cost[index, 7]}
                dict_iu_acc = {'val_mIoU': avg_cost[index, 8], 'val_acc': avg_cost[index, 9]}
                writer.add_scalars('Val_loss', dict_val, index)
                writer.add_scalars('Valid', dict_iu_acc, index)

            print(
                'EPOCH: {:04d} ITER: {:04d} | TRAIN [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} || Test [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f}'
                .format(index, iteration, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2],
                        avg_cost[index][3], avg_cost[index][4], avg_cost[index][5], avg_cost[index][6], avg_cost[index][7],
                        avg_cost[index][8],
                        avg_cost[index][9]))
            print('Top: mIoU {:.4f} Acc {:.4f}'.format(avg_cost[:, 8].max(), avg_cost[:, 9].max()))
            if avg_cost[index, 8] > best_iu:
                best_iu = avg_cost[index, 8]
                best_acc = avg_cost[index, 9]
                best_epoch = index
                np.savetxt(os.path.join(save_dir, 'best_score.npy'), np.array([best_iu, best_acc, best_epoch]), fmt='%.4f')
                torch.save(ema.model.module.state_dict(), os.path.join(save_dir, 'best_model.pth'))
    if dist.get_rank() == 0:
        writer.close()




if __name__ == '__main__':
    ##### Config Preparation #####
    with open('./config/VOC_config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    save_dir = config['Training_Setting']['save_dir'] + '/' + config['Dataset']['name'] + '/' + str(config['Dataset']['num_labels'])\
              + '/' + str(config['Seed'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + '/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    ##### Init Random Seed #####
    fix_all_seed(config['Seed'])

    ##### Init Distributed Env #####
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(config['Distributed']['world_size'])
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    mp.spawn(main, nprocs=int(config['Distributed']['world_size']), args=(config,))
