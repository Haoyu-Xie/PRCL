
# import shutup
# shutup.please()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from generalframeworks.dataset_helpers.VOC import BuildData
from generalframeworks.networks.ddp_model import Model_with_un
from generalframeworks.loss.loss import Attention_Threshold_Loss, Reco_Loss, Prcl_Loss
from generalframeworks.scheduler.my_lr_scheduler import PolyLR
from generalframeworks.util.dist_init import dist_init
from generalframeworks.util.meter import *
from generalframeworks.utils import label_onehot, fix_all_seed
from generalframeworks.util.torch_dist_sum import *
from generalframeworks.util.miou import *
import yaml
import os
import time
import torchvision.models as models
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import random

def main():
    args = parser.parse_args()  
    from torch.nn.parallel import DistributedDataParallel
    ##### Distribution init #####
    rank, local_rank, world_size = dist_init(args.port)
    print('Hello from rank {}\n'.format(rank))
    
    ##### Config init #####
    with open(args.config, 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    save_dir = './checkpoints/' + str(args.job_name) + '/' + config['Dataset']['name'] + '/' + str(config['Dataset']['num_labels'])\
              + '/' + str(config['Seed'])
    if rank == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_dir + '/config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(config)

    ##### Init Seed #####
    random.seed(config['Seed'] + 1)
    torch.manual_seed(config['Seed'] + 1)
    torch.backends.cudnn.deterministic = True

    ##### Load the dataset #####
    data = BuildData(data_path=config['Dataset']['root_dir'], num_labels=config['Dataset']['num_labels'], seed=config['Seed'])
    train_l_dataset, train_u_dataset, test_dataset = data.build()
    train_l_sampler = torch.utils.data.distributed.DistributedSampler(train_l_dataset)
    train_l_loader = torch.utils.data.DataLoader(train_l_dataset, 
                                                 batch_size=config['Dataset']['batch_size'],
                                                 num_workers=4,
                                                 pin_memory=True,
                                                 sampler=train_l_sampler,
                                                 persistent_workers=True)
    train_u_sampler = torch.utils.data.distributed.DistributedSampler(train_u_dataset)
    train_u_loader = torch.utils.data.DataLoader(train_u_dataset, 
                                                 batch_size=config['Dataset']['batch_size'],
                                                 num_workers=4,
                                                 pin_memory=True,
                                                 sampler=train_u_sampler,
                                                 persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=config['Dataset']['batch_size'],
                                              num_workers=4,
                                              pin_memory=True,
                                              persistent_workers=True)

    ##### Model init #####
    backbone = models.resnet101()
    ckpt = torch.load('./pretrained/resnet101.pth', map_location='cpu')
    backbone.load_state_dict(ckpt)
    model = Model_with_un(backbone, num_classes=config['Network']['num_class'], output_dim=256, ema_alpha=config['EMA']['alpha'], config=config)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(local_rank) # Added
    model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)

    ##### Loss init #####
    criterion = {'ce_loss': nn.CrossEntropyLoss(ignore_index=-1).cuda(local_rank),
                 'unsup_loss': Attention_Threshold_Loss(strong_threshold=config['Prcl_Loss']['strong_threshold']).cuda(local_rank),
                 'prcl_loss': Prcl_Loss(strong_threshold=config['Prcl_Loss']['test_strong_threshold'],
                                        num_queries=config['Prcl_Loss']['num_queries'],
                                        num_negatives=args.num_negatives,
                                        temp=config['Prcl_Loss']['temp']).cuda(local_rank)
                }
    

    ##### Other init #####
    optimizer = torch.optim.SGD(model.module.model.parameters(), lr=float(config['Optim']['lr']), weight_decay=float(config['Optim']['weight_decay']),
                          momentum=0.9, nesterov=True)
    optimizer_uncer = torch.optim.SGD(model.module.uncer_head.parameters(), lr=float(config['Optim']['uncer_lr']), weight_decay=float(config['Optim']['weight_decay']),
                          momentum=0.9, nesterov=True)
    total_epoch = config['Training_Setting']['epoch']
    lr_scheduler = PolyLR(optimizer, total_epoch)
    lr_scheduler_uncer = PolyLR(optimizer_uncer, total_epoch)

    if os.path.exists(args.resume):
        print('resume from', args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.module.model.load_state_dict(checkpoint['model'])
        model.module.ema_model.load_state_dict(checkpoint['ema'])
        model.module.uncer_head.load_state_dict(checkpoint['uncer_head'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_uncer.load_state_dict(checkpoint['optimizer_uncer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0


    best_miou = 0

    model.module.model.train()
    model.module.ema_model.train()
    model.module.uncer_head.train()
    for epoch in range(start_epoch, total_epoch):
        train(train_l_loader, train_u_loader, model, rank, local_rank, world_size, optimizer, optimizer_uncer, criterion, epoch, lr_scheduler, lr_scheduler_uncer, config, args)
        miou = test(test_loader, model.module.ema_model, rank)
        best_miou = max(best_miou, miou)
        if rank == 0:
            print('Epoch:{} * mIoU {:.4f} Best_mIoU {:.4f} Time {}'.format(epoch, miou, best_miou, time.asctime( time.localtime(time.time()) )))
            # Save model
            if miou == best_miou:
                save_dir = './checkpoints/' + str(args.job_name) + '/' + config['Dataset']['name'] + '/' + str(config['Dataset']['num_labels'])\
                        + '/' + str(config['Seed'])
                torch.save(
                    {
                        'epoch': epoch+1,
                        'ema': model.module.ema_model.state_dict(),
                        'model': model.module.model.state_dict(),
                        'uncer_head': model.module.uncer_head.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'optimizer_uncer': optimizer_uncer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'lr_scheduler_uncer': lr_scheduler_uncer.state_dict()
                    }, os.path.join(save_dir, 'best_model.pth'))
        


def train(train_l_loader, train_u_loader, model, rank, local_rank, world_size, optimizer, optimizer_uncer, criterion, epoch, scheduler, scheduler_uncer, config, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    sup_loss_meter = AverageMeter('Sup_loss', ':6.3f')
    unsup_loss_meter = AverageMeter('Unsup_loss', ':6.3f')
    contr_loss_meter = AverageMeter('Contr_loss', ':6.3f')
    num_class = config['Network']['num_class'] # VOC=21
    mious_conf_l = ConfMatrix(num_classes=num_class, fmt=':6.3f', name='l_miou') 
    mious_conf_u = ConfMatrix(num_classes=num_class, fmt=':6.3f', name='u_miou')
    iter_num = int(2000 / config['Dataset']['batch_size'] / world_size / len(train_l_loader)) #2000 img in a epoch
    progress = ProgressMeter(
        iter_num,
        [batch_time, data_time, sup_loss_meter, unsup_loss_meter, contr_loss_meter, mious_conf_l, mious_conf_u],
        prefix='Epoch: [{}]'.format(epoch)
    )
    # switch to train mode
    model.module.model.train()
    model.module.ema_model.train()
    model.module.uncer_head.train()

    end = time.time()
    train_u_loader.sampler.set_epoch(epoch)
    training_u_iter = iter(train_u_loader)
    for iter_i in range(iter_num):
        train_l_loader.sampler.set_epoch(epoch + iter_i * 200)
        for i, (train_l_image, train_l_label) in enumerate(train_l_loader):
            data_time.update(time.time() - end)
            train_l_image, train_l_label = train_l_image.cuda(local_rank), train_l_label.cuda(local_rank)
            train_u_image, train_u_label = training_u_iter.next()
            train_u_image, train_u_label = train_u_image.cuda(local_rank), train_u_label.cuda(local_rank)
            pred_l_large, pred_u_large, train_u_aug_label, train_u_aug_logits, rep_all, pred_all, pred_u_large_raw, uncer_all = model(train_l_image, train_u_image)

            sup_loss = criterion['ce_loss'](pred_l_large, train_l_label)
            unsup_loss = criterion['unsup_loss'](pred_u_large, train_u_aug_label, train_u_aug_logits)

            ##### Contrastive learning #####
            with torch.no_grad():
                train_u_aug_mask = train_u_aug_logits.ge(config['Prcl_Loss']['weak_threshold']).float()
                mask_all = torch.cat(((train_l_label.unsqueeze(1) >= 0).float(), train_u_aug_mask.unsqueeze(1)))
                mask_all = F.interpolate(mask_all, size=pred_all.shape[2:], mode='nearest')

                label_l = F.interpolate(label_onehot(train_l_label, num_class), size=pred_all.shape[2:], mode='nearest')
                label_u = F.interpolate(label_onehot(train_u_aug_label, num_class), size=pred_all.shape[2:], mode='nearest')
                label_all = torch.cat((label_l, label_u))

                prob_all = torch.softmax(pred_all, dim=1)
            
            prcl_loss = criterion['prcl_loss'](rep_all, uncer_all, label_all, mask_all, prob_all, epoch, (i+len(train_l_loader)*iter_i))
            total_loss = sup_loss + unsup_loss + args.lamb * prcl_loss

            # Update Meter
            sup_loss_meter.update(sup_loss.item(), pred_all.shape[0])
            unsup_loss_meter.update(unsup_loss.item(), pred_all.shape[0])
            mious_conf_l.update(pred_l_large.argmax(1).flatten(), train_l_label.flatten())
            mious_conf_u.update(pred_u_large_raw.argmax(1).flatten(), train_u_label.flatten())
            contr_loss_meter.update(prcl_loss.item(), pred_all.shape[0])
            optimizer.zero_grad()
            optimizer_uncer.zero_grad()
            total_loss.backward()
            optimizer.step()
            optimizer_uncer.step()
            model.module.ema_update()
            batch_time.update(time.time() - end)
            end = time.time()
            # if i % 20 ==0 and rank == 0:
            #     progress.display(iter_i)
    scheduler.step()
    scheduler_uncer.step()


@torch.no_grad()
def test(test_loader, model, rank):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    miou_meter = ConfMatrix(num_classes=21, fmt=':6.4f', name='test_miou')

    # switch to eval mode
    model.eval()

    end = time.time()
    test_iter = iter(test_loader)
    for _ in range(len(test_loader)):
        data_time.update(time.time() - end)
        test_image, test_label = test_iter.next()
        test_image, test_label = test_image.cuda(), test_label.cuda()
        
        pred, _, _ = model(test_image)
        pred = F.interpolate(pred, size=test_label.shape[1:], mode='bilinear', align_corners=True)

        miou_meter.update(pred.argmax(1).flatten(), test_label.flatten())
        batch_time.update(time.time() - end)
        end = time.time()

    mat = torch_dist_sum(rank, miou_meter.mat) # We refine the func without reshape
    miou = mean_intersection_over_union(mat[0]) 

    return miou



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=23456)
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--job_name', type=str, default='')
    parser.add_argument('--lamb', type=float, default=1.0)
    parser.add_argument('--num_negatives', type=int, default=512)


    main()
    
