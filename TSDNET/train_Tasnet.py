import sys
sys.path.append('./')

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader as Loader
from data_loader.Dataset_light import Datasets
from model.model import TSDNet,TSDNet_one_hot, TSDNet_plus_one_hot
from logger import set_logger
import logging
from config import option
import argparse
import torch
from trainer import trainer_Tasnet,trainer_Tasnet_one_hot,trainer_Tasnet_one_hot_regresion
import torch.optim.lr_scheduler as lr_scheduler
import random
import torch.backends.cudnn as cudnn
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.backends.cudnn.deterministic = True
DEVICE = torch.device(DEVICE)
seed = 19980228
if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def make_dataloader(opt):
    # make train's dataloader
    
    train_dataset = Datasets(
        opt['datasets']['train']['dataroot_mix'],
        opt['datasets']['train']['dataroot_targets'][0],
        opt['datasets']['train']['dataroot_targets'][1],
        opt['datasets']['train']['dataroot_targets'][2],
        opt['datasets']['audio_setting']['sample_rate'],
        opt['datasets']['audio_setting']['class_num'],
        opt['datasets']['audio_setting']['audio_length'])
    train_dataloader = Loader(train_dataset,
                              batch_size=opt['datasets']['dataloader_setting']['batch_size'],
                              num_workers=opt['datasets']['dataloader_setting']['num_workers'],
                              shuffle=opt['datasets']['dataloader_setting']['shuffle'])
    
    # make validation dataloader
    
    val_dataset = Datasets(
        opt['datasets']['val']['dataroot_mix'],
        opt['datasets']['val']['dataroot_targets'][0],
        opt['datasets']['val']['dataroot_targets'][1],
        opt['datasets']['val']['dataroot_targets'][2],
        opt['datasets']['audio_setting']['sample_rate'],
        opt['datasets']['audio_setting']['class_num'],
        opt['datasets']['audio_setting']['audio_length'])
    val_dataloader = Loader(val_dataset,
                            batch_size=opt['datasets']['dataloader_setting']['batch_size'],
                            num_workers=opt['datasets']['dataloader_setting']['num_workers'],
                            shuffle=opt['datasets']['dataloader_setting']['shuffle'])

    # make test dataloader

    test_dataset = Datasets(
        opt['datasets']['test']['dataroot_mix'],
        opt['datasets']['test']['dataroot_targets'][0],
        opt['datasets']['test']['dataroot_targets'][1],
        opt['datasets']['test']['dataroot_targets'][2],
        opt['datasets']['audio_setting']['sample_rate'],
        opt['datasets']['audio_setting']['class_num'],
        opt['datasets']['audio_setting']['audio_length'])
    test_dataloader = Loader(test_dataset,
                            batch_size=opt['datasets']['dataloader_setting']['batch_size'],
                            num_workers=opt['datasets']['dataloader_setting']['num_workers'],
                            shuffle=opt['datasets']['dataloader_setting']['shuffle'])

    return train_dataloader, val_dataloader, test_dataloader


def make_optimizer(params, opt):
    optimizer = getattr(torch.optim, opt['optim']['name'])
    if opt['optim']['name'] == 'Adam':
        optimizer = optimizer(
            params, lr=opt['optim']['lr'], weight_decay=opt['optim']['weight_decay'])
    else:
        optimizer = optimizer(params, lr=opt['optim']['lr'], weight_decay=opt['optim']
                              ['weight_decay'], momentum=opt['optim']['momentum'])

    return optimizer


def train():
    parser = argparse.ArgumentParser(
        description='Parameters for training Conv-TasNet')
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = option.parse(args.opt)
    set_logger.setup_logger(opt['logger']['name'], opt['logger']['path'],
                            screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
    logger = logging.getLogger(opt['logger']['name'])
    # build model
    logger.info("Building the model of Conv-Tasnet")
    logger.info(opt['logger']['experimental_description'])
    print(opt['model_name'])
    if opt['model_name'] == 'TSDNet_one_hot':
        net = TSDNet_one_hot(nFrameLen=opt['datasets']['audio_setting']['nFrameLen'],
                            nFrameShift=opt['datasets']['audio_setting']['nFrameShift'],
                            cls_num = opt['datasets']['audio_setting']['class_num'],
                            CNN10_settings=opt['Conv_Tasnet']['CNN10_settings'],
                            pretrainedCNN10='/apdcephfs/private_donchaoyang/tsss/Dual-Path-RNN-Pytorch2/model/Cnn10_mAP=0.380.pth'
                            )
    elif opt['model_name'] == 'TSDNet_plus_one_hot':
        net = TSDNet_plus_one_hot(nFrameLen=opt['datasets']['audio_setting']['nFrameLen'],
                            nFrameShift=opt['datasets']['audio_setting']['nFrameShift'],
                            cls_num = opt['datasets']['audio_setting']['class_num'],
                            CNN10_settings=opt['Conv_Tasnet']['CNN10_settings'],
                            pretrainedCNN10='/apdcephfs/private_donchaoyang/tsss/Dual-Path-RNN-Pytorch2/model/Cnn10_mAP=0.380.pth'
                            )
    elif opt['model_name'] =='TSDNet':
        net = TSDNet(nFrameLen=opt['datasets']['audio_setting']['nFrameLen'],
                            nFrameShift=opt['datasets']['audio_setting']['nFrameShift'],
                            cls_num = opt['datasets']['audio_setting']['class_num'],
                            CNN10_settings=opt['Conv_Tasnet']['CNN10_settings'],
                            pretrainedCNN10='/apdcephfs/private_donchaoyang/tsss/Dual-Path-RNN-Pytorch2/model/Cnn10_mAP=0.380.pth'
                            )
    else:
        assert 1==2
    # build optimizer
    logger.info("Building the optimizer of Conv-Tasnet")
    optimizer = make_optimizer(net.parameters(), opt)
    # build dataloader
    logger.info('Building the dataloader of Conv-Tasnet')
    train_dataloader, val_dataloader, test_dataloader = make_dataloader(opt)
    logger.info('Train Datasets Length: {}, Val Datasets Length: {}, Test Datasets Length: {}'.format(
        len(train_dataloader), len(val_dataloader), len(test_dataloader)))
    # build scheduler
    # scheduler = ReduceLROnPlateau(
    #     optimizer, mode='min',
    #     factor=opt['scheduler']['factor'],
    #     patience=opt['scheduler']['patience'],
    #     verbose=True, min_lr=opt['scheduler']['min_lr'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, opt['train']['epoch'])
    # build trainer
    logger.info('Building the Trainer of Conv-Tasnet')
    if opt['one_hot']:
        if opt['reg']:
            trainer = trainer_Tasnet_one_hot_regresion.Trainer(train_dataloader, val_dataloader, test_dataloader, net, optimizer, scheduler, opt)
        else:
            trainer = trainer_Tasnet_one_hot.Trainer(train_dataloader, val_dataloader, test_dataloader, net, optimizer, scheduler, opt)
    else:
        trainer = trainer_Tasnet.Trainer(train_dataloader, val_dataloader, test_dataloader, net, optimizer, scheduler, opt)
    trainer.run()


if __name__ == "__main__":
    train()

