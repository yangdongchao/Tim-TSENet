import sys
sys.path.append('./')
from torch.utils.data import DataLoader as Loader
from data_loader.Dataset import Datasets
from model.model import TSENet,TSENet_one_hot
from logger import set_logger
import logging
from config import option
import argparse
import torch
from trainer import trainer_Tasnet,trainer_Tasnet_one_hot
import torch.optim.lr_scheduler as lr_scheduler

def make_dataloader(opt):
    # make training dataloader
    train_dataset = Datasets(
        opt['datasets']['train']['dataroot_mix'],
        opt['datasets']['train']['dataroot_targets'][0], # s1
        opt['datasets']['train']['dataroot_targets'][1], # ref
        opt['datasets']['train']['dataroot_targets'][2], # time information
        opt['datasets']['audio_setting']['sample_rate'],
        opt['datasets']['audio_setting']['class_num'],
        opt['datasets']['audio_setting']['audio_length'],
        opt['datasets']['audio_setting']['nFrameShift'])
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
        opt['datasets']['audio_setting']['audio_length'],
        opt['datasets']['audio_setting']['nFrameShift'])
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
        opt['datasets']['audio_setting']['audio_length'],
        opt['datasets']['audio_setting']['nFrameShift'])
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
    parser = argparse.ArgumentParser(description='Parameters for training TSENet')
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = option.parse(args.opt)
    set_logger.setup_logger(opt['logger']['name'], opt['logger']['path'],
                            screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
    logger = logging.getLogger(opt['logger']['name'])
    # build model
    logger.info("Building the model of TSENet")
    logger.info(opt['logger']['experimental_description'])
    if opt['one_hot'] == 1:
        net = TSENet_one_hot(N=opt['TSENet']['N'],
                 B=opt['TSENet']['B'],
                 H=opt['TSENet']['H'],
                 P=opt['TSENet']['P'],
                 X=opt['TSENet']['X'],
                 R=opt['TSENet']['R'],
                 norm=opt['TSENet']['norm'],
                 num_spks=opt['TSENet']['num_spks'],
                 causal=opt['TSENet']['causal'],
                 cls_num=opt['TSENet']['class_num'],
                 nFrameLen=opt['datasets']['audio_setting']['nFrameLen'],
                 nFrameShift=opt['datasets']['audio_setting']['nFrameShift'],
                 nFFT=opt['datasets']['audio_setting']['nFFT'],
                 fusion=opt['TSENet']['fusion'],
                 usingEmb=opt['TSENet']['usingEmb'],
                 usingTsd=opt['TSENet']['usingTsd'],
                 CNN10_settings=opt['TSENet']['CNN10_settings'],
                 fixCNN10=opt['TSENet']['fixCNN10'],
                 fixTSDNet=opt['TSENet']['fixTSDNet'],
                 pretrainedCNN10=opt['TSENet']['pretrainedCNN10'],
                 pretrainedTSDNet=opt['TSENet']['pretrainedTSDNet'],
                 threshold=opt['TSENet']['threshold'])
    else:
        net = TSENet(N=opt['TSENet']['N'],
                    B=opt['TSENet']['B'],
                    H=opt['TSENet']['H'],
                    P=opt['TSENet']['P'],
                    X=opt['TSENet']['X'],
                    R=opt['TSENet']['R'],
                    norm=opt['TSENet']['norm'],
                    num_spks=opt['TSENet']['num_spks'],
                    causal=opt['TSENet']['causal'],
                    cls_num=opt['TSENet']['class_num'],
                    nFrameLen=opt['datasets']['audio_setting']['nFrameLen'],
                    nFrameShift=opt['datasets']['audio_setting']['nFrameShift'],
                    nFFT=opt['datasets']['audio_setting']['nFFT'],
                    fusion=opt['TSENet']['fusion'],
                    usingEmb=opt['TSENet']['usingEmb'],
                    usingTsd=opt['TSENet']['usingTsd'],
                    CNN10_settings=opt['TSENet']['CNN10_settings'],
                    fixCNN10=opt['TSENet']['fixCNN10'],
                    fixTSDNet=opt['TSENet']['fixTSDNet'],
                    pretrainedCNN10=opt['TSENet']['pretrainedCNN10'],
                    pretrainedTSDNet=opt['TSENet']['pretrainedTSDNet'],
                    threshold=opt['TSENet']['threshold'])
    # build optimizer
    logger.info("Building the optimizer of TSENet")
    optimizer = make_optimizer(net.parameters(), opt)
    # build dataloader
    logger.info('Building the dataloader of TSENet')
    train_dataloader, val_dataloader, test_dataloader = make_dataloader(opt)

    logger.info('Train Datasets Length: {}, Val Datasets Length: {}, Test Datasets Length: {}'.format(
        len(train_dataloader), len(val_dataloader), len(test_dataloader)))

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, opt['train']['epoch'])
    # build trainer
    logger.info('Building the Trainer of TSENet')
    if opt['one_hot'] == 1:
        trainer = trainer_Tasnet_one_hot.Trainer(train_dataloader, val_dataloader, test_dataloader, net, optimizer, scheduler, opt)
    else:
        trainer = trainer_Tasnet.Trainer(train_dataloader, val_dataloader, test_dataloader, net, optimizer, scheduler, opt)
    trainer.run()
    #trainer.only_test()


if __name__ == "__main__":
    train()

