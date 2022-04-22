import sys
sys.path.append('../')

from utils.util import check_parameters
import time
import logging
from logger.set_logger import setup_logger
from model.loss import get_loss, get_loss_one_hot, get_loss_one_hot_focal, get_loss_one_hot_focal_sim
import torch
import os
import matplotlib.pyplot as plt
from torch.nn.parallel import data_parallel

class Trainer(object):
    def __init__(self, train_dataloader, val_dataloader, test_dataloader, Conv_Tasnet, optimizer, scheduler, opt):
        super(Trainer).__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.scheduler = scheduler
        self.num_spks = opt['num_spks']
        self.cur_epoch = 0
        self.total_epoch = opt['train']['epoch']
        self.early_stop = opt['train']['early_stop']
        self.opt = opt
        self.print_freq = opt['logger']['print_freq']
        # setup_logger(opt['logger']['name'], opt['logger']['path'],
        #             screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
        self.logger = logging.getLogger(opt['logger']['name'])
        self.checkpoint = opt['train']['path']
        self.name = opt['name']
        self.nFrameShift = opt['datasets']['audio_setting']['nFrameShift']
        self.audio_length = opt['datasets']['audio_setting']['audio_length']
        self.sr = opt['datasets']['audio_setting']['sample_rate']
        self.ratio = 0.3

        if opt['train']['gpuid']:
            self.logger.info('Load Nvida GPU .....')
            self.device = torch.device(
                'cuda:{}'.format(opt['train']['gpuid'][0]))
            self.gpuid = opt['train']['gpuid']
            self.convtasnet = Conv_Tasnet.to(self.device)
            self.logger.info(
                'Loading Conv-TasNet parameters: {:.3f} Mb'.format(check_parameters(self.convtasnet)))
        else:
            self.logger.info('Load CPU ...........')
            self.device = torch.device('cpu')
            self.convtasnet = Conv_Tasnet.to(self.device)
            self.logger.info(
                'Loading Conv-TasNet parameters: {:.3f} Mb'.format(check_parameters(self.convtasnet)))

        if opt['resume']['state']:
            ckp = torch.load(opt['resume']['path'], map_location='cpu')
            self.cur_epoch = ckp['epoch']
            self.logger.info("Resume from checkpoint {}: epoch {:.3f}".format(
                opt['resume']['path'], self.cur_epoch))
            self.convtasnet = Conv_Tasnet.load_state_dict(
                ckp['model_state_dict']).to(self.device)
            self.optimizer = optimizer.load_state_dict(ckp['optim_state_dict'])
        else:
            self.convtasnet = Conv_Tasnet.to(self.device)
            self.optimizer = optimizer

        if opt['optim']['clip_norm']:
            self.clip_norm = opt['optim']['clip_norm']
            self.logger.info(
                "Gradient clipping by {}, default L2".format(self.clip_norm))
        else:
            self.clip_norm = 0

    def train(self, epoch):
        self.logger.info(
            'Start training from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.convtasnet.train()
        num_batchs = len(self.train_dataloader)
        total_loss = 0.0
        total_loss_cls = 0.0
        total_loss_tsd = 0.0
        num_index = 1
        start_time = time.time()
        for mix, s1, ref, cls, onset, offset, tsd_lab,sim_lab,L_lab in self.train_dataloader:
            mix = mix.to(self.device)
            ref = ref.to(self.device)
            s1 = [s1.to(self.device) for i in range(self.num_spks)]
            cls = cls.to(self.device)
            cls_index = cls.argmax(1)
            onset = onset.to(self.device)
            offset = offset.to(self.device)
            tsd_lab = tsd_lab.to(self.device)
            sim_lab = sim_lab.to(self.device)
            L_lab = L_lab.to(self.device)
            self.optimizer.zero_grad()
            if self.gpuid:
                # model = torch.nn.DataParallel(self.convtasnet)
                # out = model(mix, ref)
                est_cls, est_tsd_time,est_tsd_time_up, sim_cos = self.convtasnet(mix, ref, cls_index.long())
            else:
                est_cls, est_tsd_time,est_tsd_time_up, sim_cos = self.convtasnet(mix, ref, cls_index.long())
            # l = Loss(out, s1)
            if self.opt['sim']:
                epoch_loss, loss_cls, loss_tsd = get_loss_one_hot_focal_sim(est_cls, cls, est_tsd, tsd_lab, sim_cos, sim_lab)
            elif self.opt['focal_loss']:
                epoch_loss, loss_cls, loss_tsd = get_loss_one_hot_focal(est_cls, cls, est_tsd, tsd_lab, sim_cos, sim_lab)
            else:
                epoch_loss, loss_cls, loss_tsd = get_loss_one_hot(est_cls, cls, est_tsd_time, L_lab, sim_cos, sim_lab)
            total_loss += epoch_loss.item()
            total_loss_cls += loss_cls.item()
            total_loss_tsd += loss_tsd.item()
            epoch_loss.backward()
            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.convtasnet.parameters(), self.clip_norm)

            self.optimizer.step()
            if num_index % self.print_freq == 0:
                message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, total_loss:{:.3f}, total_loss_cls:{:.3f}, total_loss_tsd:{:.3f}>'.format(
                    epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss / num_index,
                                                                            total_loss_cls / num_index,
                                                                            total_loss_tsd / num_index)
                self.logger.info(message)
            num_index += 1
        end_time = time.time()
        total_loss = total_loss / num_index
        total_loss_cls = total_loss_cls / num_index
        total_loss_tsd = total_loss_tsd / num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, loss_cls:{:.3f}, loss_tsd:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, total_loss_cls, total_loss_tsd, (end_time - start_time) / 60)
        self.logger.info(message)
        return total_loss, total_loss_cls, total_loss_tsd

    def validation(self, epoch):
        self.logger.info('Start Validation from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.convtasnet.eval()
        num_batchs = len(self.val_dataloader)
        num_index = 1
        total_loss = 0.0
        total_loss_cls = 0.0
        total_loss_tsd = 0.0
        start_time = time.time()
        with torch.no_grad():
            for mix, s1, ref, cls, onset, offset, tsd_lab, sim_lab, L_lab in self.val_dataloader:
                mix = mix.to(self.device)
                ref = ref.to(self.device)
                s1 = [s1.to(self.device) for i in range(self.num_spks)]
                cls = cls.to(self.device)
                cls_index = cls.argmax(1)
                onset = onset.to(self.device)
                offset = offset.to(self.device)
                tsd_lab = tsd_lab.to(self.device)
                sim_lab = sim_lab.to(self.device)
                L_lab = L_lab.to(self.device)
                self.optimizer.zero_grad()
                if self.gpuid:
                    #model = torch.nn.DataParallel(self.convtasnet)
                    #out = model(mix)
                    # out = torch.nn.parallel.data_parallel(self.convtasnet,mix,device_ids=self.gpuid)
                    est_cls, est_tsd_time, est_tsd_time_up, sim_cos = self.convtasnet(mix, ref, cls_index.long())
                else:
                    est_cls, est_tsd_time,est_tsd_time_up, sim_cos = self.convtasnet(mix, ref, cls_index.long())
                # l = Loss(out, s1)
                if self.opt['sim']:
                    epoch_loss, loss_cls, loss_tsd = get_loss_one_hot_focal_sim(est_cls, cls, est_tsd, tsd_lab, sim_cos, sim_lab)
                elif self.opt['focal_loss']:
                    epoch_loss, loss_cls, loss_tsd = get_loss_one_hot_focal(est_cls, cls, est_tsd_time, tsd_lab, sim_cos, sim_lab)
                else:
                    epoch_loss, loss_cls, loss_tsd = get_loss_one_hot(est_cls, cls, est_tsd_time, L_lab, sim_cos, sim_lab)
                #epoch_loss, loss_cls, loss_tsd = get_loss_one_hot_focal(est_cls, cls, est_tsd, tsd_lab, sim_cos, sim_lab)
                #epoch_loss, loss_cls, loss_tsd = get_loss_one_hot(est_cls, cls, est_tsd, tsd_lab, sim_cos, sim_lab)
                total_loss += epoch_loss.item()
                total_loss_cls += loss_cls.item()
                total_loss_tsd += loss_tsd.item()
                num_index += 1
        end_time = time.time()
        total_loss = total_loss / num_index
        total_loss_cls = total_loss_cls / num_index
        total_loss_tsd = total_loss_tsd / num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, loss_cls:{:.3f}, loss_tsd:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, total_loss_cls, total_loss_tsd, (end_time - start_time) / 60)
        self.logger.info(message)
        return total_loss, total_loss_cls, total_loss_tsd

    def test(self, epoch):
        self.logger.info(
            'Start Test from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.convtasnet.eval()
        num_batchs = len(self.test_dataloader)
        num_index = 1
        total_loss = 0.0
        total_loss_cls = 0.0
        total_loss_tsd = 0.0
        start_time = time.time()
        with torch.no_grad():
            for mix, s1, ref, cls, onset, offset, tsd_lab, sim_lab, L_lab in self.test_dataloader:
                mix = mix.to(self.device)
                ref = ref.to(self.device)
                s1 = [s1.to(self.device) for i in range(self.num_spks)]
                cls = cls.to(self.device)
                cls_index = cls.argmax(1)
                onset = onset.to(self.device)
                offset = offset.to(self.device)
                tsd_lab = tsd_lab.to(self.device)
                sim_lab = sim_lab.to(self.device)
                L_lab = L_lab.to(self.device)
                self.optimizer.zero_grad()
                if self.gpuid:
                    # model = torch.nn.DataParallel(self.convtasnet)
                    # out = model(mix)
                    # out = torch.nn.parallel.data_parallel(self.convtasnet, mix, device_ids=self.gpuid)
                    est_cls, est_tsd_time,est_tsd_time_up, sim_cos = self.convtasnet(mix, ref, cls_index.long())
                else:
                    est_cls, est_tsd_time,est_tsd_time_up, sim_cos = self.convtasnet(mix, ref, cls_index.long())
                if self.opt['sim']:
                    epoch_loss, loss_cls, loss_tsd = get_loss_one_hot_focal_sim(est_cls, cls, est_tsd, tsd_lab, sim_cos, sim_lab)
                elif self.opt['focal_loss']:
                    epoch_loss, loss_cls, loss_tsd = get_loss_one_hot_focal(est_cls, cls, est_tsd, tsd_lab, sim_cos, sim_lab)
                else:
                    epoch_loss, loss_cls, loss_tsd = get_loss_one_hot(est_cls, cls, est_tsd_time, L_lab, sim_cos, sim_lab)
                # l = Loss(out, s1)
                #epoch_loss, loss_cls, loss_tsd = get_loss_one_hot_focal(est_cls, cls, est_tsd, tsd_lab, sim_cos, sim_lab)
                #epoch_loss, loss_cls, loss_tsd = get_loss_one_hot(est_cls, cls, est_tsd, tsd_lab, sim_cos, sim_lab)
                total_loss += epoch_loss.item()
                total_loss_cls += loss_cls.item()
                total_loss_tsd += loss_tsd.item()
                num_index += 1
        end_time = time.time()
        total_loss = total_loss / num_index
        total_loss_cls = total_loss_cls / num_index
        total_loss_tsd = total_loss_tsd / num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, loss_cls:{:.3f}, loss_tsd:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, total_loss_cls, total_loss_tsd, (end_time - start_time) / 60)
        self.logger.info(message)
        return total_loss, total_loss_cls, total_loss_tsd


    def run(self):
        train_loss = []
        val_loss = []
        test_loss = []
        with torch.cuda.device(self.gpuid[0]):
            self.save_checkpoint(self.cur_epoch, best=False)
            v_loss,_,_ = self.validation(self.cur_epoch)
            best_loss = v_loss
            self.logger.info("Starting epoch from {:d}, loss = {:.4f}".format(self.cur_epoch, best_loss))
            no_improve = 0
            # starting training part
            while self.cur_epoch < self.total_epoch:
                self.cur_epoch += 1
                t_loss,_,_ = self.train(self.cur_epoch)
                v_loss,_,_ = self.validation(self.cur_epoch)
                tt_loss,_,_ = self.test(self.cur_epoch)

                train_loss.append(t_loss)
                val_loss.append(v_loss)
                test_loss.append(tt_loss)

                # schedule here
                self.scheduler.step()

                if v_loss >= best_loss:
                    no_improve += 1
                    self.logger.info(
                        'No improvement, Best Loss: {:.4f}'.format(best_loss))
                else:
                    best_loss = v_loss
                    no_improve = 0
                    self.save_checkpoint(self.cur_epoch, best=True)
                    self.logger.info('Epoch: {:d}, Now Best Loss Change: {:.4f}'.format(
                        self.cur_epoch, best_loss))
                    self.logger.info('Epoch: {:d}, Best Loss Test: {:.4f}'.format(
                        self.cur_epoch, tt_loss))

                if no_improve == self.early_stop:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(
                            no_improve))
                    break
            self.save_checkpoint(self.cur_epoch, best=False)
            self.logger.info("Training for {:d}/{:d} epoches done!".format(
                self.cur_epoch, self.total_epoch))

        # draw loss image
        plt.title("Loss of train, val and test")
        x = [i for i in range(self.cur_epoch)]
        plt.plot(x, train_loss, 'b-', label=u'train_loss', linewidth=0.8)
        plt.plot(x, val_loss, 'c-', label=u'val_loss', linewidth=0.8)
        plt.plot(x, test_loss, 'g', label=u'test_loss', linewidth=0.8)
        plt.legend()
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('loss.png')

    def save_checkpoint(self, epoch, best=True):
        '''
           save model
           best: the best model
        '''
        os.makedirs(os.path.join(self.checkpoint, self.name), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.convtasnet.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        },
            os.path.join(self.checkpoint, self.name, '{0}.pt'.format('best' if best else 'last')))

