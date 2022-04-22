import sys
sys.path.append('../')

from utils.util import check_parameters
import time
import logging
from model.loss import get_loss,get_loss_one_hot
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
def time_to_frame(tm,st=True):
    radio = 10.0/624
    if st:
        n_fame = tm//radio
    else:
        n_fame = math.ceil(tm/radio)

    if n_fame >= 624:
        n_fame = 623
    if n_fame < 0:
        n_fame = 0
    return n_fame
def get_mask(onset,offset):
    out_mask = np.zeros((onset.shape[0],257,624))
    for i in range(onset.shape[0]):
        st_frame = time_to_frame(onset[i])
        ed_frame = time_to_frame(offset[i])
        st_frame = st_frame.numpy()
        ed_frame = ed_frame.numpy()
        # print('st_t,ed_t ',onset[i],offset[i])
        # print('st_frame,ed_frame',st_frame,ed_frame)
        # assert 1==2 
        out_mask[i,:,int(st_frame):int(ed_frame)+1] = 1
    out_mask = torch.from_numpy(out_mask)
    return out_mask
class Trainer(object):
    def __init__(self, train_dataloader, val_dataloader, test_dataloader, TSENet, optimizer, scheduler, opt):
        super(Trainer).__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.scheduler = scheduler
        self.num_spks = opt['num_spks']
        self.cur_epoch = 0
        self.total_epoch = opt['train']['epoch']
        self.early_stop = opt['train']['early_stop']
        self.print_freq = opt['logger']['print_freq']
        self.logger = logging.getLogger(opt['logger']['name'])
        self.checkpoint = opt['train']['path'] # training path
        self.name = opt['name']
        self.nFrameShift = opt['datasets']['audio_setting']['nFrameShift'] # hop_length? 
        self.audio_length = opt['datasets']['audio_setting']['audio_length'] # 10
        self.sr = opt['datasets']['audio_setting']['sample_rate']
        self.weighting_ratio = opt['train']['weighting_ratio'] # 0.3
        self.metric_ratio = opt['train']['metirc_ratio'] # 0.5
        self.loss_type = opt['train']['loss'] # 15?
        if opt['train']['gpuid']:
            self.logger.info('Load Nvida GPU .....')
            self.device = torch.device(
                'cuda:{}'.format(opt['train']['gpuid'][0]))
            self.gpuid = opt['train']['gpuid']
            self.net = TSENet.to(self.device)
            self.logger.info(
                'Loading Conv-TasNet parameters: {:.3f} Mb'.format(check_parameters(self.net)))
        else:
            self.logger.info('Load CPU ...........')
            self.device = torch.device('cpu')
            self.net = TSENet.to(self.device)
            self.logger.info(
                'Loading Conv-TasNet parameters: {:.3f} Mb'.format(check_parameters(self.net)))

        if opt['resume']['state']: # load pre-train?
            ckp = torch.load(opt['resume']['path']+'/'+'best.pt', map_location='cpu')
            self.cur_epoch = ckp['epoch']
            self.logger.info("Resume from checkpoint {}: epoch {:.3f}".format(
                opt['resume']['path'], self.cur_epoch))
            self.net = TSENet.to(self.device)
            self.net.load_state_dict(ckp['model_state_dict'])
            self.optimizer = optimizer
            self.optimizer.load_state_dict(ckp['optim_state_dict'])
        else:
            self.net = TSENet.to(self.device)
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
        self.net.train()
        total_loss = 0.0
        total_loss_sisnr_all = 0.0
        total_loss_spec_all = 0.0
        total_loss_mse_all = 0.0
        total_sisnrI_all = 0.0
        total_loss_sisnr_w = 0.0
        total_loss_spec_w = 0.0
        total_loss_mse_w = 0.0
        total_sisnrI_w = 0.0
        total_loss_cls = 0.0
        num_index = 1
        start_time = time.time()
        for mix, s1, ref, cls, onset, offset, framelab in self.train_dataloader:
            mix = mix.to(self.device)
            ref = ref.to(self.device)
            s1 = [s1.to(self.device) for i in range(self.num_spks)]
            cls = cls.to(self.device)
            cls_index = cls.argmax(1)
            onset = onset.to(self.device)
            offset = offset.to(self.device)
            self.optimizer.zero_grad()
            out, lps, lab, est_cls = self.net(mix, ref, cls_index.long(), s1[0])
            epoch_loss, loss_sisnr_all, loss_spec_all, loss_mse_all, sisnrI_all, \
            loss_sisnr_w, loss_spec_w, loss_mse_w, sisnrI_w, \
            loss_cls = get_loss_one_hot(self.loss_type, out[0], s1[0], mix, lps, lab, est_cls,
                                          cls, onset, offset,
                                          self.nFrameShift, self.sr, self.audio_length, self.weighting_ratio)
            total_loss += epoch_loss.item()
            total_loss_sisnr_all += loss_sisnr_all.item()
            total_loss_mse_all += loss_mse_all.item()
            total_loss_spec_all += loss_spec_all.item()
            total_sisnrI_all += sisnrI_all.item()
            total_loss_sisnr_w += loss_sisnr_w.item()
            total_loss_mse_w += loss_mse_w.item()
            total_loss_spec_w += loss_spec_w.item()
            total_sisnrI_w += sisnrI_w.item()
            total_loss_cls += loss_cls.item()
            epoch_loss.backward()

            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.clip_norm)

            self.optimizer.step()
            if num_index % self.print_freq == 0:
                message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, total_loss:{:.6f}, ' \
                          'loss_sisnr:{:.6f}, loss_mse:{:.6f}, loss_spec:{:.6f}, sisnrI:{:.6f}, loss_sisnr_w:{:.6f}, loss_mse_w:{:.6f}, loss_spec_w:{:.6f}, sisnrI_w:{:.6f}, ' \
                          'loss_cls:{:.6f}>'.format(
                    epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss / num_index,
                                                                            total_loss_sisnr_all / num_index,
                                                                            total_loss_mse_all / num_index,
                                                                            total_loss_spec_all / num_index,
                                                                            total_sisnrI_all / num_index,
                                                                            total_loss_sisnr_w / num_index,
                                                                            total_loss_mse_w / num_index,
                                                                            total_loss_spec_w / num_index,
                                                                            total_sisnrI_w / num_index,
                                                                            total_loss_cls / num_index)
                self.logger.info(message)
            num_index += 1
        end_time = time.time()
        total_loss = total_loss / num_index
        total_loss_sisnr_all = total_loss_sisnr_all / num_index
        total_loss_mse_all = total_loss_mse_all / num_index
        total_loss_spec_all = total_loss_spec_all / num_index
        total_sisnrI_all = total_sisnrI_all / num_index
        total_loss_sisnr_w = total_loss_sisnr_w / num_index
        total_loss_mse_w = total_loss_mse_w / num_index
        total_loss_spec_w = total_loss_spec_w / num_index
        total_sisnrI_w = total_sisnrI_w / num_index
        total_loss_cls = total_loss_cls / num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.6f}, ' \
                  'loss_sisnr:{:.6f}, loss_mse:{:.6f}, loss_spec:{:.6f}, sisnrI:{:.6f}, loss_sisnr_w:{:.6f}, loss_mse_w:{:.6f}, loss_spec_w:{:.6f}, sisnrI_w:{:.6f}, ' \
                  'loss_cls:{:.6f}, Total time:{:.6f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, total_loss_sisnr_all, total_loss_mse_all,
            total_loss_spec_all, total_sisnrI_all, total_loss_sisnr_w, total_loss_mse_w,
            total_loss_spec_w, total_sisnrI_w, total_loss_cls, (end_time - start_time) / 60)
        self.logger.info(message)
        return total_loss, -total_loss_sisnr_all, -total_loss_sisnr_w, total_sisnrI_all, total_sisnrI_w

    def validation(self, epoch):
        self.logger.info(
            'Start Validation from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.net.eval()
        num_index = 1
        total_loss = 0.0
        total_loss_sisnr_all = 0.0
        total_sisnrI_all = 0.0
        total_loss_sisnr_w = 0.0
        total_sisnrI_w = 0.0
        start_time = time.time()
        with torch.no_grad():
            for mix, s1, ref, cls, onset, offset, framelab in self.val_dataloader:
                mix = mix.to(self.device)
                ref = ref.to(self.device)
                s1 = [s1.to(self.device) for i in range(self.num_spks)]
                cls = cls.to(self.device)
                cls_index = cls.argmax(1)
                onset = onset.to(self.device)
                offset = offset.to(self.device)

                out, lps, lab, est_cls = self.net(mix, ref, cls_index.long(), s1[0])

                epoch_loss, loss_sisnr_all, loss_spec_all, loss_mse_all, sisnrI_all, \
                loss_sisnr_w, loss_spec_w, loss_mse_w, sisnrI_w, \
                loss_cls = get_loss_one_hot(self.loss_type, out[0], s1[0], mix, lps, lab, est_cls,
                                              cls, onset, offset,
                                              self.nFrameShift, self.sr, self.audio_length, self.weighting_ratio)

                total_loss += epoch_loss.item()
                total_loss_sisnr_all += loss_sisnr_all.item()
                total_sisnrI_all += sisnrI_all.item()
                total_loss_sisnr_w += loss_sisnr_w.item()
                total_sisnrI_w += sisnrI_w.item()
                num_index += 1
        end_time = time.time()
        total_loss = total_loss / num_index
        total_loss_sisnr_all = total_loss_sisnr_all / num_index
        total_sisnrI_all = total_sisnrI_all / num_index
        total_loss_sisnr_w = total_loss_sisnr_w / num_index
        total_sisnrI_w = total_sisnrI_w / num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.6f}, ' \
                  'loss_sisnr:{:.6f}, sisnrI:{:.6f}, loss_sisnr_w:{:.6f}, sisnrI_w:{:.6f}' \
                  ', Total time:{:.6f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, total_loss_sisnr_all, total_sisnrI_all,
            total_loss_sisnr_w, total_sisnrI_w, (end_time - start_time) / 60)
        self.logger.info(message)
        return total_loss, -total_loss_sisnr_all, -total_loss_sisnr_w, total_sisnrI_all, total_sisnrI_w

    def test(self, epoch):
        self.logger.info(
            'Start Test from epoch: {:d}, iter: {:d}'.format(epoch, 0))
        self.net.eval()
        num_index = 1
        total_loss = 0.0
        total_loss_sisnr_all = 0.0
        total_sisnrI_all = 0.0
        total_loss_sisnr_w = 0.0
        total_sisnrI_w = 0.0
        start_time = time.time()
        with torch.no_grad():
            for mix, s1, ref, cls, onset, offset, framelab in self.test_dataloader:
                mix = mix.to(self.device)
                ref = ref.to(self.device)
                s1 = [s1.to(self.device) for i in range(self.num_spks)]
                cls = cls.to(self.device)
                cls_index = cls.argmax(1)
                out_mask = get_mask(onset, offset)
                out_mask = out_mask.to(self.device).float()
                onset = onset.to(self.device)
                offset = offset.to(self.device)
                framelab = framelab.to(self.device)
                out, lps, lab, est_cls = self.net(mix, ref, cls_index.long(), s1[0], out_mask)

                epoch_loss, loss_sisnr_all, loss_spec_all, loss_mse_all, sisnrI_all, \
                loss_sisnr_w, loss_spec_w, loss_mse_w, sisnrI_w, \
                loss_cls = get_loss_one_hot(self.loss_type, out[0], s1[0], mix, lps, lab, est_cls,
                                              cls, onset, offset,
                                              self.nFrameShift, self.sr, self.audio_length, self.weighting_ratio)

                total_loss += epoch_loss.item()
                total_loss_sisnr_all += loss_sisnr_all.item()
                total_sisnrI_all += sisnrI_all.item()
                total_loss_sisnr_w += loss_sisnr_w.item()
                total_sisnrI_w += sisnrI_w.item()
                num_index += 1
        end_time = time.time()
        total_loss = total_loss / num_index
        total_loss_sisnr_all = total_loss_sisnr_all / num_index
        total_sisnrI_all = total_sisnrI_all / num_index
        total_loss_sisnr_w = total_loss_sisnr_w / num_index
        total_sisnrI_w = total_sisnrI_w / num_index
        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.6f}, ' \
                  'loss_sisnr:{:.6f}, sisnrI:{:.6f}, loss_sisnr_w:{:.6f}, sisnrI_w:{:.6f}, ' \
                  'Total time:{:.6f} min> '.format(
            epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss, total_loss_sisnr_all, total_sisnrI_all,
            total_loss_sisnr_w, total_sisnrI_w, (end_time - start_time) / 60)
        self.logger.info(message)
        return total_loss, -total_loss_sisnr_all, -total_loss_sisnr_w, total_sisnrI_all, total_sisnrI_w

    def only_test(self):
        tt_loss, tt_sisnr, tt_sisnr_w, tt_sisnrI, tt_sisnrI_w = self.validation(0)

    def run(self):
        train_loss = []
        val_loss = []
        test_loss = []
        train_sisnrI = [] # sisnrI ?
        val_sisnrI = []
        test_sisnrI = []
        train_sisnr = []
        val_sisnr = []
        test_sisnr = []
        train_sisnrI_w = []
        val_sisnrI_w= []
        test_sisnrI_w = []
        train_sisnr_w = []
        val_sisnr_w = []
        test_sisnr_w = []
        train_metric = []
        val_metric = []
        test_metric = []

        with torch.cuda.device(self.gpuid[0]):
            self.save_checkpoint(self.cur_epoch, best=False)
            v_loss,v_sisnr,v_sisnr_w,v_sisnrI,v_sisnrI_w= self.validation(self.cur_epoch)
            best_loss = v_loss
            best_sisnrI = v_sisnrI
            best_sisnr = v_sisnr
            best_sisnrI_w = v_sisnrI_w
            best_sisnr_w = v_sisnr_w
            best_metric = self.metric_ratio * best_sisnrI_w + (1. - self.metric_ratio) * best_sisnrI
            self.logger.info("Starting epoch from {:d}, metric = {:.4f}, loss = {:.4f}, sisnrI = {:.4f}, sisnr = {:.4f}, sisnrI_w = {:.4f}, sisnr_w = {:.4f}".format(self.cur_epoch, best_metric, best_loss, best_sisnrI, best_sisnr, best_sisnrI_w, best_sisnr_w))
            no_improve = 0
            # starting training part
            while self.cur_epoch < self.total_epoch:
                self.cur_epoch += 1
                t_loss, t_sisnr, t_sisnr_w, t_sisnrI, t_sisnrI_w = self.train(self.cur_epoch)
                v_loss, v_sisnr, v_sisnr_w, v_sisnrI, v_sisnrI_w = self.validation(self.cur_epoch)
                tt_loss, tt_sisnr, tt_sisnr_w, tt_sisnrI, tt_sisnrI_w = self.test(self.cur_epoch)

                t_metric = self.metric_ratio * t_sisnrI_w + (1. - self.metric_ratio) * t_sisnrI
                v_metric = self.metric_ratio * v_sisnrI_w + (1. - self.metric_ratio) * v_sisnrI
                tt_metric = self.metric_ratio * tt_sisnrI_w + (1. - self.metric_ratio) * tt_sisnrI

                train_metric.append(t_metric)
                val_metric.append(v_metric)
                test_metric.append(tt_metric)
                train_loss.append(t_loss)
                val_loss.append(v_loss)
                test_loss.append(tt_loss)
                train_sisnrI.append(t_sisnrI)
                val_sisnrI.append(v_sisnrI)
                test_sisnrI.append(tt_sisnrI)
                train_sisnr.append(t_sisnr)
                val_sisnr.append(v_sisnr)
                test_sisnr.append(tt_sisnr)
                train_sisnrI_w.append(t_sisnrI_w)
                val_sisnrI_w.append(v_sisnrI_w)
                test_sisnrI_w.append(tt_sisnrI_w)
                train_sisnr_w.append(t_sisnr_w)
                val_sisnr_w.append(v_sisnr_w)
                test_sisnr_w.append(tt_sisnr_w)

                # schedule here
                self.scheduler.step()

                if v_metric <= best_metric:
                    no_improve += 1
                    self.logger.info(
                        'No improvement, Best metric: {:.4f}, sisnrI = {:.4f}, sisnr = {:.4f}, sisnrI_w = {:.4f}, sisnr_w = {:.4f}'.format(best_metric, best_sisnrI, best_sisnr, best_sisnrI_w, best_sisnr_w))
                else:
                    best_loss = v_loss
                    best_metric = v_metric
                    best_sisnrI = v_sisnrI
                    best_sisnr = v_sisnr
                    best_sisnrI_w = v_sisnrI_w
                    best_sisnr_w = v_sisnr_w
                    no_improve = 0
                    self.save_checkpoint(self.cur_epoch, best=True)
                    self.logger.info('Epoch: {:d}, Now Best Metric Change: {:.4f}, sisnrI: {:.4f}, sisnr = {:.4f}, sisnrI_w = {:.4f}, sisnr_w = {:.4f}'.format(
                        self.cur_epoch, best_metric, best_sisnrI, best_sisnr, best_sisnrI_w, best_sisnr_w))
                    self.logger.info('Epoch: {:d}, Best Metirc Test: {:.4f}, sisnrI: {:.4f}, sisnr = {:.4f}, sisnrI_w: {:.4f}, sisnr_w = {:.4f}'.format(
                        self.cur_epoch, tt_metric, tt_sisnrI, tt_sisnr, tt_sisnrI_w, tt_sisnr_w))

                if no_improve == self.early_stop:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(no_improve))
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

        plt.plot(x, train_sisnrI, 'b-', label=u'train_sisnrI', linewidth=0.8)
        plt.plot(x, val_sisnrI, 'c-', label=u'val_sisnrI', linewidth=0.8)
        plt.plot(x, test_sisnrI, 'g', label=u'test_sisnrI', linewidth=0.8)
        plt.legend()
        plt.ylabel('sisnrI')
        plt.xlabel('epoch')
        plt.savefig('sisnrI.png')


        plt.plot(x, train_sisnr, 'b-', label=u'train_sisnr', linewidth=0.8)
        plt.plot(x, val_sisnr, 'c-', label=u'val_sisnr', linewidth=0.8)
        plt.plot(x, test_sisnr, 'g', label=u'test_sisnr', linewidth=0.8)
        plt.legend()
        plt.ylabel('sisnr')
        plt.xlabel('epoch')
        plt.savefig('sisnr.png')

        plt.plot(x, train_sisnrI_w, 'b-', label=u'train_sisnrI_w', linewidth=0.8)
        plt.plot(x, val_sisnrI_w, 'c-', label=u'val_sisnrI_w', linewidth=0.8)
        plt.plot(x, test_sisnrI_w, 'g', label=u'test_sisnrI_w', linewidth=0.8)
        plt.legend()
        plt.ylabel('sisnrI_w')
        plt.xlabel('epoch')
        plt.savefig('sisnrI_w.png')


        plt.plot(x, train_sisnr_w, 'b-', label=u'train_sisnr_w', linewidth=0.8)
        plt.plot(x, val_sisnr_w, 'c-', label=u'val_sisnr_w', linewidth=0.8)
        plt.plot(x, test_sisnr_w, 'g', label=u'test_sisnr_w', linewidth=0.8)
        plt.legend()
        plt.ylabel('sisnr_w')
        plt.xlabel('epoch')
        plt.savefig('sisnr_w.png')

        plt.plot(x, train_metric, 'b-', label=u'train_metric', linewidth=0.8)
        plt.plot(x, val_metric, 'c-', label=u'val_metric', linewidth=0.8)
        plt.plot(x, test_metric, 'g', label=u'test_metric', linewidth=0.8)
        plt.legend()
        plt.ylabel('metric')
        plt.xlabel('epoch')
        plt.savefig('metric.png')

    def save_checkpoint(self, epoch, best=True):
        '''
           save model
           best: the best model
        '''
        os.makedirs(os.path.join(self.checkpoint, self.name), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        },
            os.path.join(self.checkpoint, self.name, '{0}.pt'.format('best' if best else 'last')))




