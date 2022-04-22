import torch
from itertools import permutations
import numpy as np
from pypesq import pesq
import torch.nn as nn
class FocalLoss(nn.Module):
    def __init__(self,alpha=0.35, gamma=2):
        super(FocalLoss,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-9
        #self.bce = BCELoss()
    def forward(self,predict,target,frame_level_time=None):
        # print()
        tmp = self.alpha*(1-predict)**self.gamma # t = 1
        # print('predict ',predict.shape)
        # print('target ',target.shape)
        # assert 1==2
        # tmp = self.alpha
        # print('tmp ',tmp.shape)
        tmp2 = (1-self.alpha)*(predict)**self.gamma # t=0
        # tmp2 = (1-self.alpha)
        # print('tmp2 ',tmp2.shape)
        # print(torch.log(predict).shape)
        # print('target ',target.shape)
        # assert 1==2
        loss = (-target*tmp*torch.log(predict+self.eps)).mean() + (-(1-target)*tmp2*torch.log(1-predict+self.eps)).mean()
        #loss = (-target*torch.log(predict)).mean() + (-(1-target)*torch.log(1-predict)).mean()
        #loss_bce = (-target*torch.log(predict)).mean() + (-(1-target)*torch.log(1-predict)).mean()
        # print('loss_bce ',loss_bce)
        # loss_function = torch.nn.BCELoss()
        # loss = loss_function(predict, target.squeeze())
        # print(self.bce(predict,target))
        # print('loss ',loss)
        # assert 1==2
        return loss

def nll_loss(output, target):
    '''Negative likelihood loss. The output should be obtained using F.log_softmax(x).

    Args:
      output: (N, classes_num)
      target: (N, classes_num)
    '''
    loss = - torch.mean(target * output)
    return loss

def tsd_loss(output, target):
    '''BCE loss.
    Args:
      output: (N)
      target: (N)
    '''
    loss_function = torch.nn.BCELoss()
    loss = loss_function(output, target.squeeze())
    return loss

def sisnr_loss(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """
    if x.shape != s.shape:
        if x.shape[-1] > s.shape[-1]:
            x = x[:, :s.shape[-1]]
        else:
            s = s[:, :x.shape[-1]]
    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)
    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    loss = -20. * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))
    return torch.sum(loss) / x.shape[0]


def sisnri(x, s, m):
    """
    Arguments:
    x: separated signal, BS x S
    s: reference signal, BS x S
    m: mixture signal, BS x S
    Return:
    sisnri: N tensor
    """
    sisnr = sisnr_loss(x, s)
    sisnr_ori = sisnr_loss(m, s)
    return sisnr_ori - sisnr

def lfb_mse_loss(x, s):
    """
    est_spec, ref_spec: BS x F x T
    return: log fbank MSE: BS tensor
    """
    if x.shape != s.shape:
        if x.shape[-1] > s.shape[-1]:
            x = x[:, :, :s.shape[-1]]
        else:
            s = s[:, :, :x.shape[-1]]
    t = torch.sum((x - s) ** 2)/(x.shape[0]*x.shape[1]*x.shape[2])
    return t

def mse_loss(x, s):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          return: N tensor
    """
    if x.shape != s.shape:
        if x.shape[-1] > s.shape[-1]:
            x = x[:, :s.shape[-1]]
        else:
            s = s[:, :x.shape[-1]]

    t = torch.sum((x - s) ** 2)/(x.shape[0]*x.shape[1])
    return t

def get_pesq(est_wav, lab_wav):
    num = est_wav.shape[0]
    score = 0.0
    for i in range(num):
        score += pesq(est_wav[i].cpu().detach(), lab_wav[i].cpu().detach(), 16000)
    score = score / num
    return score

def get_loss(est_cls, lab_cls, est_tsd, lab_tsd):
    loss_cls = nll_loss(est_cls, lab_cls)
    loss_tsd = tsd_loss(est_tsd, lab_tsd)
    # print('loss_cls ',loss_cls)
    # print('loss_tsd ',loss_tsd)
    # assert 1==2
    loss = loss_cls * 10. + loss_tsd
    return loss, loss_cls, loss_tsd


def get_loss_one_hot(est_cls, lab_cls, est_tsd, lab_tsd, sim_cos=None, sim_lab=None):
    # loss_cls = nll_loss(est_cls, lab_cls)
    # print('est_tsd ', est_tsd.shape)
    # print('lab_tsd ',lab_tsd.shape)
    # assert 1==2
    loss_tsd = tsd_loss(est_tsd, lab_tsd)
    # if sim_cos !=None:
    #     loss_cls = mse_loss(sim_cos,sim_lab)
    # else:
    #     loss_cls = loss_tsd
    loss_cls = loss_tsd
    # print('loss_cls ',loss_cls)
    # print('loss_tsd ',loss_tsd)
    # assert 1==2
    loss = loss_tsd 
    return loss, loss_cls, loss_tsd

def get_loss_one_hot_reg(est,lab):
    # loss_cls = nll_loss(est_cls, lab_cls)
    loss_tsd = mse_loss(est,lab)
    # if sim_cos !=None:
    #     loss_cls = mse_loss(sim_cos,sim_lab)
    # else:
    #     loss_cls = loss_tsd
    loss_cls = loss_tsd
    # print('loss_cls ',loss_cls)
    # print('loss_tsd ',loss_tsd)
    # assert 1==2
    loss = loss_tsd 
    return loss, loss_cls, loss_tsd

def get_loss_one_hot_reg_two(st, ed, lab):
    crossentropyloss = nn.CrossEntropyLoss()
    # loss_cls = nll_loss(est_cls, lab_cls)
    loss_st = crossentropyloss(st,lab[:,0].long())
    loss_ed = crossentropyloss(ed,lab[:,1].long())
    # if sim_cos !=None:
    #     loss_cls = mse_loss(sim_cos,sim_lab)
    # else:
    #     loss_cls = loss_tsd
    # print('loss_cls ',loss_cls)
    # print('loss_tsd ',loss_tsd)
    # assert 1==2
    loss = loss_st + loss_st
    return loss, loss_st, loss_st

def get_loss_one_hot_focal(est_cls, lab_cls, est_tsd, lab_tsd, sim_cos=None, sim_lab=None):
    # loss_cls = nll_loss(est_cls, lab_cls)
    focalLoss = FocalLoss()
    loss_tsd = focalLoss(est_tsd, lab_tsd.squeeze())
    # print('loss_tsd ',loss_tsd)
    # assert 1==2
    # loss_cls = loss_tsd
    # if sim_cos !=None:
    #     loss_cls = mse_loss(sim_cos,sim_lab)
    # else:
    #     loss_cls = loss_tsd
    loss_cls = loss_tsd
    # print('loss_cls ',loss_cls)
    # print('loss_tsd ',loss_tsd)
    # assert 1==2
    loss = loss_tsd
    # assert 1==2
    return loss, loss_cls, loss_tsd

def get_loss_one_hot_focal_sim(est_cls, lab_cls, est_tsd, lab_tsd, sim_cos=None, sim_lab=None):
    # loss_cls = nll_loss(est_cls, lab_cls)
    focalLoss = FocalLoss()
    loss_tsd = focalLoss(est_tsd, lab_tsd.squeeze())
    # print('loss_tsd ',loss_tsd)
    # assert 1==2
    # loss_cls = loss_tsd
    if sim_cos !=None:
        loss_cls = mse_loss(sim_cos,sim_lab)
    else:
        loss_cls = loss_tsd
    # print('loss_cls ',loss_cls)
    # print('loss_tsd ',loss_tsd)
    # assert 1==2
    loss = 2*loss_tsd + loss_cls
    # assert 1==2
    return loss, loss_cls, loss_tsd
