import torch
import numpy as np

def nll_loss(output, target):
    '''Negative likelihood loss. The output should be obtained using F.log_softmax(x).
    Args:
      output: (N, classes_num)
      target: (N, classes_num)
    '''
    loss = - torch.mean(target * output)
    return loss

def sisnr_loss(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor, estimate value
          s: reference signal, N x S tensor, True value
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


def sisnri(x, s, m): # sisnr improvement
    """
    Arguments:
    x: separated signal, BS x S predicted sound  
    s: reference signal, BS x S target sound
    m: mixture signal, BS x S mixture sound
    Return: 
    sisnri: N tensor
    """
    sisnr = sisnr_loss(x, s)
    sisnr_ori = sisnr_loss(m, s)
    return sisnr_ori - sisnr # 

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

def get_loss(loss_type, est_wav, lab_wav, mix_wav, est_mask, lab_mask, est_cls, lab_cls, onset, offset, nFrameShift, sr, audio_length, ratio):
    """
    loss type:
    1: enrollment: spec mse loss
    2: enrollment: wave mse loss
    3: enrollment: wave sisnrI loss
    4: enrollment: spec mse loss + wave mse loss
    5: enrollment: spec mse loss + wave sisnrI loss
    6: enrollment: wave mse loss + wave sisnrI loss
    7: enrollment: spec mse loss + wave mse loss + wave sisnrI loss
    8: enrollment: spec mse loss (w)
    9: enrollment: wave mse loss (w)
    10: enrollment: wave sisnrI loss (w)
    11: enrollment: spec mse loss (w) + wave mse loss (w)
    12: enrollment: spec mse loss (w) + wave sisnrI loss (w)
    13: enrollment: wave mse loss (w) + wave sisnrI loss (w)
    14: enrollment: spec mse loss (w) + wave mse loss (w) + wave sisnrI loss (w)
    15: enrollment: spec mse loss + wave mse loss + wave sisnrI loss + cls1 loss
    16: enrollment: spec mse loss (w) + wave mse loss (w) + wave sisnrI loss (w) + cls1 loss
    """
    loss_sisnr_w = 0.0
    loss_mse_w = 0.0
    loss_spec_w = 0.0
    sisnrI_w = 0.0
    onset = onset.cpu().numpy()
    offset = offset.cpu().numpy()
    sample_num = onset.shape[0] # batch_size
    for i in range(sample_num):
        assert onset[i] < offset[i]
        max_wav = audio_length * sr - 1
        # print('max_wav ',max_wav)
        max_frame = sr * audio_length // nFrameShift - 2
        # print('max_frame ',max_frame)
        onset_wav = round(sr * onset[i]) if round(sr * onset[i]) >= 0 else 0 # target sound begin sample
        # print('onset[i], onset_wav ',onset[i],onset_wav)
        offset_wav = round(sr * offset[i]) if round(sr * offset[i]) < max_wav else max_wav # end
        # print('offset[i], offset_wav ',offset[i],offset_wav)
        onset_frame = round(onset[i] * (sr // nFrameShift - 1)) if round(onset[i] * (sr // nFrameShift - 1)) >= 0 else 0
        # print('onset_frame ',onset_frame)
        offset_frame = round(offset[i] * (sr // nFrameShift - 1)) if round(offset[i] * (sr // nFrameShift - 1)) < max_frame else max_frame
        # print('offset_frame ',offset_frame)
        est_wav_w = est_wav[i, onset_wav:offset_wav] # est_wav
        est_wav_w = est_wav_w[None, :] # (1,N)
        lab_wav_w = lab_wav[i, onset_wav:offset_wav] # lab_wav
        lab_wav_w = lab_wav_w[None, :]
        est_mask_w = est_mask[i, :, onset_frame:offset_frame]
        est_mask_w = est_mask_w[None, :]
        lab_mask_w = lab_mask[i, :, onset_frame:offset_frame]
        lab_mask_w = lab_mask_w[None, :]
        loss_sisnr_w += sisnr_loss(est_wav_w, lab_wav_w) # weighted sisnr
        # print('loss_sisnr_w ',loss_sisnr_w)
        loss_mse_w += mse_loss(est_wav_w, lab_wav_w) # weighted mse
        loss_spec_w += lfb_mse_loss(est_mask_w, lab_mask_w) # mask loss
        # assert loss_mse_w is nan
        # print('loss_mse_w ',loss_mse_w)
        # print('loss_spec_w ',loss_spec_w)
        # assert 1==2
        mix_wav_w = mix_wav[i, onset_wav:offset_wav] # mix wav
        mix_wav_w = mix_wav_w[None, :] 
        sisnrI_w += sisnri(est_wav_w, lab_wav_w, mix_wav_w) # inmprovemnt

    loss_sisnr_w = loss_sisnr_w / sample_num
    loss_mse_w = loss_mse_w / sample_num
    loss_spec_w = loss_spec_w / sample_num
    sisnrI_w = sisnrI_w / sample_num
    loss_sisnr_all = sisnr_loss(est_wav, lab_wav) # 整个音频的loss
    loss_mse_all = mse_loss(est_wav, lab_wav)
    # print(est_wav[0])
    # print(lab_wav[0])
    # assert 1==2
    # print('loss_mse_all ',loss_mse_all)
    # assert 1==2
    loss_spec_all = lfb_mse_loss(est_mask, lab_mask)
    sisnrI_all = sisnri(est_wav, lab_wav, mix_wav)
    loss_cls = nll_loss(est_cls, lab_cls) # 分类损失
    # loss_emb = torch.cosine_similarity(emb, emb2, dim=-1)
    # loss_emb = 1.-torch.mean(loss_emb)
    if loss_type == 1:
        loss = loss_spec_all * 100.
    elif loss_type == 2:
        loss = loss_mse_all * 1000.
    elif loss_type == 3:
        loss = - sisnrI_all
    elif loss_type == 4:
        loss = loss_spec_all * 100. + loss_mse_all * 1000.
    elif loss_type == 5:
        loss = loss_spec_all * 100. - sisnrI_all
    elif loss_type == 6:
        loss = loss_mse_all * 1000. - sisnrI_all
    elif loss_type == 7:
        loss = loss_spec_all * 100. + loss_mse_all * 1000. - sisnrI_all
    elif loss_type == 8:
        loss = (loss_spec_all + ratio * loss_spec_w) * 100.
    elif loss_type == 9:
        loss = (loss_mse_all + ratio * loss_mse_w) * 1000.
    elif loss_type == 10:
        loss = - sisnrI_all - ratio * sisnrI_w
    elif loss_type == 11:
        loss = (loss_spec_all + ratio * loss_spec_w) * 100. + (loss_mse_all + ratio * loss_mse_w) * 1000.
    elif loss_type == 12:
        loss = (loss_spec_all + ratio * loss_spec_w) * 100. - sisnrI_all - ratio * sisnrI_w
    elif loss_type == 13:
        loss = (loss_mse_all + ratio * loss_mse_w) * 1000. - sisnrI_all - ratio * sisnrI_w
    elif loss_type == 14:
        loss = (loss_spec_all + ratio * loss_spec_w) * 100. + (loss_mse_all + ratio * loss_mse_w) * 1000. - sisnrI_all - ratio * sisnrI_w
    elif loss_type == 15:
        loss = loss_spec_all * 100. + loss_mse_all * 1000. - sisnrI_all + loss_cls * 1000.
    elif loss_type == 16:
        loss = (loss_spec_all + ratio * loss_spec_w) * 100. + (loss_mse_all + ratio * loss_mse_w) * 1000. - sisnrI_all - ratio * sisnrI_w + loss_cls * 1000.
    elif loss_type == 17:
        loss = (loss_spec_all + ratio * loss_spec_w) * 10. + (loss_mse_all + ratio * loss_mse_w) * 100000. - sisnrI_all - ratio * sisnrI_w + loss_cls * 100.
    # print('loss_spec_all ',loss_spec_all)
    # print('loss_spec_w ',loss_spec_w)
    # print('loss_mse_all ',loss_mse_all)
    # print('loss_mse_w ',loss_mse_w)
    # print('sisnrI_all ',sisnrI_all)
    # print('sisnrI_w ',sisnrI_w)
    # print('loss_cls ',loss_cls)
    # assert 1==2
    return loss, loss_sisnr_all, loss_spec_all, loss_mse_all, sisnrI_all, loss_sisnr_w, loss_spec_w, loss_mse_w, sisnrI_w, loss_cls


def get_loss_one_hot(loss_type, est_wav, lab_wav, mix_wav, est_mask, lab_mask, est_cls, lab_cls, onset, offset, nFrameShift, sr, audio_length, ratio):
    """
    loss type:
    1: enrollment: spec mse loss
    2: enrollment: wave mse loss
    3: enrollment: wave sisnrI loss
    4: enrollment: spec mse loss + wave mse loss
    5: enrollment: spec mse loss + wave sisnrI loss
    6: enrollment: wave mse loss + wave sisnrI loss
    7: enrollment: spec mse loss + wave mse loss + wave sisnrI loss
    8: enrollment: spec mse loss (w)
    9: enrollment: wave mse loss (w)
    10: enrollment: wave sisnrI loss (w)
    11: enrollment: spec mse loss (w) + wave mse loss (w)
    12: enrollment: spec mse loss (w) + wave sisnrI loss (w)
    13: enrollment: wave mse loss (w) + wave sisnrI loss (w)
    14: enrollment: spec mse loss (w) + wave mse loss (w) + wave sisnrI loss (w)
    15: enrollment: spec mse loss + wave mse loss + wave sisnrI loss + cls1 loss
    16: enrollment: spec mse loss (w) + wave mse loss (w) + wave sisnrI loss (w) + cls1 loss
    """
    loss_sisnr_w = 0.0
    loss_mse_w = 0.0
    loss_spec_w = 0.0
    sisnrI_w = 0.0
    onset = onset.cpu().numpy()
    offset = offset.cpu().numpy()
    sample_num = onset.shape[0] # batch_size
    for i in range(sample_num):
        assert onset[i] < offset[i]
        max_wav = audio_length * sr - 1
        # print('max_wav ',max_wav)
        max_frame = sr * audio_length // nFrameShift - 2
        # print('max_frame ',max_frame)
        onset_wav = round(sr * onset[i]) if round(sr * onset[i]) >= 0 else 0 # target sound begin sample
        # print('onset[i], onset_wav ',onset[i],onset_wav)
        offset_wav = round(sr * offset[i]) if round(sr * offset[i]) < max_wav else max_wav # end
        # print('offset[i], offset_wav ',offset[i],offset_wav)
        onset_frame = round(onset[i] * (sr // nFrameShift - 1)) if round(onset[i] * (sr // nFrameShift - 1)) >= 0 else 0
        # print('onset_frame ',onset_frame)
        offset_frame = round(offset[i] * (sr // nFrameShift - 1)) if round(offset[i] * (sr // nFrameShift - 1)) < max_frame else max_frame
        # print('offset_frame ',offset_frame)
        est_wav_w = est_wav[i, onset_wav:offset_wav] # est_wav
        est_wav_w = est_wav_w[None, :] # (1,N)
        lab_wav_w = lab_wav[i, onset_wav:offset_wav] # lab_wav
        lab_wav_w = lab_wav_w[None, :]
        est_mask_w = est_mask[i, :, onset_frame:offset_frame]
        est_mask_w = est_mask_w[None, :]
        lab_mask_w = lab_mask[i, :, onset_frame:offset_frame]
        lab_mask_w = lab_mask_w[None, :]
        loss_sisnr_w += sisnr_loss(est_wav_w, lab_wav_w) # weighted sisnr
        # print('loss_sisnr_w ',loss_sisnr_w)
        loss_mse_w += mse_loss(est_wav_w, lab_wav_w) # weighted mse
        loss_spec_w += lfb_mse_loss(est_mask_w, lab_mask_w) # mask loss
        # assert loss_mse_w is nan
        # print('loss_mse_w ',loss_mse_w)
        # print('loss_spec_w ',loss_spec_w)
        # assert 1==2
        mix_wav_w = mix_wav[i, onset_wav:offset_wav] # mix wav
        mix_wav_w = mix_wav_w[None, :] 
        sisnrI_w += sisnri(est_wav_w, lab_wav_w, mix_wav_w) # inmprovemnt

    loss_sisnr_w = loss_sisnr_w / sample_num
    loss_mse_w = loss_mse_w / sample_num
    loss_spec_w = loss_spec_w / sample_num
    sisnrI_w = sisnrI_w / sample_num
    loss_sisnr_all = sisnr_loss(est_wav, lab_wav) # 整个音频的loss
    loss_mse_all = mse_loss(est_wav, lab_wav)
    # print(est_wav[0])
    # print(lab_wav[0])
    # assert 1==2
    # print('loss_mse_all ',loss_mse_all)
    # assert 1==2
    loss_spec_all = lfb_mse_loss(est_mask, lab_mask)
    sisnrI_all = sisnri(est_wav, lab_wav, mix_wav)
    # loss_cls = nll_loss(est_cls, lab_cls) # 分类损失
    loss_cls = loss_spec_all
    # loss_emb = torch.cosine_similarity(emb, emb2, dim=-1)
    # loss_emb = 1.-torch.mean(loss_emb)
    if loss_type == 1:
        loss = loss_spec_all * 100.
    elif loss_type == 2:
        loss = loss_mse_all * 1000.
    elif loss_type == 3:
        loss = - sisnrI_all
    elif loss_type == 4:
        loss = loss_spec_all * 100. + loss_mse_all * 1000.
    elif loss_type == 5:
        loss = loss_spec_all * 100. - sisnrI_all
    elif loss_type == 6:
        loss = loss_mse_all * 1000. - sisnrI_all
    elif loss_type == 7:
        loss = loss_spec_all * 100. + loss_mse_all * 1000. - sisnrI_all
    elif loss_type == 8:
        loss = (loss_spec_all + ratio * loss_spec_w) * 100.
    elif loss_type == 9:
        loss = (loss_mse_all + ratio * loss_mse_w) * 1000.
    elif loss_type == 10:
        loss = - sisnrI_all - ratio * sisnrI_w
    elif loss_type == 11:
        loss = (loss_spec_all + ratio * loss_spec_w) * 100. + (loss_mse_all + ratio * loss_mse_w) * 1000.
    elif loss_type == 12:
        loss = (loss_spec_all + ratio * loss_spec_w) * 100. - sisnrI_all - ratio * sisnrI_w
    elif loss_type == 13:
        loss = (loss_mse_all + ratio * loss_mse_w) * 1000. - sisnrI_all - ratio * sisnrI_w
    elif loss_type == 14:
        loss = (loss_spec_all + ratio * loss_spec_w) * 100. + (loss_mse_all + ratio * loss_mse_w) * 1000. - sisnrI_all - ratio * sisnrI_w
    elif loss_type == 15:
        loss = loss_spec_all * 100. + loss_mse_all * 1000. - sisnrI_all 
    elif loss_type == 16:
        loss = (loss_spec_all + ratio * loss_spec_w) * 100. + (loss_mse_all + ratio * loss_mse_w) * 1000. - sisnrI_all - ratio * sisnrI_w 
    elif loss_type == 17:
        loss = (loss_spec_all + ratio * loss_spec_w) * 10. + (loss_mse_all + ratio * loss_mse_w) * 100000. - sisnrI_all - ratio * sisnrI_w
    # print('loss_spec_all ',loss_spec_all)
    # print('loss_spec_w ',loss_spec_w)
    # print('loss_mse_all ',loss_mse_all)
    # print('loss_mse_w ',loss_mse_w)
    # print('sisnrI_all ',sisnrI_all)
    # print('sisnrI_w ',sisnrI_w)
    # print('loss_cls ',loss_cls)
    # assert 1==2
    return loss, loss_sisnr_all, loss_spec_all, loss_mse_all, sisnrI_all, loss_sisnr_w, loss_spec_w, loss_mse_w, sisnrI_w, loss_cls



