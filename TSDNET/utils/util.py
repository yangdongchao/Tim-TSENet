import torch
import torch.nn as nn


def handle_scp(scp_path):
    '''
    Read scp file script
    input: 
          scp_path: .scp file's file path
    output: 
          scp_dict: {'key':'wave file path'}
    '''
    scp_dict = dict()
    line = 0
    lines = open(scp_path, 'r').readlines()
    for l in lines:
        scp_parts = l.strip().split()
        line += 1
        if len(scp_parts) != 2:
            raise RuntimeError("For {}, format error in line[{:d}]: {}".format(
                scp_path, line, scp_parts))
        if len(scp_parts) == 2:
            key, value = scp_parts
        if key in scp_dict:
            raise ValueError("Duplicated key \'{0}\' exists in {1}".format(
                key, scp_path))

        scp_dict[key] = value

    return scp_dict

def handle_scp_inf(scp_path):
    '''
    Read information scp file script
    input:
          scp_path: .scp file's file path
    output:
          scp_dict: {'key':'wave file path'}
    '''
    scp_dict_cls = dict()
    scp_dict_onset = dict()
    scp_dict_offset = dict()
    line = 0
    lines = open(scp_path, 'r').readlines()
    for l in lines:
        scp_parts = l.strip().split()
        line += 1
        if len(scp_parts) != 4:
            raise RuntimeError("For {}, format error in line[{:d}]: {}".format(
                scp_path, line, scp_parts))
        if len(scp_parts) == 4:
            key, cls, onset, offset = scp_parts
        if key in scp_dict_cls:
            raise ValueError("Duplicated key \'{0}\' exists in {1}".format(
                key, scp_path))

        scp_dict_cls[key] = int(cls)
        scp_dict_onset[key] = float(onset)
        scp_dict_offset[key] = float(offset)

    return scp_dict_cls, scp_dict_onset, scp_dict_offset

def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


