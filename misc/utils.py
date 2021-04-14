from __future__ import division

import os
import random
from PIL import Image
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn import init
import pandas as pd
import math
import misc.nilm_metric as nm

from pdb import set_trace as st
from .Arguments import *

def make_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().detach().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    return image_numpy.astype(imtype)


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal_rnn(m):
    classname = m.__class__.__name__
    if classname.find('LSTM') != -1:
        init.orthogonal_(m.all_weights[0][0], gain=1)
        init.orthogonal_(m.all_weights[0][1], gain=1)
        init.constant_(m.all_weights[0][2], 1)
        init.constant_(m.all_weights[0][3], 1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    elif init_type == 'orthogonal_rnn':
        net.apply(weights_init_orthogonal_rnn)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)                


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


# def get_data_loader(name, data_root, batch_size, getreal):
#     """Get data loader by name."""
#     if name == "D3MAD":
#         return get_D3MAD(data_root, batch_size,getreal)


def init_model(net, restore, init_type, init= True):
    """Init models with cuda and weights."""
    # init weights of model
    if init:
        init_weights(net, init_type)
    
    # restore model weights
    if restore is not None and os.path.exists(restore):

        # original saved file with DataParallel
        state_dict = torch.load(restore)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        #new_state_dict = OrderedDict()
        #for k, v in state_dict.items():
        #    name = k[7:] # remove `module.`
        #    new_state_dict[name] = v
        # load params
        net.load_state_dict(state_dict)
        
        # net.load_state_dict(torch.load(restore))
        net.restored = True
        print("*************Restore model from: {}".format(os.path.abspath(restore)))

    if torch.cuda.is_available():
        cudnn.benchmark = True
        net = net.cuda()

    return net


# def save_model(net, filename):
#     """Save trained model."""
#     if not os.path.exists(args.model_root):
#         os.makedirs(args.model_root)
#     torch.save(net.state_dict(),
#                os.path.join(args.model_root, filename))
#     print("save pretrained model to: {}".format(os.path.join(args.model_root,
#                                                              filename)))

# def save_trainedmodel(net, filename):
#     """Save trained model."""
#     if not os.path.exists(os.path.join(args.model_root, args.namesave)):
#         os.makedirs(os.path.join(args.model_root, args.namesave))
#     torch.save(net.state_dict(),
#                os.path.join(args.model_root, args.namesave, filename))
#     print("save pretrained model to: {}".format(os.path.join(args.model_root, args.namesave,
#                                                              filename)))


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def get_sampled_data_loader(dataset, candidates_num, shuffle=True):
    """Get data loader for sampled dataset."""
    # get indices
    indices = torch.arange(0, len(dataset))
    if shuffle:
        indices = torch.randperm(len(dataset))
    # slice indices
    candidates_num = min(len(dataset), candidates_num)
    excerpt = indices.narrow(0, 0, candidates_num).long()
    sampler = torch.utils.data.sampler.SubsetRandomSampler(excerpt)
    return make_data_loader(dataset, sampler=sampler, shuffle=False)

 
    

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil = image_pil.resize((128, 128), resample=Image.BICUBIC)
    image_pil.save(image_path)



def get_inf_iterator(data_loader):
    """Inf data iterator."""
    while True:
        for catimages, depthimages, labels in data_loader:
            yield (catimages, depthimages, labels)

def get_inf_iterator_tst(data_loader):
    """Inf data iterator."""
    while True:
        for catimages, labels in data_loader:
            yield (catimages, labels)

def custom_test(args,
                test_provider,
                model,
                output_length=1,
                test_kwag=None,
                prt=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    output_container = []
    with torch.no_grad():
        for X_out in test_provider.feed(test_kwag['inputs']):
            x = torch.from_numpy(X_out).float()
            if torch.cuda.is_available():
                x = x.cuda()
            output = model(x)
            output_array = output.cpu().detach().numpy().reshape(-1, output_length)
            output_container.append(output_array)

    test_prediction = np.vstack(output_container)
    # ------------------------------------- Performance evaluation----------------------------------------------------------

    # Parameters
    max_power = params_appliance[args.domain_target]['max_on_power']
    threshold = params_appliance[args.domain_target]['on_power_threshold']
    aggregate_mean = 522
    aggregate_std = 814


    appliance_mean = params_appliance[args.domain_target]['mean']
    appliance_std = params_appliance[args.domain_target]['std']

    prediction = test_prediction * appliance_std + appliance_mean
    prediction[prediction <= 0.0] = 0.0
    ground_truth = test_kwag['ground_truth'] * appliance_std + appliance_mean
    #print(ground_truth[1:20], prediction[1:20],test_kwag['ground_truth'][1:20])
    # ------------------------------------------ metric evaluation----------------------------------------------------------
    sample_second = 8.0  # sample time is 8 seconds

    r = nm.get_abs_error(ground_truth.flatten(), prediction.flatten())

    sae = nm.get_sae(ground_truth.flatten(), prediction.flatten(), sample_second)

    if prt:
        print('aggregate_mean: ' + str(aggregate_mean))
        print('aggregate_std: ' + str(aggregate_std))
        print('appliance_mean: ' + str(appliance_mean))
        print('appliance_std: ' + str(appliance_std))
        print('F1:{0}'.format(nm.get_F1(ground_truth.flatten(), prediction.flatten(), threshold)))
        print('NDE:{0}'.format(nm.get_nde(ground_truth.flatten(), prediction.flatten())))
        print(
            '\nMAE: {:}\n    -std: {:}\n    -min: {:}\n    -max: {:}\n    -q1: {:}\n    -median: {:}\n    -q2: {:}'.format(
                *r))
        print('SAE: {:}'.format(sae))
        print('Energy per Day: {:}'.format(nm.get_Epd(ground_truth.flatten(), prediction.flatten(), sample_second)))
    return r[0], sae

def load_dataset(filename, header=0, offset = 0):
    data_frame = pd.read_csv(filename,
                             nrows=None,
                             header=header,
                             na_filter=False,
                             #memory_map=True
                             )

    test_set_x = np.round(np.array(data_frame.iloc[:, 0], float), 5)
    test_set_y = np.round(np.array(data_frame.iloc[:, 1], float), 5)
    ground_truth = np.round(np.array(data_frame.iloc[offset:-offset, 1]), 5)
    #ground_truth = np.round(np.array(data_frame.iloc[0:-2*offset, 1]), 5)
    del data_frame
    return test_set_x, test_set_y, ground_truth
