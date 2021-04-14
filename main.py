"""Main script for AED."""
import os
import os.path as osp
import argparse

import torch
from torch import nn
from core import Train, Pre_train, Train2
from datasets.DataProvider import *
from misc.utils import init_model, init_random_seed, load_dataset , custom_test
from misc.saver import Saver
from misc.Arguments import params_appliance
import models

from pdb import set_trace as st

def main(args):

    if args.training_type is 'Train':
        savefilename = osp.join(args.resultdir,args.dataset,args.domain_target)
    elif  args.training_type is 'Pre_train':    
        savefilename = osp.join(args.pre_modeldir,args.dataset,args.domain_target) 

    args.seed = init_random_seed(args.manual_seed)

    # some constant parameters
    CHUNK_SIZE = 5*10**6
    # offset parameter from window length
    offset = int(0.5*(params_appliance[args.domain_target]['windowlength']-1.0))

    ##################### load seed#####################  

    #####################load datasets##################### 
 
    # the appliance to train on
    appliance_name = args.domain_target
    # path for training data
    training_path = args.datadir + args.dataset + '/' +appliance_name + '/' + appliance_name + '_training_' + '.csv'
    print('Training dataset: ' + training_path)
    # path for validation data
    validation_path = args.datadir + args.dataset + '/' +appliance_name + '/' + appliance_name + '_validation_' + '.csv'
    print('Validation dataset: ' + validation_path)
    # path for test data
    test_path = args.datadir + args.dataset + '/' +appliance_name + '/' + appliance_name + '_test_' + '.csv'
    print('Test dataset: ' + test_path)
    test_set_x, test_set_y, ground_truth = load_dataset(test_path,offset = offset)
    test_kwag = {
        'inputs': test_set_x,
        'targets': test_set_y,
        'ground_truth': ground_truth
    }
    # Defining object for training set loading and windowing provider (DataProvider.py)
    tra_provider = ChunkDoubleSourceSlider2(filename=training_path,
                                        batchsize=args.batchsize,
                                        chunksize = CHUNK_SIZE,
                                        crop=args.crop_dataset,
                                        shuffle=True,
                                        offset=offset,
                                        header=0,
                                        )

    # Defining object for validation set loading and windowing provider (DataProvider.py)
    val_provider = ChunkDoubleSourceSlider2(filename=validation_path,
                                        batchsize=args.batchsize,
                                        chunksize=CHUNK_SIZE,
                                        crop=args.crop_dataset,
                                        shuffle=False,
                                        offset=offset,
                                        header=0,
                                        )
    # Defining object for test set loading and windowing provider (DataProvider.py)   
    test_provider = DoubleSourceProvider3(nofWindows=args.nosOfWindows,
                                      offset=offset)                                    


    ##################### load models##################### 

    FeatExtmodel = models.create(args.arch_FeatExt)
    FeatExtmodel_pre1 = models.create(args.arch_FeatExt)
    FeatExtmodel_pre2 = models.create(args.arch_FeatExt)
    FeatExtmodel_pre3 = models.create(args.arch_FeatExt)
    FeatExtmodel_pre4 = models.create(args.arch_FeatExt)
    FeatExtmodel_pre5 = models.create(args.arch_FeatExt)

    FeatEmbdmodel = models.create(args.arch_FeatEmbd, embed_size=args.embed_size)
    FeatEmbdmodel_pre1 = models.create(args.arch_FeatEmbd, embed_size=args.embed_size)
    FeatEmbdmodel_pre2 = models.create(args.arch_FeatEmbd, embed_size=args.embed_size)
    FeatEmbdmodel_pre3 = models.create(args.arch_FeatEmbd, embed_size=args.embed_size)
    FeatEmbdmodel_pre4 = models.create(args.arch_FeatEmbd, embed_size=args.embed_size)
    FeatEmbdmodel_pre5 = models.create(args.arch_FeatEmbd, embed_size=args.embed_size)

    Seq2PointModel = models.create(args.arch_Seq2Point, extractor = FeatExtmodel , embedder = FeatEmbdmodel)
    Seq2PointModel_pre1 = models.create(args.arch_Seq2Point, extractor = FeatExtmodel_pre1 , embedder = FeatEmbdmodel_pre1)
    Seq2PointModel_pre2 = models.create(args.arch_Seq2Point, extractor = FeatExtmodel_pre2 , embedder = FeatEmbdmodel_pre2)
    Seq2PointModel_pre3 = models.create(args.arch_Seq2Point, extractor = FeatExtmodel_pre3 , embedder = FeatEmbdmodel_pre3)
    Seq2PointModel_pre4 = models.create(args.arch_Seq2Point, extractor = FeatExtmodel_pre4 , embedder = FeatEmbdmodel_pre4)
    Seq2PointModel_pre5 = models.create(args.arch_Seq2Point, extractor = FeatExtmodel_pre5 , embedder = FeatEmbdmodel_pre5)

    Dismodel1 = models.create(args.arch_Dis)
    Dismodel2 = models.create(args.arch_Dis)
    Dismodel3 = models.create(args.arch_Dis)
    Dismodel4 = models.create(args.arch_Dis)
    Dismodel5 = models.create(args.arch_Dis)

    if args.training_type is 'Train':

        Seq2Point_restore = None
        Seq2Point_pre1_restore = osp.join(args.pre_modeldir,args.dataset,args.domain1,"Pre-S2P-Final.pt")
        Seq2Point_pre2_restore = osp.join(args.pre_modeldir,args.dataset,args.domain2,"Pre-S2P-Final.pt")
        Seq2Point_pre3_restore = osp.join(args.pre_modeldir,args.dataset,args.domain3,"Pre-S2P-Final.pt")
        Seq2Point_pre4_restore = osp.join(args.pre_modeldir,args.dataset,args.domain4,"Pre-S2P-Final.pt")
        if args.dataset is not 'redd':
            Seq2Point_pre5_restore = osp.join(args.pre_modeldir,args.dataset,args.domain5,"Pre-S2P-Final.pt")
        else:
            Seq2Point_pre5_restore = None
        Dis_restore1 = None
        Dis_restore2 = None
        Dis_restore3 = None
        Dis_restore4 = None
        Dis_restore5 = None


    elif args.training_type is 'Pre_train':
        Seq2Point_restore = None
        Seq2Point_pre1_restore = None
        Seq2Point_pre2_restore = None
        Seq2Point_pre3_restore = None
        Seq2Point_pre4_restore = None
        Seq2Point_pre5_restore = None

        Dis_restore1 = None
        Dis_restore2 = None
        Dis_restore3 = None
        Dis_restore4 = None
        Dis_restore5 = None
    elif args.training_type is 'Pre_test':

        Seq2Point_restore = osp.join(args.pre_modeldir,args.dataset,args.domain_target,"Pre-S2P-{}.pt".format(args.test_epoch))
        Seq2Point_pre1_restore = None
        Seq2Point_pre2_restore = None
        Seq2Point_pre3_restore = None
        Seq2Point_pre4_restore = None
        Seq2Point_pre5_restore = None

        Dis_restore1 = None
        Dis_restore2 = None
        Dis_restore3 = None
        Dis_restore4 = None
        Dis_restore5 = None
    elif args.training_type is 'Test':
        Seq2Point_restore = osp.join(args.resultdir,args.dataset,args.domain_target,"Gen-S2P-{}.pt".format(args.test_epoch))
        Seq2Point_pre1_restore = None
        Seq2Point_pre2_restore = None
        Seq2Point_pre3_restore = None
        Seq2Point_pre4_restore = None
        Seq2Point_pre5_restore = None

        Dis_restore1 = None
        Dis_restore2 = None
        Dis_restore3 = None
        Dis_restore4 = None
        Dis_restore5 = None

    else:
        raise NotImplementedError('method type [%s] is not implemented' % args.training_type)

    Seq2PointModel = init_model(net=Seq2PointModel, init_type = args.init_type, restore=Seq2Point_restore)
    Seq2PointModel_pre1 = init_model(net=Seq2PointModel_pre1, init_type = args.init_type, restore=Seq2Point_pre1_restore)
    Seq2PointModel_pre2 = init_model(net=Seq2PointModel_pre2, init_type = args.init_type, restore=Seq2Point_pre2_restore)
    Seq2PointModel_pre3 = init_model(net=Seq2PointModel_pre3, init_type = args.init_type, restore=Seq2Point_pre3_restore)
    Seq2PointModel_pre4 = init_model(net=Seq2PointModel_pre4, init_type = args.init_type, restore=Seq2Point_pre4_restore)
    Seq2PointModel_pre5 = init_model(net=Seq2PointModel_pre5, init_type = args.init_type, restore=Seq2Point_pre5_restore)

    Discriminator1 = init_model(net=Dismodel1, init_type = args.init_type, restore=Dis_restore1)
    Discriminator2 = init_model(net=Dismodel2, init_type = args.init_type, restore=Dis_restore2)
    Discriminator3 = init_model(net=Dismodel3, init_type = args.init_type, restore=Dis_restore3)
    Discriminator4 = init_model(net=Dismodel4, init_type = args.init_type, restore=Dis_restore4)
    Discriminator5 = init_model(net=Dismodel5, init_type = args.init_type, restore=Dis_restore5)

    ##################### tarining models##################### 

    if args.training_type is 'Train':
        if args.dataset is not 'redd':
            Train(args, Seq2PointModel, Seq2PointModel_pre1, Seq2PointModel_pre2, Seq2PointModel_pre3, Seq2PointModel_pre4, Seq2PointModel_pre5,
                Discriminator1, Discriminator2, Discriminator3,Discriminator4,Discriminator5,
                tra_provider, val_provider, test_provider, test_kwag, savefilename)
        else:
            Train(args, Seq2PointModel, Seq2PointModel_pre1, Seq2PointModel_pre2, Seq2PointModel_pre3, Seq2PointModel_pre4, None,
                Discriminator1, Discriminator2, Discriminator3,Discriminator4,None,
                tra_provider, val_provider, test_provider, test_kwag, savefilename)

    elif args.training_type is 'Pre_train':

        Pre_train(args, Seq2PointModel, tra_provider, val_provider, savefilename ,test_provider,test_kwag)

    elif args.training_type is 'Pre_test' or 'Test':
        Seq2PointModel.eval()
        custom_test(args, test_provider, Seq2PointModel, 1, test_kwag)

    else:
        raise NotImplementedError('method type [%s] is not implemented' % args.training_type)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # domains and datasets 
    parser.add_argument('--dataset', type=str, default='ukdale')
    parser.add_argument('--domain1', type=str, default='microwave')
    parser.add_argument('--domain2', type=str, default='fridge')
    parser.add_argument('--domain3', type=str, default='dishwasher')
    parser.add_argument('--domain4', type=str, default='washingmachine')
    parser.add_argument('--domain5', type=str, default='kettle') 
    parser.add_argument('--domain_target', type=str, default='washingmachine')
    # model
    parser.add_argument('--arch_FeatExt', type=str, default='MyFeatExtractor')
    parser.add_argument('--arch_FeatEmbd', type=str, default='MyFeatEmbedder')
    parser.add_argument('--arch_Seq2Point', type=str, default='Seq2Point')  
    parser.add_argument('--arch_Dis', type=str, default='Discriminator2')

    parser.add_argument('--init_type', type=str, default='xavier')
    parser.add_argument('--embed_size', type=int, default=1024)

    # optimizer
    parser.add_argument('--lr_pre', type=float, default=0.001)
    parser.add_argument('--lr_gen', type=float, default=0.001)
    parser.add_argument('--lr_critic', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    # # training configs
    parser.add_argument('--training_type', type=str, default='Pre_test')
    parser.add_argument('--results_path', type=str, default='./results/Train_20191008')
    parser.add_argument('--batchsize', type=int, default=1000)
    parser.add_argument('--nosOfWindows', type=int,default=100,help='The number of windows for prediction for each iteration.')


    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--pre_epochs', type=int, default=100)
    parser.add_argument('--test_epoch', type=str, default='Final')
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--tst_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=int, default=500)
    parser.add_argument('--model_save_epoch', type=int, default=1)
    parser.add_argument('--manual_seed', type=int, default=None)

    parser.add_argument('--W_pred', type=float, default=1.0)
    parser.add_argument('--W_genave', type=float, default=0.05)

    parser.add_argument('--crop_dataset',
                        type=int,
                        default=None,
                        help='for debugging porpose should be helpful to crop the training dataset size')
    #path
    parser.add_argument('--datadir',type=str,default='./data/',help='this is the directory of the training samples')
    parser.add_argument('--pre_modeldir',type=str,default='./mypremodels',help='this is the directory of the pre_training samples')
    parser.add_argument('--resultdir',type=str,default='./myresults',help='this is the directory of the training samples')


    print(parser.parse_args())
    main(parser.parse_args())

