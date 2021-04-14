import os
from collections import OrderedDict
import torchvision.utils as vutils
from torch.autograd import Variable
import torch
import torch.optim as optim
from torch import nn
from misc.utils import get_inf_iterator, mkdir, custom_test
import numpy as np
import time
from misc.utils import init_model
import os.path as osp

from pdb import set_trace as st

def Pre_train(args,
              Seq2Point, 
              tra_provider, 
              val_provider,
              model_save_path,
              test_provider, 
              test_kwag, 
              save_model = -1,
              print_freq=1,
              earlystopping=True,
              min_epoch=1,
              patience=10,
              validate=False):

    ####################
    # 1. setup network #
    ####################
    # set train state for Dropout and BN layers
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Seq2Point.to(device)

    criterion = torch.nn.MSELoss()

    optimizer = optim.Adam(Seq2Point.parameters(),
                           lr=args.lr_pre,
                           betas=(args.beta1, args.beta2))     
    # optimizer = optim.SGD(Seq2Point.parameters(),
    #                     lr=args.lr_pre)     
    
    # parameters for earlystopping
    best_valid = np.inf
    best_mae = np.inf
    best_mae_epoch = min_epoch

    # Training info
    total_train_loss = []
    total_val_loss = []

    mkdir(model_save_path) 

    ####################
    # 2. train network #
    ####################
    print("Start training the network ...")
    start_time_begin = time.time()
    for epoch in range(args.pre_epochs):
        start_time = time.time()
        loss_ep = 0
        n_step = 0
        print("------------------------- Epoch %d of %d --------------------------" % (epoch + 1, args.pre_epochs))
        Seq2Point.train()
        for batch in tra_provider.feed_chunk():

            X_train_a, y_train_a = batch
            X_train_a = torch.from_numpy(X_train_a).float()
            y_train_a = torch.from_numpy(y_train_a).float()
            #forward
            inputs = Variable(X_train_a)
            target = Variable(y_train_a)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                target = target.cuda()
            out = Seq2Point(inputs)
            loss = criterion(out,target)
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ep += loss
            n_step += 1

        loss_ep = loss_ep / n_step
        print('loss_ep: %f' % loss_ep)

        if (epoch >= 0 or (epoch + 1) % print_freq == 0) and validate:
            # evaluate the val error at each epoch.
            with torch.no_grad():
                Seq2Point.eval()
                if val_provider is not None:
                    print("Epoch %d of %d took %fs" % (epoch + 1, args.pre_epochs, time.time() - start_time))
                    print("Validation...")
                    train_loss, n_batch_train = 0, 0
                    for batch in tra_provider.feed_chunk():
                        X_train_a, y_train_a = batch
                        X_train_a = Variable(torch.from_numpy(X_train_a).float())
                        y_train_a = Variable(torch.from_numpy(y_train_a).float())
                        if torch.cuda.is_available():
                            X_train_a = X_train_a.cuda()
                            y_train_a = y_train_a.cuda()
                        out = Seq2Point(X_train_a)
                        loss = criterion(out,y_train_a)
                        train_loss += loss
                        n_batch_train += 1
                    total_train_loss.append(train_loss/n_batch_train)
                    print("   train loss/n_batch_train: %f" % (train_loss / n_batch_train))
                    print("   train loss: %f, n_batch_train: %d" % (train_loss, n_batch_train))

                    val_loss, n_batch_val = 0, 0

                    for batch in val_provider.feed_chunk():
                        X_val_a, y_val_a = batch
                        X_val_a = Variable(torch.from_numpy(X_val_a).float())
                        y_val_a = Variable(torch.from_numpy(y_val_a).float())
                        if torch.cuda.is_available():
                            X_val_a = X_val_a.cuda()
                            y_val_a = y_val_a.cuda()
                        out = Seq2Point(X_val_a)
                        loss = criterion(out,y_val_a)

                        val_loss += loss
                        n_batch_val += 1
                    print("    val loss: %f" % (val_loss / n_batch_val))
                    total_val_loss.append(val_loss/n_batch_val)
        Seq2Point.eval()
        mae , sae = custom_test(args, test_provider, Seq2Point, 1, test_kwag,False)
        print('MAE:{},SAE:{}, at epoch{}'.format(mae,sae,epoch+1))
        if earlystopping:
            if epoch >= min_epoch:
                print("Evaluate earlystopping parameters...")
                #current_valid = val_loss / n_batch_val
                current_epoch = epoch
                #current_train_loss = train_loss / n_batch_train
                current_mae =  mae
                #print('    Current valid loss was {:.6f}, '
                #    'train loss was {:.6f}, at epoch {}.'
                #    .format(current_valid, current_train_loss, current_epoch+1))
                torch.save(Seq2Point.state_dict(), os.path.join(model_save_path, "Pre-S2P-{}.pt".format(epoch + 1)))
                if current_mae < best_mae:
                    best_mae = current_mae
                    best_mae_epoch = current_epoch

                    # save the model parameters
                    #torch.save(Seq2Point.state_dict(), os.path.join(model_save_path,"Pre-S2P-{}.pt".format(epoch+1)))

                    print('Best mae was {:.6f} at epoch {}.'.format(
                          best_mae, best_mae_epoch+1))
                elif best_mae_epoch + patience < current_epoch:
                    print('Early stopping.')
                    torch.save(Seq2Point.state_dict(), os.path.join(model_save_path, "Pre-S2P-{}.pt".format(current_epoch + 1)))
                    print('Best mae was {:.6f} at epoch {}.'.format(
                          best_mae, best_mae_epoch+1))
                    break

        else:
            current_val_loss = val_loss / n_batch_val
            current_epoch = epoch
            current_train_loss = train_loss / n_batch_train
            current_mae = mae
            print('    Current mae was {:.6f}, validate loss was {:.6f}, train loss was {:.6f}, at epoch {}.'
                .format(current_mae,current_val_loss,current_train_loss, current_epoch+1))

            #print(save_model > 0, epoch % save_model == 0, epoch/save_model > 0)
            if save_model > 0 and epoch % save_model == 0:
                torch.save(Seq2Point.state_dict(), os.path.join(model_save_path,"Pre-S2P-{}.pt".format(epoch+1)))

    if not earlystopping:
        if save_model == -1:
            torch.save(Seq2Point.state_dict(), os.path.join(model_save_path,"Pre-S2P-Final.pt"))

    print("Total training time: %fs" % (time.time() - start_time_begin))
    custom_test(args, test_provider, Seq2Point, 1, test_kwag)
    Seq2Point = init_model(net=Seq2Point, init_type = args.init_type, restore=osp.join(args.pre_modeldir,args.dataset,args.domain_target,"Pre-S2P-{}.pt".format(best_mae_epoch+1)))
    Seq2Point.eval()
    custom_test(args, test_provider, Seq2Point,1,test_kwag)
    return total_train_loss, total_val_loss