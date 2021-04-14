import itertools
import os

from collections import OrderedDict
import torch
import torch.optim as optim
from misc.utils import custom_test, mkdir
from models.loss import  GANLoss
from torch.autograd import Variable
import numpy as np
from misc.utils import init_model
import os.path as osp


def Train(args, Seq2PointModel, Seq2PointModel_pre1, Seq2PointModel_pre2, Seq2PointModel_pre3, Seq2PointModel_pre4,
          Seq2PointModel_pre5,
          Discriminator1, Discriminator2, Discriminator3, Discriminator4, Discriminator5,
          tra_provider, val_provider, test_provider, test_kwag, savefilename, earlystopping=True, save_model=-1, validate=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patience = 10
    best_mae = np.inf
    best_mae_epoch = 1
    # Training info
    total_train_loss = []
    total_val_loss = []
    mkdir(savefilename)
    ####################
    # 1. setup network #
    ####################
    # set train state for Dropout and BN layers
    Seq2PointModel.train()
    Discriminator1.train()
    Discriminator2.train()
    Discriminator3.train()
    Discriminator4.train()

    Seq2PointModel_pre1.eval()
    Seq2PointModel_pre2.eval()
    Seq2PointModel_pre3.eval()
    Seq2PointModel_pre4.eval()

    Seq2PointModel.to(device)
    Discriminator1.to(device)
    Discriminator2.to(device)
    Discriminator3.to(device)
    Discriminator4.to(device)
    Seq2PointModel_pre1.to(device)
    Seq2PointModel_pre2.to(device)
    Seq2PointModel_pre3.to(device)
    Seq2PointModel_pre4.to(device)
    if Seq2PointModel_pre5 is not None:
        Discriminator5.train()
        Seq2PointModel_pre5.eval()
        Discriminator5.to(device)
        Seq2PointModel_pre5.to(device)

    # setup criterion and optimizer
    criterion_pred = torch.nn.MSELoss()
    criterionAdv = GANLoss()

    optimizer_gen = optim.Adam(Seq2PointModel.parameters(),
                               lr=args.lr_gen,
                               betas=(args.beta1, args.beta2))

    optimizer_critic1 = optim.Adam(Discriminator1.parameters(),
                                   lr=args.lr_critic,
                                   betas=(args.beta1, args.beta2))

    optimizer_critic2 = optim.Adam(Discriminator2.parameters(),
                                   lr=args.lr_critic,
                                   betas=(args.beta1, args.beta2))

    optimizer_critic3 = optim.Adam(Discriminator3.parameters(),
                                   lr=args.lr_critic,
                                   betas=(args.beta1, args.beta2))

    optimizer_critic4 = optim.Adam(Discriminator4.parameters(),
                                   lr=args.lr_critic,
                                   betas=(args.beta1, args.beta2))
    if Seq2PointModel_pre5 is not None:
        optimizer_critic5 = optim.Adam(Discriminator5.parameters(),
                                       lr=args.lr_critic,
                                       betas=(args.beta1, args.beta2))

    ####################
    # 2. train network #
    ####################
    global_step = 0

    for epoch in range(args.epochs):

        step = 0
        Seq2PointModel.train()
        Discriminator1.train()
        Discriminator2.train()
        Discriminator3.train()
        Discriminator4.train()
        if Seq2PointModel_pre5 is not None:
            Discriminator5.train()
        print('------------------------epoch {}------------------------------------'.format(epoch + 1))
        for batch in tra_provider.feed_chunk():
            # ============ one batch extraction ============#
            # target train data
            X_train_a, y_train_a = batch
            X_train_a = Variable(torch.from_numpy(X_train_a).float())
            y_train_a = Variable(torch.from_numpy(y_train_a).float())
            if torch.cuda.is_available():
                X_train_a = X_train_a.cuda()
                y_train_a = y_train_a.cuda()

            with torch.no_grad():
                pre_feat_ext1 = Seq2PointModel_pre1.extractor(X_train_a)
                pre_feat_ext2 = Seq2PointModel_pre2.extractor(X_train_a)
                pre_feat_ext3 = Seq2PointModel_pre3.extractor(X_train_a)
                pre_feat_ext4 = Seq2PointModel_pre4.extractor(X_train_a)
                if 'redd' not in args.dataset:
                    pre_feat_ext5 = Seq2PointModel_pre5.extractor(X_train_a)

            # ============ domain generalization supervision ============#
            feat_ext = Seq2PointModel.extractor(X_train_a)
            optimizer_gen.zero_grad()

            # ************************* confusion all **********************************#

            # predict on generator
            loss_generator1 = criterionAdv(Discriminator1(feat_ext), True)

            loss_generator2 = criterionAdv(Discriminator2(feat_ext), True)

            loss_generator3 = criterionAdv(Discriminator3(feat_ext), True)

            loss_generator4 = criterionAdv(Discriminator4(feat_ext), True)

            if Seq2PointModel_pre5 is not None:
                loss_generator5 = criterionAdv(Discriminator5(feat_ext), True)
                Loss_gen = args.W_genave * (
                            loss_generator1 + loss_generator2 + loss_generator3 + loss_generator4 + loss_generator5)

            else:
                Loss_gen = args.W_genave * (loss_generator1 + loss_generator2 + loss_generator3 + loss_generator4)

            Loss_gen.backward()
            optimizer_gen.step()

            # ************************* confusion domain 1  **********************************#

            # predict on discriminator
            optimizer_critic1.zero_grad()

            real_loss = criterionAdv(Discriminator1(pre_feat_ext1), True)
            fake_loss = criterionAdv(Discriminator1(feat_ext.detach()), False)

            loss_critic1 = 0.5 * (real_loss + fake_loss)

            loss_critic1.backward()
            optimizer_critic1.step()

            # ************************* confusion domain 2  **********************************#

            # predict on discriminator
            optimizer_critic2.zero_grad()

            real_loss = criterionAdv(Discriminator2(pre_feat_ext2), True)
            fake_loss = criterionAdv(Discriminator2(feat_ext.detach()), False)

            loss_critic2 = 0.5 * (real_loss + fake_loss)

            loss_critic2.backward()
            optimizer_critic2.step()

            # ************************* confusion domain 3 **********************************#

            # predict on discriminator
            optimizer_critic3.zero_grad()

            real_loss = criterionAdv(Discriminator3(pre_feat_ext3), True)
            fake_loss = criterionAdv(Discriminator3(feat_ext.detach()), False)

            loss_critic3 = 0.5 * (real_loss + fake_loss)

            loss_critic3.backward()
            optimizer_critic3.step()
            # ************************* confusion domain 4  **********************************#

            # predict on discriminator
            optimizer_critic4.zero_grad()

            real_loss = criterionAdv(Discriminator4(pre_feat_ext4), True)
            fake_loss = criterionAdv(Discriminator4(feat_ext.detach()), False)

            loss_critic4 = 0.5 * (real_loss + fake_loss)

            loss_critic4.backward()
            optimizer_critic4.step()

            # ************************* confusion domain 5 **********************************#

            # predict on discriminator
            if Seq2PointModel_pre5 is not None:
                optimizer_critic5.zero_grad()

                real_loss = criterionAdv(Discriminator5(pre_feat_ext5), True)
                fake_loss = criterionAdv(Discriminator5(feat_ext.detach()), False)

                loss_critic5 = 0.5 * (real_loss + fake_loss)

                loss_critic5.backward()
                optimizer_critic5.step()

            # ============ prediction supervision ============#

            ######### predict loss #########
            optimizer_gen.zero_grad()
            feat_ext = Seq2PointModel.extractor(X_train_a)
            power_Pre = Seq2PointModel.embedder(feat_ext)

            Loss_pred = args.W_pred * criterion_pred(power_Pre, y_train_a)

            Loss_pred.backward()

            optimizer_gen.step()
            # ============ print the print info ============#
            # if (step + 1) % args.log_step == 0:
            #     errors = OrderedDict([
            #         ('Loss_pred', Loss_pred.item()),

            #         ('loss_critic1', loss_critic1.item()),
            #         ('loss_generator1', loss_generator1.item()),

            #         ('loss_critic2', loss_critic2.item()),
            #         ('loss_generator2', loss_generator2.item()),

            #         ('loss_critic3', loss_critic3.item()),
            #         ('loss_generator3', loss_generator3.item()),

            #         ('loss_critic4', loss_critic4.item()),
            #         ('loss_generator4', loss_generator4.item())])
            #     if Seq2PointModel_pre5 is not None:
            #         OrderedDict.append(('loss_critic5', loss_critic5.item()),
            #                            ('loss_generator5', loss_generator5.item()))
            #     print(errors)
            step+=1
            global_step += 1

        # evaluate the val error at each epoch.
        with torch.no_grad():
            Seq2PointModel.eval()
            if val_provider is not None and validate:
                print("Validation...")
                train_loss, n_batch_train = 0, 0
                for batch in tra_provider.feed_chunk():
                    X_train_a, y_train_a = batch
                    X_train_a = Variable(torch.from_numpy(X_train_a).float())
                    y_train_a = Variable(torch.from_numpy(y_train_a).float())
                    if torch.cuda.is_available():
                        X_train_a = X_train_a.cuda()
                        y_train_a = y_train_a.cuda()
                    out = Seq2PointModel(X_train_a)
                    loss = criterion_pred(out, y_train_a)
                    train_loss += loss
                    n_batch_train += 1
                total_train_loss.append(train_loss / n_batch_train)
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
                    out = Seq2PointModel(X_val_a)
                    loss = criterion_pred(out, y_val_a)

                    val_loss += loss
                    n_batch_val += 1
                print("    val loss: %f" % (val_loss / n_batch_val))
                total_val_loss.append(val_loss / n_batch_val)

        mae, sae = custom_test(args, test_provider, Seq2PointModel, 1, test_kwag, False)
        print('MAE:{},SAE:{}, at epoch {}'.format(mae, sae, epoch + 1))
        # save the model parameters
        #torch.save(Seq2PointModel.state_dict(), os.path.join(savefilename, "Gen-S2P-{}.pt".format(epoch + 1)))
        if earlystopping:
            if epoch >= 1:
                if validate:
                    print("Evaluate earlystopping parameters...")
                    current_valid = val_loss / n_batch_val
                    current_epoch = epoch
                    current_train_loss = train_loss / n_batch_train
                    # current_mae = mae
                    print('    Current valid loss was {:.6f}, '
                          'train loss was {:.6f}, at epoch {}.'
                          .format(current_valid, current_train_loss, current_epoch + 1))
                current_mae = mae
                current_epoch = epoch
                if current_mae < best_mae:
                    best_mae = current_mae
                    best_mae_epoch = current_epoch
                    print('Best mae was {:.6f} at epoch {}.'.format(
                        best_mae, best_mae_epoch + 1))
                elif best_mae_epoch + patience < current_epoch:
                    print('Early stopping.')
                    print('Best mae was {:.6f} at epoch {}.'.format(
                        best_mae, best_mae_epoch + 1))
                    break

        else:
            current_val_loss = val_loss / n_batch_val
            current_epoch = epoch
            current_train_loss = train_loss / n_batch_train
            current_mae = mae
            print('    Current mae was {:.6f}, validate loss was {:.6f}, train loss was {:.6f}, at epoch {}.'
                  .format(current_mae, current_val_loss, current_train_loss, current_epoch + 1))

    print("Training compelete!")

def Train2(args, Seq2PointModel, Seq2PointModel_pre1, Seq2PointModel_pre2, Seq2PointModel_pre3, Seq2PointModel_pre4,
          Seq2PointModel_pre5,
          Discriminator1, Discriminator2, Discriminator3, Discriminator4, Discriminator5,
          tra_provider, val_provider, test_provider, test_kwag, savefilename, earlystopping=True, save_model=-1, validate=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patience = 10
    best_mae = np.inf
    best_mae_epoch = 1
    # Training info
    total_train_loss = []
    total_val_loss = []
    mkdir(savefilename)
    ####################
    # 1. setup network #
    ####################
    # set train state for Dropout and BN layers
    Seq2PointModel.train()
    Discriminator1.train()
    Discriminator2.train()
    Discriminator3.train()
    Discriminator4.train()

    Seq2PointModel_pre1.eval()
    Seq2PointModel_pre2.eval()
    Seq2PointModel_pre3.eval()
    Seq2PointModel_pre4.eval()

    Seq2PointModel.to(device)
    Discriminator1.to(device)
    Discriminator2.to(device)
    Discriminator3.to(device)
    Discriminator4.to(device)
    Seq2PointModel_pre1.to(device)
    Seq2PointModel_pre2.to(device)
    Seq2PointModel_pre3.to(device)
    Seq2PointModel_pre4.to(device)
    if Seq2PointModel_pre5 is not None:
        Discriminator5.train()
        Seq2PointModel_pre5.eval()
        Discriminator5.to(device)
        Seq2PointModel_pre5.to(device)

    # setup criterion and optimizer
    criterion_pred = torch.nn.MSELoss()
    criterionAdv = GANLoss()

    optimizer_gen = optim.Adam(Seq2PointModel.parameters(),
                               lr=args.lr_gen,
                               betas=(args.beta1, args.beta2))

    optimizer_critic1 = optim.Adam(Discriminator1.parameters(),
                                   lr=args.lr_critic,
                                   betas=(args.beta1, args.beta2))

    optimizer_critic2 = optim.Adam(Discriminator2.parameters(),
                                   lr=args.lr_critic,
                                   betas=(args.beta1, args.beta2))

    optimizer_critic3 = optim.Adam(Discriminator3.parameters(),
                                   lr=args.lr_critic,
                                   betas=(args.beta1, args.beta2))

    optimizer_critic4 = optim.Adam(Discriminator4.parameters(),
                                   lr=args.lr_critic,
                                   betas=(args.beta1, args.beta2))
    if Seq2PointModel_pre5 is not None:
        optimizer_critic5 = optim.Adam(Discriminator5.parameters(),
                                       lr=args.lr_critic,
                                       betas=(args.beta1, args.beta2))

    ####################
    # 2. train network #
    ####################
    global_step = 0

    for epoch in range(args.epochs):

        step = 0
        Seq2PointModel.train()
        Discriminator1.train()
        Discriminator2.train()
        Discriminator3.train()
        Discriminator4.train()
        if Seq2PointModel_pre5 is not None:
            Discriminator5.train()
        print('------------------------epoch {}------------------------------------'.format(epoch + 1))
        for batch in tra_provider.feed_chunk():
            # ============ one batch extraction ============#
            # target train data
            X_train_a, y_train_a = batch
            X_train_a = Variable(torch.from_numpy(X_train_a).float())
            y_train_a = Variable(torch.from_numpy(y_train_a).float())
            if torch.cuda.is_available():
                X_train_a = X_train_a.cuda()
                y_train_a = y_train_a.cuda()

            with torch.no_grad():
                pre_feat_ext1 = Seq2PointModel_pre1.extractor(X_train_a)
                pre_feat_ext2 = Seq2PointModel_pre2.extractor(X_train_a)
                pre_feat_ext3 = Seq2PointModel_pre3.extractor(X_train_a)
                pre_feat_ext4 = Seq2PointModel_pre4.extractor(X_train_a)
                if 'redd' not in args.dataset:
                    pre_feat_ext5 = Seq2PointModel_pre5.extractor(X_train_a)
            # ============ prediction supervision ============#

            ######### predict loss #########
            feat_ext = Seq2PointModel.extractor(X_train_a)
            power_Pre = Seq2PointModel.embedder(feat_ext)

            Loss_pred = args.W_pred * criterion_pred(power_Pre, y_train_a)

            # ============ domain generalization loss ============#
            # ************************* confusion all **********************************#

            # predict on generator
            loss_generator1 = criterionAdv(Discriminator1(feat_ext), True)

            loss_generator2 = criterionAdv(Discriminator2(feat_ext), True)

            loss_generator3 = criterionAdv(Discriminator3(feat_ext), True)

            loss_generator4 = criterionAdv(Discriminator4(feat_ext), True)

            if Seq2PointModel_pre5 is not None:
                loss_generator5 = criterionAdv(Discriminator5(feat_ext), True)
                Loss_gen = args.W_genave * (
                            loss_generator1 + loss_generator2 + loss_generator3 + loss_generator4 + loss_generator5)

            else:
                Loss_gen = args.W_genave * (loss_generator1 + loss_generator2 + loss_generator3 + loss_generator4)

            optimizer_gen.zero_grad()
            loss_both = Loss_pred + Loss_gen
            loss_both.backward()
            optimizer_gen.step()

            # ************************* confusion domain 1  **********************************#

            # predict on discriminator
            optimizer_critic1.zero_grad()

            real_loss = criterionAdv(Discriminator1(pre_feat_ext1), True)
            fake_loss = criterionAdv(Discriminator1(feat_ext.detach()), False)

            loss_critic1 = 0.5 * (real_loss + fake_loss)

            loss_critic1.backward()
            optimizer_critic1.step()

            # ************************* confusion domain 2  **********************************#

            # predict on discriminator
            optimizer_critic2.zero_grad()

            real_loss = criterionAdv(Discriminator2(pre_feat_ext2), True)
            fake_loss = criterionAdv(Discriminator2(feat_ext.detach()), False)

            loss_critic2 = 0.5 * (real_loss + fake_loss)

            loss_critic2.backward()
            optimizer_critic2.step()

            # ************************* confusion domain 3 **********************************#

            # predict on discriminator
            optimizer_critic3.zero_grad()

            real_loss = criterionAdv(Discriminator3(pre_feat_ext3), True)
            fake_loss = criterionAdv(Discriminator3(feat_ext.detach()), False)

            loss_critic3 = 0.5 * (real_loss + fake_loss)

            loss_critic3.backward()
            optimizer_critic3.step()
            # ************************* confusion domain 4  **********************************#

            # predict on discriminator
            optimizer_critic4.zero_grad()

            real_loss = criterionAdv(Discriminator4(pre_feat_ext4), True)
            fake_loss = criterionAdv(Discriminator4(feat_ext.detach()), False)

            loss_critic4 = 0.5 * (real_loss + fake_loss)

            loss_critic4.backward()
            optimizer_critic4.step()

            # ************************* confusion domain 5 **********************************#

            # predict on discriminator
            if Seq2PointModel_pre5 is not None:
                optimizer_critic5.zero_grad()

                real_loss = criterionAdv(Discriminator5(pre_feat_ext5), True)
                fake_loss = criterionAdv(Discriminator5(feat_ext.detach()), False)

                loss_critic5 = 0.5 * (real_loss + fake_loss)

                loss_critic5.backward()
                optimizer_critic5.step()

            # ============ print the print info ============#
            # if (step + 1) % args.log_step == 0:
            #     errors = OrderedDict([
            #         ('Loss_pred', Loss_pred.item()),

            #         ('loss_critic1', loss_critic1.item()),
            #         ('loss_generator1', loss_generator1.item()),

            #         ('loss_critic2', loss_critic2.item()),
            #         ('loss_generator2', loss_generator2.item()),

            #         ('loss_critic3', loss_critic3.item()),
            #         ('loss_generator3', loss_generator3.item()),

            #         ('loss_critic4', loss_critic4.item()),
            #         ('loss_generator4', loss_generator4.item())])
            #     if Seq2PointModel_pre5 is not None:
            #         OrderedDict.append(('loss_critic5', loss_critic5.item()),
            #                            ('loss_generator5', loss_generator5.item()))

                # Saver.print_current_errors((epoch+1), (step+1), errors)
                # print(errors)
            global_step += 1

        # evaluate the val error at each epoch.
        with torch.no_grad():
            Seq2PointModel.eval()
            if val_provider is not None and validate:
                print("Validation...")
                train_loss, n_batch_train = 0, 0
                for batch in tra_provider.feed_chunk():
                    X_train_a, y_train_a = batch
                    X_train_a = Variable(torch.from_numpy(X_train_a).float())
                    y_train_a = Variable(torch.from_numpy(y_train_a).float())
                    if torch.cuda.is_available():
                        X_train_a = X_train_a.cuda()
                        y_train_a = y_train_a.cuda()
                    out = Seq2PointModel(X_train_a)
                    loss = criterion_pred(out, y_train_a)
                    train_loss += loss
                    n_batch_train += 1
                total_train_loss.append(train_loss / n_batch_train)
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
                    out = Seq2PointModel(X_val_a)
                    loss = criterion_pred(out, y_val_a)

                    val_loss += loss
                    n_batch_val += 1
                print("    val loss: %f" % (val_loss / n_batch_val))
                total_val_loss.append(val_loss / n_batch_val)

        mae, sae = custom_test(args, test_provider, Seq2PointModel, 1, test_kwag, False)
        print('MAE:{},SAE:{}, at epoch {}'.format(mae, sae, epoch + 1))
        # save the model parameters
        torch.save(Seq2PointModel.state_dict(), os.path.join(savefilename, "Gen-S2P-{}.pt".format(epoch + 1)))
        if earlystopping:
            if epoch >= 1:
                if validate:
                    print("Evaluate earlystopping parameters...")
                    current_valid = val_loss / n_batch_val
                    current_epoch = epoch
                    current_train_loss = train_loss / n_batch_train
                    # current_mae = mae
                    print('    Current valid loss was {:.6f}, '
                          'train loss was {:.6f}, at epoch {}.'
                          .format(current_valid, current_train_loss, current_epoch + 1))
                current_mae = mae
                current_epoch = epoch
                if current_mae < best_mae:
                    best_mae = current_mae
                    best_mae_epoch = current_epoch
                    print('Best mae was {:.6f} at epoch {}.'.format(
                        best_mae, best_mae_epoch + 1))
                elif best_mae_epoch + patience < current_epoch:
                    print('Early stopping.')
                    print('Best mae was {:.6f} at epoch {}.'.format(
                        best_mae, best_mae_epoch + 1))
                    break

        else:
            current_val_loss = val_loss / n_batch_val
            current_epoch = epoch
            current_train_loss = train_loss / n_batch_train
            current_mae = mae
            print('    Current mae was {:.6f}, validate loss was {:.6f}, train loss was {:.6f}, at epoch {}.'
                  .format(current_mae, current_val_loss, current_train_loss, current_epoch + 1))

    print("Training compelete!")

