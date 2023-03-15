#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 22:55:37 2021

@author: Mohamed A. Suliman

email: mohamedabdall78@hotmail.com
"""
import os

import numpy as np

device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from models.DDR_dataloader import MRIImages
from models.DDR_fine_model import DDR_fine
from utils.utils import LossFuns, grad_loss, save_gifti
from tqdm import tqdm
from utils.utils import icosphere_upsampling
from models.DDR_coarse_model import DDR_coarse

#################################
### Set your hyper-parameters ###
#################################
ver_dic = {0: 12, 1: 42, 2: 162, 3: 642, 4: 2562, 5: 10242, 6: 40962, 7: 163842, 8: 655362}
batch_size = 1
in_channels = 2
learning_rate = 0.0001
num_feat = [2, 4, 8, 16, 32]

data_ico = 6  # input data ico
labels_ico_coar = 8  # Labels ico level
control_ico_coar = 4  # Control ico level
num_labels_coar = 1000

lambda_mse = 1.0
lambda_cc = 1.0
lambda_reg = 0.6

loss_pen = 'l1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'cpu'#

best_val = 1000
print_step = 1

m_checkpoint = True
testing = True
testing_best = True
save_def_gifti = True

Num_Epochs = 100

############################
### Set your directories ###
############################
num_worker = 1
moving_dir = 'moving_images/'  # moving imgs location
target_dir = 'target_images/'  # target imgs location

ddr_files_dir = 'DDR_files/'  # DDR files directory

# moving imgs are named as e.g., ID.L.sulc.affine.ico6.shape.gii
# target img is named as e.g., MSMSulc.L.sulc.ico6.shape.gii
# IDs are loaded into Id_files

# moving imgs Ids files
Id_file_t1 = ddr_files_dir + 'Subjects_IDs/Subjects_ID_1'
Id_file_t2 = ddr_files_dir + 'Subjects_IDs/Subjects_ID_2'
Id_file_t3 = ddr_files_dir + 'Subjects_IDs/Subjects_ID_3'
Id_file_val = ddr_files_dir + 'Subjects_IDs/Subjects_ID_val'
Id_file_test = ddr_files_dir + 'Subjects_IDs/Subjects_ID_test'  # if testing == True

upsample3=torch.LongTensor(np.load(ddr_files_dir+'upsample_to_ico3.npy'))
upsample4=torch.LongTensor(np.load(ddr_files_dir+'upsample_to_ico4.npy'))

moving_suffix = '.affine.ico-6.shape.gii'  # names without the Id number
target_prefix = 'MSMSulc'
target_suffix = '.L.sulc.ico6.shape.gii'

edge_in = torch.LongTensor(np.load(ddr_files_dir + 'edge_index_' + str(data_ico) + '.npy')).to(device)
hex_in = torch.LongTensor(np.load(ddr_files_dir + 'hexagons_' + str(data_ico) + '.npy')).to(device)

save_model_dir = 'results/models/'  # where to save the model. Only if m_checkpoint=True
save_def_ico_dir = 'results/deformed_icos/'  # where to save the gifti format of deformed control icos. Only if save_def_gifti ==True

### you probably don't need to change anything beyond this point ###



## define val and tesing functions ##
def Coarse_eval(moving_imgs, target_imgs,edge_in):
    coarse_model.eval()

    with torch.no_grad():

        warped_moving_test, def_control_ico, data_ico_warps = coarse_model(moving_imgs, target_imgs, edge_in)

        #test_losses_mse, test_losses_cc = LossFuns(warped_moving_test,target_imgs)

        coarse_warp = def_control_ico.reshape(ver_dic[2], 3)
        coarse_warp = icosphere_upsampling(162, coarse_warp, 162 * 4 - 6, upsample3, device=device)
        coarse_warp = icosphere_upsampling(642, coarse_warp, 642 * 4 - 6, upsample4, device=device)
        coarse_warp = coarse_warp / (torch.norm(coarse_warp, dim=1, keepdim=True).repeat(1, 3) + 1e-12)

    #return test_losses_mse, test_losses_cc, coarse_warp
    return coarse_warp

def DDR_validation(dataloader, edge_in):

    model_fine.eval()

    val_losses_mse = torch.zeros((len(dataloader), 1))
    val_losses_cc = torch.zeros((len(dataloader), 1))

    for batch_idx, (moving_imgs, target_imgs) in enumerate(dataloader):

        moving_imgs, target_imgs = (moving_imgs.squeeze(0)).to(device), (target_imgs.squeeze(0)).to(device)
        #coarse_mse, coarse_cc, coarse_warp = Coarse_eval(moving_imgs, target_imgs, edge_in)
        coarse_warp = Coarse_eval(moving_imgs, target_imgs, edge_in)
        with torch.no_grad():
            warped_moving_val, _, _ = model_fine(moving_imgs, target_imgs, edge_in,coarse_warp)

            val_losses_mse[batch_idx, :], val_losses_cc[batch_idx, :] = LossFuns(warped_moving_val, target_imgs)


    return val_losses_cc + val_losses_mse, val_losses_mse


def DDR_testing(dataloader, edge_in):
    test_model.eval()

    registered_imgs = torch.zeros(ver_dic[data_ico], len(dataloader))
    def_control_icos = torch.zeros(ver_dic[control_ico_coar] * 3, len(dataloader))


    test_losses_mse = torch.zeros((len(dataloader), 1))
    test_losses_cc = torch.zeros((len(dataloader), 1))


    for batch_idx, (moving_imgs, target_imgs) in enumerate(dataloader):
        moving_imgs, target_imgs = (moving_imgs.squeeze(0)).to(device), (target_imgs.squeeze(0)).to(device)
        #coarse_mse, coarse_cc, coarse_warp = Coarse_eval(moving_imgs, target_imgs, edge_in)
        coarse_warp = Coarse_eval(moving_imgs, target_imgs, edge_in)
        with torch.no_grad():
            warped_moving_test, def_control_ico, _ = test_model(moving_imgs, target_imgs, edge_in,coarse_warp)

            registered_imgs[:, batch_idx] = warped_moving_test.squeeze(1)

            test_losses_mse[batch_idx, :], test_losses_cc[batch_idx, :] = LossFuns(warped_moving_test,
                                                                                   target_imgs)
            def_control_icos[:, batch_idx] = (def_control_ico.reshape(-1, 1)).squeeze()


    return registered_imgs, test_losses_mse, test_losses_cc, def_control_icos


def print_during_training(epoch, trn_ls, trn_cc,trn_mse, val_ls, val_mse):
    print('\n')
    print('(Ep: {}) * (T.L = {:.4}) *** (T.CC = {:.4})*** (T.MSE = {:.4})\n ******** (V.L = {:.4}) *** (V.MSE = {:.4}) *********'
          .format(epoch, trn_ls, trn_cc, trn_mse, val_ls, val_mse))


## load coarse model

coarse_model = DDR_coarse(in_ch=in_channels, num_features=[32,64,128,256,512], data_ico=6,
                            labels_ico=5, control_ico=2,
                            num_labels=600, device=device)
torch.cuda.empty_cache()

coarse_model.to(device)

coarse_model.load_state_dict(torch.load(save_model_dir + 'best_coarse_model.pkl', map_location=device))


## define tensors to be used during training ##
train_loss = torch.zeros(Num_Epochs, 1)
train_loss_mse = torch.zeros(Num_Epochs, 1)
train_loss_cc = torch.zeros(Num_Epochs, 1)
val_loss = torch.zeros(Num_Epochs, 1)
val_loss_mse = torch.zeros(Num_Epochs, 1)


## define datasets ##

train_dataset = MRIImages(moving_dir, target_dir, moving_suffix, target_prefix, target_suffix,
                          Id_file1=Id_file_t1, Id_file2=Id_file_t2, Id_file3=Id_file_t3)

val_dataset = MRIImages(moving_dir, target_dir, moving_suffix, target_prefix, target_suffix,
                        Id_file1=Id_file_val)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_worker, pin_memory=True)

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=num_worker, pin_memory=True)

num_train_data = len(train_dataloader)

print('Number of Train Images = ', num_train_data)
print('Number of Val Images  = ', len(val_dataloader))
print('Number of Epochs =', Num_Epochs)

## define the fine model ##
model_fine = DDR_fine(in_ch=in_channels, num_features=num_feat,
                          data_ico=data_ico, labels_ico=labels_ico_coar, control_ico=control_ico_coar,
                          num_labels=num_labels_coar, device=device)
model_fine.to(device)
optimizer = torch.optim.Adam(model_fine.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True,
                                                       threshold=0.0001, threshold_mode='rel', min_lr=1e-7)

print("The device is '{}' ".format(device))
print("The DDR fine model has {} Paramerters".format(sum(x.numel() for x in model_fine.parameters())))
print('DDR Settings: L_R = {}, Reg = {}, data ico = {}, cont ico = {}, lab ico = {}'
      .format(learning_rate, lambda_reg, data_ico, control_ico_coar, labels_ico_coar))

## start training ##
for epoch in range(Num_Epochs):

    running_losses = 0
    running_losses_mse = 0
    running_losses_cc = 0


    ## start training/epoch ##
    for moving_imgs, target_imgs in tqdm(train_dataloader):
        model_fine.train()
        moving_imgs, target_imgs = (moving_imgs.squeeze(0)).to(device), (target_imgs.squeeze(0)).to(device)

        #coarse_mse,coarse_cc,coarse_warp=Coarse_eval(moving_imgs, target_imgs, edge_in)
        '''
        coarse_warp: coarse deformed ico(upsample coarse control level(2) to fine control level(4)
        '''
        coarse_warp = Coarse_eval(moving_imgs, target_imgs, edge_in)

        optimizer.zero_grad()

        warped_moving_img, _, warps_moving = model_fine(moving_imgs, target_imgs, edge_in, coarse_warp)

        loss_mse, loss_cc = LossFuns(warped_moving_img, target_imgs)

        loss_g = grad_loss(warps_moving, hex_in, reg_fun=loss_pen, device=device)

        loss_total = (lambda_mse * loss_mse + lambda_cc * loss_cc) + lambda_reg * loss_g

        loss_total.requires_grad_(True)

        loss_total.backward()

        optimizer.step()

        running_losses += loss_total.item()
        running_losses_mse += loss_mse.item()
        running_losses_cc += loss_cc.item()



    ## done from training/epoch ##
    train_loss[epoch] = torch.tensor(running_losses / num_train_data)
    train_loss_mse[epoch] = torch.tensor(running_losses_mse / num_train_data)
    train_loss_cc[epoch] = torch.tensor(running_losses_cc / num_train_data)

    ## start validation ##
    val_losses, val_losses_mse = DDR_validation(val_dataloader, edge_in)
    val_loss[epoch] = torch.mean(val_losses, axis=0)
    val_loss_mse[epoch] = torch.mean(val_losses_mse, axis=0)

    ## print losses ##
    if (epoch + 1) % print_step == 0:
        print_during_training(epoch, trn_ls=train_loss[epoch].numpy()[0],
                              trn_cc=train_loss_cc[epoch].numpy()[0],
                              trn_mse=train_loss_mse[epoch].numpy()[0],
                              val_ls=val_loss[epoch].numpy()[0],
                              val_mse=val_loss_mse[epoch].numpy()[0])

    scheduler.step(val_loss[epoch])

    ## save model ? ##
    if m_checkpoint:
        torch.save(model_fine.state_dict(), save_model_dir + 'trained_coarse_to_fine_model.pkl')

        ## save the model if it works better on the val set ##
        if val_loss[epoch].numpy()[0] < best_val:
            best_val = val_loss[epoch].numpy()[0]
            torch.save(model_fine.state_dict(), save_model_dir + 'best_coarse_to_fine_model.pkl')
            print('-- New best model saved --')

print('Done from training.')

## start testing ##

if testing:
    print('\nStart testing...')

    test_model = DDR_fine(in_ch=in_channels, num_features=num_feat, data_ico=data_ico,
                            labels_ico=labels_ico_coar, control_ico=control_ico_coar,
                            num_labels=num_labels_coar, device=device)
    if not m_checkpoint:
        test_model.load_state_dict(model_fine.state_dict())

    del model_fine
    torch.cuda.empty_cache()

    test_model.to(device)

    test_dataset = MRIImages(moving_dir, target_dir, moving_suffix, target_prefix, target_suffix,
                             Id_file1=Id_file_test)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_worker, pin_memory=True)
    print('Number of Test Images = ', len(test_dataloader))

    print('Loading model...')

    if testing_best and m_checkpoint:
        test_model.load_state_dict(torch.load(save_model_dir + 'best_coarse_to_fine_model.pkl', map_location=device))
    elif m_checkpoint:
        test_model.load_state_dict(torch.load(save_model_dir + 'trained_coarse_to_fine_model.pkl', map_location=device))

    registered_feats, test_losses_mse, test_losses_cc, def_control_icos = DDR_testing(test_dataloader, edge_in)

    test_loss_mes = torch.mean(test_losses_mse, axis=0)
    test_loss_cc = torch.mean(test_losses_cc, axis=0)

    print('\n### Test Results ###')
    print('Test loss (MSE) = {:.5}'.format(test_loss_mes.numpy()[0]))
    print('Test acc (CC) = {:.5}'.format(1 - test_loss_cc.numpy()[0]))

    ## save deformed control icos in gifti formats ##

    if save_def_gifti:
        print('\nStart saving in gifti format...')

        test_subj_ids = open(Id_file_test, "r").read().splitlines()

        for idx, subj_id in enumerate(test_subj_ids):
            def_control_ico = def_control_icos[:, idx].reshape(ver_dic[control_ico_coar], 3).numpy()

            save_gifti(def_control_ico, subj_id, file_to_save=save_def_ico_dir)
