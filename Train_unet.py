import os
import errno
import torch
import timeit

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.functional as F
from torch.utils import data

from sklearn.metrics.pairwise import cosine_distances
from scipy import spatial
from sklearn.metrics import mean_squared_error
from torch.optim import lr_scheduler
from Loss import dice_loss, DiceLoss
from Utilis import segmentation_scores, dice_coef_custom, dice_coef_default, binary_dice_coefficient
from Utilis import DiceScore, dice_coef_simplified
from Utilis import CustomDataset, evaluate, test
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from Models import UNet, SkinNet, UNet_v3


def trainUnet(dataset_tag,
              dataset_name,
              data_directory,
              input_dim,
              class_no,
              repeat,
              train_batchsize,
              validate_batchsize,
              num_epochs,
              learning_rate,
              width,
              depth,
              augmentation='all_flip',
              loss_f='dice',
              path_name = './Results',
              labels_mode = 'avrg'):

    """ This is the panel to control the training of baseline U-net.

    Args:
        input_dim: channel number of input image, for example, 3 for RGB
        class_no: number of classes of classification
        repeat: repat the same experiments with different stochastic seeds, we normally run each experiment at least 3 times
        train_batchsize: training batch size, this depends on the GPU memory
        validate_batchsize: we normally set-up as 1
        num_epochs: training epoch length
        learning_rate:
        input_height: resolution of input image
        input_width: resolution of input image
        alpha: regularisation strength hyper-parameter
        width: channel number of first encoder in the segmentation network, for the standard U-net, it is 64
        depth: down-sampling stages of the segmentation network
        data_path: path to where you store your all of your data
        dataset_tag: 'mnist' for MNIST; 'brats' for BRATS 2018; 'lidc' for LIDC lung data set
        label_mode: 'multi' for multi-class of proposed method; 'p_unet' for baseline probabilistic u-net; 'normal' for binary on MNIST; 'binary' for general binary segmentation
        loss_f: 'noisy_label' for our noisy label function, or 'dice' for dice loss
        save_probability_map: if True, we save all of the probability maps of output of networks
        labels_mode: 'avrg' if majority_vote, 'staple' if STAPLE, ...

    Returns:

    """
    for j in range(1, repeat + 1):
        #
        Exp = UNet(in_ch=input_dim,
                   width=width,
                   depth=depth,
                   class_no=class_no,
                   norm='in',
                   dropout=True,
                   apply_last_layer=True)
        #

        Skin = SkinNet(in_ch = input_dim, width =width, depth = depth)
        # Skin = UNet_v3(n_channels = input_dim, n_classes = class_no)
        Exp_name = 'UNet' + '_width' + str(width) + \
                   '_depth' + str(depth) + \
                   '_repeat' + str(j) + '_' + labels_mode
        #
        # ====================================================================================================================================================================
        trainloader, validateloader, testloader, data_length = getData(data_directory, dataset_name, dataset_tag, train_batchsize, validate_batchsize, augmentation, labels_mode)
        # ==================
        trainSingleModel(Skin,
                         Exp_name,
                         num_epochs,
                         data_length,
                         learning_rate,
                         dataset_tag,
                         train_batchsize,
                         trainloader,
                         validateloader,
                         testloader,
                         losstag=loss_f,
                         class_no=class_no,
                         path_name = path_name)


def getData(data_directory, dataset_name, dataset_tag, train_batchsize, validate_batchsize, data_augment, labels_mode):
    #
    train_image_folder = data_directory + 'train/' + 'images'
    #'/' + dataset_name + '/' + \
    #    dataset_tag + '/train/patches'
    train_label_folder = data_directory + 'train/' + labels_mode
    #'/' + dataset_name + '/' + \
    #    dataset_tag + '/train/labels'
    validate_image_folder = data_directory + 'validate/' + 'images'
    #'/' + dataset_name + '/' + \
    #    dataset_tag + '/validate/patches'
    validate_label_folder = data_directory + 'validate/' + labels_mode
    # '/' + dataset_name + '/' + \
    #     dataset_tag + '/validate/labels'
    test_image_folder = data_directory + 'test/' + 'images'
    #'/' + dataset_name + '/' + \
    #    dataset_tag + '/test/patches'
    test_label_folder = data_directory + 'test/' + labels_mode
    #'/' + dataset_name + '/' + \
    #    dataset_tag + '/test/labels'
    #
    print(train_image_folder)
    print(train_label_folder)
    print(validate_image_folder)
    print(validate_label_folder)
    print(test_image_folder)
    print(test_label_folder)
    #
    train_dataset = CustomDataset(train_image_folder, train_label_folder, data_augment)
    #
    validate_dataset = CustomDataset(validate_image_folder, validate_label_folder, data_augment)
    #
    test_dataset = CustomDataset(test_image_folder, test_label_folder, 'none')
    #
    trainloader = data.DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=5, drop_last=True)
    #
    validateloader = data.DataLoader(validate_dataset, batch_size=validate_batchsize, shuffle=False, num_workers=validate_batchsize, drop_last=False)
    #
    testloader = data.DataLoader(test_dataset, batch_size=validate_batchsize, shuffle=False, num_workers=validate_batchsize, drop_last=False)
    #
    return trainloader, testloader, validateloader, len(train_dataset)

# =====================================================================================================================================


def trainSingleModel(model,
                     model_name,
                     num_epochs,
                     data_length,
                     learning_rate,
                     datasettag,
                     train_batchsize,
                     trainloader,
                     validateloader,
                     testdata,
                     losstag,
                     class_no,
                     path_name):
    # change log names
    iteration_amount = data_length // train_batchsize - 1
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    lr_str = str(learning_rate)
    #
    epoches_str = str(num_epochs)
    #
    save_model_name = model_name + '_' + datasettag + '_e' + epoches_str + '_lr' + lr_str
    #
    saved_information_path = path_name
    #
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    #
    saved_information_path = saved_information_path + '/Results_' + save_model_name
    #
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    saved_model_path = saved_information_path + '/trained_models'
    try:
        os.mkdir(saved_model_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    #
    print('The current model is:')
    #
    print(save_model_name)
    #
    print('\n')
    #
    writer = SummaryWriter(path_name + '/Log_' + datasettag + '/' + save_model_name)

    model_tl = True

    if model_tl:

        from collections import OrderedDict

        path_load_model = "./pretrained/Skin_model_d5.pt"
        def map_keys(loaded_state_dict):
            new_state_dict = OrderedDict()
            for k, v in loaded_state_dict.items():
                new_key = k                                 # Modify the key here based on the mismatch pattern
                new_state_dict[new_key] = v
            return new_state_dict

        loaded_state_dict = torch.load(path_load_model)
        modified_state_dict = map_keys(loaded_state_dict)
        model.load_state_dict(modified_state_dict, strict = False)
        model.eval()

        ### All parameters - GRAD ###
        for param in model.parameters():
            param.requires_grad = True
        ### ===================== ###

    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of params: ", total_params)
    total_params_grad  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of params with grad: ", total_params_grad)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    start = timeit.default_timer()

    for epoch in range(num_epochs):
        #
        model.train()
        running_loss = 0
        running_iou = 0
        #
        for j, (images, labels, imagename) in enumerate(trainloader):
            # check label values:
            # unique, counts = np.unique(labels, return_counts=True)
            # print(np.asarray((unique, counts)).T)
            #
            # print(imagename)
            # print("Images shape: ", images[0].cpu().detach().numpy().shape)
            # print("Labels shape: ", labels[0].cpu().detach().numpy().shape)
            # print(img_.max())            
            #
            optimizer.zero_grad()
            #
            images = images.to(device=device, dtype=torch.float32)

            labels = labels.to(device=device, dtype=torch.float32)

            outputs_logits = model(images)
            # print("Output shape: ", outputs_logits.cpu().detach().numpy().shape)
            #
            #
            if class_no == 2:
                #
                if losstag == 'dice':
                    #
                    loss = dice_loss(outputs_logits, labels)
                    #
                elif losstag == 'ce':
                    #
                    loss = nn.BCEWithLogitsLoss(reduction='mean')(outputs_logits, labels)
                    #
                elif losstag == 'hybrid':
                    #
                    loss = dice_loss(torch.sigmoid(outputs_logits), labels) + nn.BCEWithLogitsLoss(reduction='mean')(outputs_logits, labels)
            #
            else:
                #
                loss = nn.CrossEntropyLoss(reduction='mean')(torch.softmax(outputs_logits, dim=1), labels.squeeze(1))
                #
            loss.backward()
            optimizer.step()
            #
            if class_no == 2:
                outputs_logits = torch.sigmoid(outputs_logits)
            else:
                outputs_logits = torch.softmax(outputs_logits, dim=1)
            #
            # _, train_output = torch.max(outputs_logits, dim = 1)
            # print(imagename)
            # print("labels: ", labels.size())
            # print("logits: ", outputs_logits.size())
            # break
            # print("maxedl: ", train_output.size())
            # train_iou = segmentation_scores(labels.cpu().detach().numpy(), outputs_logits.cpu().detach().numpy(), class_no)
            # plt.imsave('./test_results/' + imagename[0] + '_segmented_max_0.png', train_output[0].cpu().detach().numpy(), cmap = 'gray')
            # plt.imsave('./test_results/' + imagename[0] + '_label_0.png', labels[0, 0].cpu().detach().numpy(), cmap = 'gray')
            # plt.imsave('./test_results/' + imagename[1] + '_segmented_max_1.png', train_output[1].cpu().detach().numpy(), cmap = 'gray')
            # plt.imsave('./test_results/' + imagename[1] + '_label_1.png', labels[1, 0].cpu().detach().numpy(), cmap = 'gray')
            # train_iou = dice_coef_simplified(outputs_logits, labels)
            train_iou = dice_coef_default(outputs_logits, labels)
            running_loss += loss
            running_iou += train_iou
            #
            if (j + 1) % iteration_amount == 0:
                #
                v_dice, v_loss = evaluate(validateloader, model, device, class_no=class_no)
                print(
                    'Step [{}/{}], '
                    'Train loss: {:.4f}, '
                    'Train dice: {:.4f}, '
                    'Validate loss: {:.4f},'
                    'Validate dice: {:.4f}, '.format(epoch + 1, num_epochs,
                                               running_loss / (j + 1),
                                               running_iou / (j + 1),
                                               v_loss / (j + 1),
                                               v_dice))
                # # # ================================================================== #
                # # #                        TensorboardX Logging                        #
                # # # # ================================================================ #
                writer.add_scalars('scalars', {'loss': running_loss / (j + 1),
                                               'train dice': running_iou / (j + 1),
                                               'validate dice': v_dice}, epoch + 1)
                #
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate*((1 - epoch / num_epochs)**0.999)
            #
    # test(testdata,
    #      model,
    #      device,
    #      class_no=class_no,
    #      save_path=saved_information_path)
    #
    # save model
    stop = timeit.default_timer()
    #
    print('Time: ', stop - start)
    #
    save_model_name_full = saved_model_path + '/' + save_model_name + '_Final.pt'
    #
    path_model = save_model_name_full
    #
    torch.save(model, path_model)
    torch.save(model.state_dict(), path_model + '_skin_Final_dict.pt')
    #
    print('\nTraining finished and model saved\n')
    #
    return model
