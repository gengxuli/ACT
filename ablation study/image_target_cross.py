import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList_idx
import random
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from torchvision.datasets import ImageFolder
from sam import SAM
from autoaugment import ImageNetPolicy
import torch.nn.functional as F
import utils
from torch.cuda.amp import autocast as autocast
from data_loader import InfiniteDataLoader
from loss import CrossEntropyLabelSmooth
import time
np.set_printoptions(threshold=sys.maxsize)

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        #param_group['weight_decay'] = 1e-3
        #param_group['momentum'] = 0.9
        #param_group['nesterov'] = True
    return optimizer

def adjust_learning_rate(optimizer, epoch,iter_num,iter_per_epoch=4762,lr=0.001):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    ratio=(epoch*iter_per_epoch+iter_num)/(10*iter_per_epoch)
    lr=lr*(1+10*ratio)**(-0.75)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def image_train(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    transforms_train = {
            "train": [None for i in range(2)]
        }
    transform_weak = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
    transform_strong = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                normalize
            ])
    transforms_train["train"][0] = transform_weak
    transforms_train["train"][1] = transform_strong
    
    
    return  transforms_train["train"]

def image_test(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])
    
def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    if not args.few_shot:
        txt_tar = open(args.t_dset_path).readlines()
        txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    #num_worker = min([os.cpu_count(), train_bs if train_bs > 1 else 0, 8])  # number of workers
    num_worker = 0
    print('Using {} dataloader workers'.format(num_worker))
    if args.few_shot:
        #target_dataset = ImageFolder(os.path.join(args.t_dset_path))
        #tr_idx, te_idx = utils.split_dataset(target_dataset, ratio=args.ratio, seed=args.seed)
        #dsets["target"], dsets["test"] = utils._SplitTrainDataset(target_dataset, tr_idx), utils._SplitTestDataset(target_dataset, te_idx)
        #dsets["target"].transform = image_train()
        #dsets["test"].transform = image_test()
        #keys = list(range(len(dsets["target"])))
        #targets = torch.tensor(dsets["target"].underlying_dataset.targets)[dsets["target"].keys]
        #assert len(keys) == len(targets)
        #few_shot_idx = utils.few_shot_subset(targets,args.few_shot)
        #dsets["target"] = torch.utils.data.Subset(dsets["target"],few_shot_idx)

        train_dataset = ImageFolder(os.path.join(args.t_dset_path))
        test_dataset = ImageFolder(os.path.join(args.test_dset_path))
        tr_idx = [i for i in range(len(train_dataset))]
        te_idx = [i for i in range(len(test_dataset))]
        random.shuffle(tr_idx)
        random.shuffle(te_idx)
        dsets["target"] = utils._SplitTrainDataset(train_dataset, tr_idx)
        dsets["test"] = utils._SplitTestDataset(test_dataset, te_idx)
        dsets["target"].transform = image_train()
        dsets["test"].transform = image_test()
        keys = list(range(len(dsets["target"])))
        targets = torch.tensor(dsets["target"].underlying_dataset.targets)[dsets["target"].keys]
        assert len(keys) == len(targets)
        few_shot_idx = utils.few_shot_subset(targets,args.few_shot)
        dsets["target"] = torch.utils.data.Subset(dsets["target"], few_shot_idx)
        te_idx = list(set(tr_idx).difference(set(few_shot_idx)))
        dsets["test"] = torch.utils.data.Subset(dsets["test"], te_idx)

    else:
        dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
        dsets["test"] = ImageList_idx(txt_test, transform=image_test())

    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=num_worker, drop_last=False)
    #dset_loaders["target"] = InfiniteDataLoader(dsets["target"], batch_size=train_bs, num_workers=num_worker)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=num_worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netC1, netC2, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs1 = netC1(netF(inputs))
            outputs2 = netC2(netF(inputs))
            outputs = outputs1 + outputs2
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent
    
def test(loader, netF, netC1, netC2, flag=False):
    test_loss = utils.AverageMeter()
    test_acc = utils.AverageMeter()
    # no grad
    with torch.no_grad():
        for i,(img, label) in enumerate(loader):
            img = img.cuda()
            label = label.cuda()
            #output1 = netC1(netB1(netF(img)))
            #output2 = netC2(netB2(netF(img)))
            output1 = netC1(netF(img))
            output2 = netC2(netF(img))
            output = output1 + output2
            loss = nn.CrossEntropyLoss()(output,label).cuda()
            if flag:
                # first
                if i == 0:
                    total_pred = output.data.max(1)[1]
                    total_label = label.detach().cpu()
                # else
                else:
                    total_pred = torch.cat(total_pred, output.data.max(1)[1])
                    total_label = torch.cat((total_label, label.detach().cpu()))
            # topk accuracy
            acc,_= utils.accuracy(output, label, topk=(1,5))
            test_acc.update(acc[0].item(), img.size(0))
            test_loss.update(loss.item(), img.size(0))
    if flag:
        # per class accuracy
        cm = confusion_matrix(total_label, total_pred)
        per_class_acc = cm.diagonal()/cm.sum(axis=1) * 100
        per_class_acc = per_class_acc.mean()            
        return test_loss.avg, test_acc.avg, per_class_acc
    else:
        return test_loss.avg, test_acc.avg

def loss_function_1(netF, netC1, netC2, inputs_test, src_log_output1, src_log_output2, tar_idx):
    gamma = 0.05
    eta = 0.0025
    beta = 0.1

    # Step A train all networks to minimize loss on source
    tgt_feature = netF(inputs_test)
    #tgt_output1, tgt_output2 = netC1(netB1(tgt_feature)), netC2(netB2(tgt_feature))
    tgt_output1, tgt_output2 = netC1(tgt_feature), netC2(tgt_feature)
    ent_loss1 = nn.CrossEntropyLoss()(tgt_output1, tar_idx).cuda()
    ent_loss2 = nn.CrossEntropyLoss()(tgt_output2, tar_idx).cuda()
    #ent_loss1 = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(tgt_output1, tar_idx).cuda()
    #ent_loss2 = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(tgt_output2, tar_idx).cuda()
    tgt_log_output1, tgt_log_output2 = F.softmax(tgt_output1, dim=1), F.softmax(tgt_output2, dim=1)
    entropy_loss = -torch.mean(torch.log(torch.mean(tgt_log_output1, 0) + 1e-6))
    entropy_loss -= torch.mean(torch.log(torch.mean(tgt_log_output2, 0) + 1e-6))

    loss1 = utils.soft_criterion(src_log_output1, tgt_log_output1)
    loss2 = utils.soft_criterion(src_log_output2, tgt_log_output2)
    
    stepA_loss = beta * (loss1 + loss2) + gamma * entropy_loss + ent_loss1 + ent_loss2
    #stepA_loss = beta * (loss1 + loss2) + 0.01 * entropy_loss + ent_loss1 + ent_loss2
    return stepA_loss

def loss_function_2(netF, netC1, netC2, inputs_test, src_log_output1, src_log_output2, tar_idx):
    gamma = 0.05
    eta = 0.0025
    beta = 0.1

    # Step B train classifier to maximize discrepancy
    tgt_feature = netF(inputs_test)
    #tgt_output1, tgt_output2 = netC1(netB1(tgt_feature)), netC2(netB2(tgt_feature))
    tgt_output1, tgt_output2 = netC1(tgt_feature), netC2(tgt_feature)
    ent_loss1 = nn.CrossEntropyLoss()(tgt_output1, tar_idx).cuda()
    ent_loss2 = nn.CrossEntropyLoss()(tgt_output2, tar_idx).cuda()
    #ent_loss1 = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(tgt_output1, tar_idx).cuda()
    #ent_loss2 = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(tgt_output2, tar_idx).cuda()
    tgt_log_output1, tgt_log_output2 = F.softmax(tgt_output1, dim=1), F.softmax(tgt_output2, dim=1)
    entropy_loss = -torch.mean(torch.log(torch.mean(tgt_log_output1, 0) + 1e-6))
    entropy_loss -= torch.mean(torch.log(torch.mean(tgt_log_output2, 0) + 1e-6))

    loss1 = utils.soft_criterion(src_log_output1, tgt_log_output1)
    loss2 = utils.soft_criterion(src_log_output2, tgt_log_output2)

    cdd_dist = utils.cdd(tgt_log_output1, tgt_log_output2)

    stepB_loss = beta * (loss1 + loss2) - eta * cdd_dist + gamma * entropy_loss + ent_loss1 + ent_loss2
    #stepB_loss = beta * (loss1 + loss2) - 0.01 * cdd_dist + 0.01 * entropy_loss + ent_loss1 + ent_loss2
    return stepB_loss

def loss_function_3(netF, netC1, netC2, inputs_test):
    gamma = 0.05
    eta = 0.0025
    beta = 0.1

    # Step C train genrator to minimize discrepancy
    tgt_feature = netF(inputs_test)
    #tgt_output1, tgt_output2 = netC1(netB1(tgt_feature)), netC2(netB2(tgt_feature))
    tgt_output1, tgt_output2 = netC1(tgt_feature), netC2(tgt_feature)
    tgt_log_output1, tgt_log_output2 = F.softmax(tgt_output1, dim=1), F.softmax(tgt_output2, dim=1)
    entropy_loss = -torch.mean(torch.log(torch.mean(tgt_log_output1, 0) + 1e-6))
    entropy_loss -= torch.mean(torch.log(torch.mean(tgt_log_output2, 0) + 1e-6))
    cdd_dist = utils.cdd(tgt_log_output1, tgt_log_output2)

    # stepC_loss = eta * cdd_dist + gamma * entropy_loss

    # result
    stepC_loss = 0.01*cdd_dist + 0.01*entropy_loss

    return stepC_loss

def loss_function_4(netF, netB1, netC1, netB2, netC2, inputs_test, tar_idx, args):
    features1 = netB1(netF(inputs_test))
    outputs1 = netC1(features1)

    features2 = netB2(netF(inputs_test))
    outputs2 = netC2(features2)

    classifier_loss1 = nn.CrossEntropyLoss()(outputs1, tar_idx)
    classifier_loss2 = nn.CrossEntropyLoss()(outputs2, tar_idx)

    classifier_loss = classifier_loss1 + classifier_loss2

    softmax_out1 = nn.Softmax(dim=1)(outputs1)
    list_svd1,_ = torch.sort(torch.sqrt(torch.sum(torch.pow(softmax_out1,2),dim=0)), descending=True)
    fbnm_loss1 = - torch.mean(list_svd1[:min(softmax_out1.shape[0],softmax_out1.shape[1])])

    softmax_out2 = nn.Softmax(dim=1)(outputs2)
    list_svd2,_ = torch.sort(torch.sqrt(torch.sum(torch.pow(softmax_out2,2),dim=0)), descending=True)
    fbnm_loss2 = - torch.mean(list_svd2[:min(softmax_out2.shape[0],softmax_out2.shape[1])])

    fbnm_loss = fbnm_loss1 + fbnm_loss2

    if args.ent:
        softmax_out1 = nn.Softmax(dim=1)(outputs1)
        entropy_loss1 = torch.mean(loss.Entropy(softmax_out1))
        if args.gent:
            msoftmax1 = softmax_out1.mean(dim=0)
            gentropy_loss1 = torch.sum(-msoftmax1 * torch.log(msoftmax1 + args.epsilon))
            entropy_loss1 -= gentropy_loss1
        im_loss1 = entropy_loss1 * args.ent_par

    if args.ent:
        softmax_out2 = nn.Softmax(dim=1)(outputs2)
        entropy_loss2 = torch.mean(loss.Entropy(softmax_out2))
        if args.gent:
            msoftmax2 = softmax_out2.mean(dim=0)
            gentropy_loss2 = torch.sum(-msoftmax2 * torch.log(msoftmax2 + args.epsilon))
            entropy_loss2 -= gentropy_loss2
        im_loss2 = entropy_loss2 * args.ent_par

    im_loss = im_loss1 + im_loss2

    total_loss = classifier_loss + fbnm_loss + im_loss

    return total_loss

def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net)
        src_netF = network.ResBase(res_name=args.net)
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net)
        src_netF = network.ResBase(res_name=args.net)
    elif args.net[0:4] == 'deit':
        if args.net == 'deit_s':
            netF = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)
            src_netF = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)
        elif args.net == 'deit_b':
            netF = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_384', pretrained=True)
            src_netF = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_384', pretrained=True)
        netF.in_features = 1000
        src_netF.in_features = 1000

    netF = netF.cuda()
    src_netF = src_netF.cuda()

    #netB1 = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    #netC1 = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    #netB2 = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    #netC2 = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    #src_netB1 = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    #src_netC1 = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    #src_netB2 = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    #src_netC2 = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    netC1 = network.Classifier(feature_dim=netF.dim, bottleneck_dim=args.bottleneck, class_num=args.class_num).cuda()
    netC2 = network.Classifier(feature_dim=netF.dim, bottleneck_dim=args.bottleneck, class_num=args.class_num).cuda()
    #netD = network.AdversarialNetwork(in_feature=netF.dim, hidden_size=1024).cuda()

    # new
    netC1.apply(network.init_weights)
    netC2.apply(network.init_weights)

    src_netC1 = network.Classifier(feature_dim=netF.dim, bottleneck_dim=args.bottleneck, class_num=args.class_num).cuda()
    src_netC2 = network.Classifier(feature_dim=netF.dim, bottleneck_dim=args.bottleneck, class_num=args.class_num).cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    src_netF.load_state_dict(torch.load(args.modelpath))

    args.modelpath = args.output_dir_src + '/source_C1.pt'   
    netC1.load_state_dict(torch.load(args.modelpath))
    src_netC1.load_state_dict(torch.load(args.modelpath))

    args.modelpath = args.output_dir_src + '/source_C2.pt'   
    netC2.load_state_dict(torch.load(args.modelpath))
    src_netC2.load_state_dict(torch.load(args.modelpath))

    #args.modelpath = args.output_dir_src + '/source_D.pt'   
    #netD.load_state_dict(torch.load(args.modelpath))

    #args.modelpath = args.output_dir_src + '/source_B1.pt'   
    #netB1.load_state_dict(torch.load(args.modelpath))
    #src_netB1.load_state_dict(torch.load(args.modelpath))

    #args.modelpath = args.output_dir_src + '/source_B2.pt'   
    #netB2.load_state_dict(torch.load(args.modelpath))
    #src_netB2.load_state_dict(torch.load(args.modelpath)) 

    
    if torch.cuda.device_count() > 1:
        gpu_list = []
        for i in range(len(args.gpu_id.split(','))):
            gpu_list.append(i)
        print("Let's use", len(gpu_list), "GPUs!")
		# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        netF = nn.DataParallel(netF, device_ids=gpu_list)
        netC1 = nn.DataParallel(netC1, device_ids=gpu_list)
        netC2 = nn.DataParallel(netC2, device_ids=gpu_list)
        #netD = nn.DataParallel(netD, device_ids=gpu_list)
        #netB1 = nn.DataParallel(netB1, device_ids=gpu_list)
        #netB2 = nn.DataParallel(netB2, device_ids=gpu_list)

        src_netF = nn.DataParallel(src_netF, device_ids=gpu_list)
        src_netC1 = nn.DataParallel(src_netC1, device_ids=gpu_list)
        src_netC2 = nn.DataParallel(src_netC2, device_ids=gpu_list)
        #src_netB1 = nn.DataParallel(src_netB1, device_ids=gpu_list)
        #src_netB2 = nn.DataParallel(src_netB2, device_ids=gpu_list)

    #param_group_f = []
    #for k, v in netF.named_parameters():
        #if args.lr_decay1 > 0:
            #param_group_f += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        #else:
            #v.requires_grad = False
    
    #param_group_c = []
    #for k, v in netB1.named_parameters():
        #if args.lr_decay2 > 0:
            #param_group_c += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        #else:
            #v.requires_grad = False
    #for k, v in netB2.named_parameters():
        #if args.lr_decay2 > 0:
            #param_group_c += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        #else:
            # v.requires_grad = False
    #for k, v in netC1.named_parameters():
        #if args.lr_decay3 > 0:
            #param_group_c += [{'params': v, 'lr': args.lr * args.lr_decay3}]
        #else:
            #v.requires_grad = False
    #for k, v in netC2.named_parameters():
        #if args.lr_decay3 > 0:
            #param_group_c += [{'params': v, 'lr': args.lr * args.lr_decay3}]
        #else:
            #v.requires_grad = False

    #param_group_all = []
    #for k, v in netF.named_parameters():
        #if args.lr_decay1 > 0:
            #param_group_all += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        #else:
            #v.requires_grad = False
    #for k, v in netB1.named_parameters():
        #if args.lr_decay2 > 0:
            #param_group_all += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        #else:
            #v.requires_grad = False
    #for k, v in netB2.named_parameters():
        #if args.lr_decay2 > 0:
            #param_group_all += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        #else:
            #v.requires_grad = False

    param_group_f = list(netF.parameters())
    param_group_c = list(netC1.parameters()) + list(netC2.parameters())
    #param_group_d = list(netD.parameters())
    #param_group_c = list(netB1.parameters()) + list(netB2.parameters()) + list(netC1.parameters()) + list(netC2.parameters())
    #param_group_all = list(netF.parameters()) + list(netB1.parameters()) + list(netB2.parameters())

    for k, v in src_netF.named_parameters():
        v.requires_grad = False
    for k, v in src_netC1.named_parameters():
        v.requires_grad = False
    for k, v in src_netC2.named_parameters():
        v.requires_grad = False
    #for k, v in src_netB1.named_parameters():
        #v.requires_grad = False
    #for k, v in src_netB2.named_parameters():
        #v.requires_grad = False

    if args.SAM:
        if args.optim == 'adam':
            optimizer_f = SAM(param_group_f, torch.optim.Adam, lr=args.lr, weight_decay=args.wd, rho=args.rho)
            optimizer_c = SAM(param_group_c, torch.optim.Adam, lr=args.lr, weight_decay=args.wd * 10 / 3, rho=args.rho)
            #optimizer_d = SAM(param_group_d, torch.optim.Adam, lr=args.lr, weight_decay=args.wd, rho=args.rho)
            #optimizer_all = SAM(param_group_all, torch.optim.Adam, lr=args.lr, weight_decay=args.wd, rho=args.rho)
        else:
            optimizer_f = SAM(param_group_f, torch.optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True, rho=args.rho)
            optimizer_c = SAM(param_group_c, torch.optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True, rho=args.rho)
            #optimizer_all = SAM(param_group_all, torch.optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True, rho=args.rho)
    else:
        if args.optim == 'adam':
            optimizer_f = torch.optim.Adam(param_group_f, lr=args.lr, weight_decay=args.wd)
            optimizer_c = torch.optim.Adam(param_group_c, lr=args.lr, weight_decay=args.wd)
            #optimizer_all = torch.optim.Adam(param_group_all, lr=args.lr, weight_decay=args.wd)
        else:
            optimizer_f = torch.optim.SGD(param_group_f, lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True)
            optimizer_c = torch.optim.SGD(param_group_c, lr=args.lr * 10 / 3, weight_decay=args.wd, momentum=0.9, nesterov=True)
            #optimizer_d = torch.optim.SGD(param_group_d, lr=args.lr, momentum=0.9, weight_decay=args.wd)
            #optimizer_all = torch.optim.SGD(param_group_all, lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True)

    optimizer_f = op_copy(optimizer_f)
    optimizer_c = op_copy(optimizer_c)
    #optimizer_d = op_copy(optimizer_d)
    #optimizer_all = op_copy(optimizer_all)

    max_epoch = args.max_iter // (args.few_shot * args.class_num)
    #interval_iter = max_epoch // args.interval
    interval_iter = 1
    best_acc = 0
    best_epoch = 0
    num_k = args.num_k

    for epoch_num in range(max_epoch + 1):
        #since = time.time()
        #iter_test = iter(dset_loaders["target"])
        #if args.few_shot:
            #inputs_w, inputs_s, tar_idx = next(iter_test)
        #else:
            #inputs_w, inputs_s, _, tar_idx = next(iter_test)
        src_netF.eval()
        src_netC1.eval()
        src_netC2.eval()
        #src_netB1.eval()
        #src_netB2.eval()

        for step, (inputs_w, inputs_s, tar_idx) in enumerate(dset_loaders["target"]):
            netF.train()
            netC1.train()
            netC2.train()
            #netD.train()
            #netB1.train()
            #netB2.train()

            lr_scheduler(optimizer_f, iter_num=epoch_num, max_iter=max_epoch)
            lr_scheduler(optimizer_c, iter_num=epoch_num, max_iter=max_epoch)
            #lr_scheduler(optimizer_all, iter_num=epoch_num, max_iter=max_epoch)

            #sadjust_learning_rate(optimizer_f, epoch_num // args.interval, step, len(dset_loaders["target"]), 0.003) 
            #adjust_learning_rate(optimizer_c, epoch_num // args.interval, step, len(dset_loaders["target"]), 0.01)

            #result
            #adjust_learning_rate(optimizer_c, epoch_num, step, len(dset_loaders["target"]), 0.01)
                
            inputs_w = inputs_w.cuda()
            inputs_s = inputs_s.cuda()
            tar_idx = tar_idx.cuda()

            inputs_test = torch.cat([inputs_w, inputs_s], dim=0)
            tar_idx = torch.cat([tar_idx, tar_idx], dim=0)


            
            with torch.no_grad():
                src_feature = src_netF(inputs_test)
                #src_output1, src_output2 = src_netC1(src_netB1(src_feature)), src_netC2(src_netB2(src_feature))
                src_output1, src_output2 = src_netC1(src_feature), src_netC2(src_feature)
                src_log_output1, src_log_output2 = F.softmax(src_output1, dim=1), F.softmax(src_output2, dim=1)
                    
            loss_1 = loss_function_1(netF, netC1, netC2, inputs_test, src_log_output1, src_log_output2, tar_idx)

            if args.SAM:
                loss_1.backward()
                optimizer_f.first_step(zero_grad=True)
                optimizer_c.first_step(zero_grad=True)
            
                loss_function_1(netF, netC1, netC2, inputs_test, src_log_output1, src_log_output2, tar_idx).backward()
                optimizer_f.second_step(zero_grad=True)
                optimizer_c.second_step(zero_grad=True)

            else:
                optimizer_f.zero_grad()
                optimizer_c.zero_grad()
                loss_1.backward()
                optimizer_f.step()
                optimizer_c.step()

            loss_2 = loss_function_2(netF, netC1, netC2, inputs_test, src_log_output1, src_log_output2, tar_idx)

            if args.SAM:
                loss_2.backward()
                optimizer_c.first_step(zero_grad=True)
            
                loss_function_2(netF, netC1, netC2, inputs_test, src_log_output1, src_log_output2, tar_idx).backward()
                optimizer_c.second_step(zero_grad=True)

            else:
                optimizer_f.zero_grad()
                optimizer_c.zero_grad()
                loss_2.backward()
                optimizer_c.step()

            for i in range(num_k):
                loss_3 = loss_function_3(netF, netC1, netC2, inputs_test)
                if args.SAM:
                    loss_3.backward()
                    optimizer_f.first_step(zero_grad=True)
            
                    loss_function_3(netF, netC1, netC2, inputs_test).backward()
                    optimizer_f.second_step(zero_grad=True)

                else:
                    optimizer_f.zero_grad()
                    optimizer_c.zero_grad()
                    loss_3.backward()
                    optimizer_f.step()

            #loss_4 = loss_function_4(netF, netB1, netC1, netB2, netC2, inputs_test, tar_idx, args)
                
            #if args.SAM:
                #loss_4.backward()
                #optimizer_all.first_step(zero_grad=True)
            
                #loss_function_4(netF, netB1, netC1, netB2, netC2, inputs_test, tar_idx, args).backward()
                #optimizer_all.second_step(zero_grad=True)

            #else:
                #optimizer_all.zero_grad()
                #loss_4.backward()
                #optimizer_all.step()

        if epoch_num % interval_iter == 0 or epoch_num == max_epoch:
            netF.eval()
            netC1.eval()
            netC2.eval()
            #netB1.eval()
            #netB2.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netC1, netC2, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, epoch_num, max_epoch, acc_s_te) + '\n' + acc_list

            else:
                test_loss, test_acc = test(dset_loaders['test'], netF, netC1, netC2, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%, Loss = {:.4f}'.format(args.name, epoch_num, max_epoch, test_acc, test_loss)
                
            if test_acc >= best_acc:
                best_acc = test_acc
                best_epoch = epoch_num
                print('Save the model with acc:', best_acc)
                print('Save the model with epoch:', best_epoch)
                best_netF = netF.state_dict()
                best_netC1 = netC1.state_dict()
                best_netC2 = netC2.state_dict()
                #best_netD = netD.state_dict()
                #best_netB1 = netB1.state_dict()
                #best_netB2 = netB2.state_dict()

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netC1.train()
            netC2.train()
            #netB1.train()
            #netB2.train()
        
        #print('time:', time.time() - since)

    torch.save(best_netF, osp.join(args.output_dir, "target_F_" + ".pt"))
    torch.save(best_netC1, osp.join(args.output_dir, "target_C1_" + ".pt"))
    torch.save(best_netC2, osp.join(args.output_dir, "target_C2_" + ".pt"))
    #torch.save(best_netD, osp.join(args.output_dir, "target_D_" + ".pt"))
    #torch.save(best_netB1, osp.join(args.output_dir, "target_B1_" + ".pt"))
    #torch.save(best_netB2, osp.join(args.output_dir, "target_B2_" + ".pt"))

        
    log_str = 'Best Accuracy = {:.2f}%'.format(best_acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return netF, netC1, netC2

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT_few_shot')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_iter', type=int, default=30000, help="max iterations")
    parser.add_argument('--interval', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=32, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=3*1e-5, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101, deit")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=1000)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--few_shot', default=None,type=int, help='adapt for a few images')
    parser.add_argument('--SAM', action='store_true', default=False, help='Use Sharpness aware minimization')
    parser.add_argument('--rho', default=0.05, type=float, help='SAM rho')
    parser.add_argument('--wd', default=0, type=float, help='Weight decay')
    parser.add_argument('--optim', default='adam', type=str, choices=['adam','sgd'],help='Optimizer for adaptation')
    parser.add_argument('--lr_decay1', type=float, default=1.0)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--lr_decay3', type=float, default=1.0, help="fix the classifier layer")
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--input_src', type=str, default='src_train', help='Load SRC training wt path')
    parser.add_argument('--output', type=str, default='SFFS_weights', help='Save ur weights here')
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--num_k', type=int, default=0, metavar='K', help='how many steps to repeat the generator update')
    parser.add_argument('--ratio', default=0.2, type=float, help='Holdout ratio for target test set')
    parser.add_argument('--src_seed', type=int, default=0, help="random seed")
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset =='domain_net':
        names = ['clipart', 'infograph', 'painting', 'quickdraw','sketch', 'real']
        args.class_num = 345
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        if args.few_shot:
            args.t_dset_path = './' + args.dset + '/' + names[args.t]
            args.test_dset_path = './' + args.dset + '/' + names[args.t]
        else:
            args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
            args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.input_src, args.da, args.dset, str(args.src_seed), names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, str(args.src_seed), names[args.s][0].upper() + names[args.t][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.out_file = open(osp.join(args.output_dir, 'log.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args)
        args.out_file.close()