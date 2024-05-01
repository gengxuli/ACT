import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
import loss
from torch.utils.data import DataLoader
from data_list import ImageList
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sam import SAM
import time

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

def image_train(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
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
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i
        
        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                new_src.append(line)
        txt_src = new_src.copy()

        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_test = new_tar.copy()

    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

    return dset_loaders

def data_load_test(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_test = open(args.test_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_test = new_tar.copy()

    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netC1, netC2, flag=False):
    start_test = True
    with torch.no_grad():
        #iter_test = iter(loader)
        #for i in range(len(loader)):
        for step, (inputs, labels) in enumerate(loader):
            #data = iter_test.next()
            #inputs = data[0]
            #labels = data[1]
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

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int),:]
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent
    
def cal_acc_oda(loader, netF, netC1, netC2):
    start_test = True
    with torch.no_grad():
        #iter_test = iter(loader)
        #for i in range(len(loader)):
        for step, (inputs, labels) in enumerate(loader):
            #data = iter_test.next()
            #inputs = data[0]
            #labels = data[1]
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

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()
    initc = np.array([[0], [1]])
    kmeans = KMeans(n_clusters=2, random_state=0, init=initc, n_init=1).fit(ent.reshape(-1,1))
    threshold = (kmeans.cluster_centers_).mean()

    predict[ent>threshold] = args.class_num
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int),:]

    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    unknown_acc = acc[-1:].item()

    return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    # return np.mean(acc), np.mean(acc[:-1])

def loss_function(netF, netC1, netC2, inputs_source, labels_source):
    outputs_source1 = netC1(netF(inputs_source))
    outputs_source2 = netC2(netF(inputs_source))
    classifier_loss1 = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source1, labels_source).cuda()
    classifier_loss2 = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source2, labels_source).cuda()
    #classifier_loss1 = nn.CrossEntropyLoss()(outputs_source1, labels_source).cuda()
    #classifier_loss2 = nn.CrossEntropyLoss()(outputs_source2, labels_source).cuda()
    classifier_loss = classifier_loss1 + classifier_loss2

    return classifier_loss

def train_source(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
    elif args.net[0:4] == 'deit':
        if args.net == 'deit_s':
            netF = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).cuda()
        elif args.net == 'deit_b':
            netF = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_384', pretrained=True).cuda()
        netF.in_features = 1000  

    #netC1 = network.Classifier(feature_dim=netF.dim, bottleneck_dim=args.bottleneck, class_num=args.class_num).cuda()
    #netC2 = network.Classifier(feature_dim=netF.dim, bottleneck_dim=args.bottleneck, class_num=args.class_num).cuda()

    # 101
    netC1 = network.Classifier(feature_dim=netF.in_features, bottleneck_dim=args.bottleneck, class_num=args.class_num).cuda()
    netC2 = network.Classifier(feature_dim=netF.in_features, bottleneck_dim=args.bottleneck, class_num=args.class_num).cuda()

    #if args.net == 'resnet50':
        #netC1.apply(network.init_weights)
        #netC2.apply(network.init_weights)

    if torch.cuda.device_count() > 1:
        gpu_list = []
        for i in range(len(args.gpu_id.split(','))):
            gpu_list.append(i)
        print("Let's use", len(gpu_list), "GPUs!")
		# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        netF = nn.DataParallel(netF, device_ids=gpu_list)
        netC1 = nn.DataParallel(netC1, device_ids=gpu_list)
        netC2 = nn.DataParallel(netC2, device_ids=gpu_list)
          
    param_group_f = list(netF.parameters())
    param_group_c = list(netC1.parameters()) + list(netC2.parameters())

    if args.SAM:
        if args.optim == 'adam':
            optimizer_f = SAM(param_group_f, torch.optim.Adam, lr=args.lr, weight_decay=args.wd, rho=args.rho)
            optimizer_c = SAM(param_group_c, torch.optim.Adam, lr=args.lr, weight_decay=args.wd, rho=args.rho)
        else:
            optimizer_f = SAM(param_group_f, torch.optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True, rho=args.rho)
            optimizer_c = SAM(param_group_c, torch.optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True, rho=args.rho)
    else:
        if args.optim == 'adam':
            optimizer_f = torch.optim.Adam(param_group_f, lr=args.lr, weight_decay=args.wd)
            optimizer_c = torch.optim.Adam(param_group_c, lr=args.lr, weight_decay=args.wd)
        else:
            optimizer_f = torch.optim.SGD(param_group_f, lr=args.lr, weight_decay=args.wd, momentum=0.9)
            optimizer_c = torch.optim.SGD(param_group_c, lr=args.lr, weight_decay=args.wd, momentum=0.9)

    optimizer_f = op_copy(optimizer_f)
    optimizer_c = op_copy(optimizer_c)

    acc_best = 0
    max_epoch = args.max_epoch
    interval_iter = max_epoch // 10
    iter_num = 0


    for iter_num in range(max_epoch + 1):
        print("iter:{}, max_iter:{}".format(iter_num, max_epoch))
        #try:
            #inputs_source, labels_source = iter_source.next()
        #except:
            #iter_source = iter(dset_loaders["source_tr"])
            #inputs_source, labels_source = iter_source.next()
        dset_loaders["source_tr"] = tqdm(dset_loaders["source_tr"])  
        for step, (inputs_source, labels_source) in enumerate(dset_loaders["source_tr"]):
            netF.train() 
            netC1.train()
            netC2.train()

            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            #lr_scheduler(optimizer_f, iter_num=iter_num, max_iter=max_epoch)
            #lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_epoch)

            loss = loss_function(netF, netC1, netC2, inputs_source, labels_source)

            if args.SAM:
                loss.backward()
                optimizer_f.first_step(zero_grad=True)
                optimizer_c.first_step(zero_grad=True)
            
                loss_function(netF, netC1, netC2, inputs_source, labels_source).backward()
                optimizer_f.second_step(zero_grad=True)
                optimizer_c.second_step(zero_grad=True)

            else:                
                optimizer_f.zero_grad()
                optimizer_c.zero_grad()
                loss.backward()
                optimizer_f.step()
                optimizer_c.step()


        if iter_num % interval_iter == 0 or iter_num == max_epoch:
            test_acc = 0
            netF.eval()
            netC1.eval()
            netC2.eval()
            dset_loaders['source_te'] = tqdm(dset_loaders['source_te'])
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], netF, netC1, netC2, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_epoch, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netC1, netC2, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_epoch, acc_s_te)


            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            for i in range(len(names)):
                if i == args.s:
                    continue
                args.t = i
                args.name = names[args.s][0].upper() + names[args.t][0].upper()

                args.test_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

                if args.dset == 'office-home':
                    if args.da == 'pda':
                        args.class_num = 65
                        args.src_classes = [i for i in range(65)]
                        args.tar_classes = [i for i in range(25)]
                    if args.da == 'oda':
                        args.class_num = 25
                        args.src_classes = [i for i in range(25)]
                        args.tar_classes = [i for i in range(65)]

                if args.dset == 'office':
                    if args.da == 'pda':
                        args.class_num = 31
                        args.src_classes = [i for i in range(31)]
                        args.tar_classes = [i for i in range(10)]
                    if args.da == 'oda':
                        args.class_num = 10
                        args.src_classes = [i for i in range(10)]
                        args.tar_classes = [i for i in range(31)]

                if args.dset == 'VISDA-C':
                    if args.da == 'pda':
                        args.class_num = 12
                        args.src_classes = [i for i in range(12)]
                        args.tar_classes = [i for i in range(6)]
                    if args.da == 'oda':
                        args.class_num = 6
                        args.src_classes = [i for i in range(6)]
                        args.tar_classes = [i for i in range(12)]

                if args.da == 'oda':
                    print(args.src_classes)
                
                dset_loaders_test = data_load_test(args)
                dset_loaders_test['test'] = tqdm(dset_loaders_test['test'])
                if args.da == 'oda':
                    if args.dset=='VISDA-C':
                        acc_os1, acc_os2, acc_unknown, acc_list = cal_acc_oda(dset_loaders_test['test'], netF, netC1, netC2)
                        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.name, acc_os2, acc_os1, acc_unknown) + '\n' + acc_list

                    else:
                        acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders_test['test'], netF, netC1, netC2)
                        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.name, acc_os2, acc_os1, acc_unknown)
                else:
                    if args.dset=='VISDA-C':
                        acc, acc_list = cal_acc(dset_loaders_test['test'], netF, netC1, netC2, True)
                        log_str = '\nTask: {}, Accuracy = {:.2f}%'.format(args.name, acc) + '\n' + acc_list
                    else:
                        acc, _ = cal_acc(dset_loaders_test['test'], netF, netC1, netC2, False)
                        log_str = '\nTask: {}, Accuracy = {:.2f}%'.format(args.name, acc)

                if args.da == 'oda':
                    test_acc += acc_os2
                else:
                    test_acc += acc
                print("test_acc:{}".format(test_acc))
                args.out_file.write(log_str + '\n')
                args.out_file.flush()
                print(log_str+'\n')
            
            print("acc_best:{}".format(acc_best))
            if test_acc >= acc_best:
                acc_best = test_acc
                print('Save the model with best test acc:', acc_best)
                best_netF = netF.state_dict()
                best_netC1 = netC1.state_dict()
                best_netC2 = netC2.state_dict()

            netF.train()
            netC1.train()
            netC2.train()
                
    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netC1, osp.join(args.output_dir_src, "source_C1.pt"))
    torch.save(best_netC2, osp.join(args.output_dir_src, "source_C2.pt"))

    log_str = 'Best Accuracy = {:.2f}%'.format(acc_best)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return netF, netC1, netC2

def test_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
    elif args.net[0:4] == 'deit':
        if args.net == 'deit_s':
            netF = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).cuda()
        elif args.net == 'deit_b':
            netF = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_384', pretrained=True).cuda()
        netF.in_features = 1000

    netC1 = network.Classifier(feature_dim=netF.dim, bottleneck_dim=args.bottleneck, class_num=args.class_num).cuda()
    netC2 = network.Classifier(feature_dim=netF.dim, bottleneck_dim=args.bottleneck, class_num=args.class_num).cuda()

    #101
    #netC1 = network.Classifier(feature_dim=netF.in_features, bottleneck_dim=args.bottleneck, class_num=args.class_num).cuda()
    #netC2 = network.Classifier(feature_dim=netF.in_features, bottleneck_dim=args.bottleneck, class_num=args.class_num).cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C1.pt'   
    netC1.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C2.pt'   
    netC2.load_state_dict(torch.load(args.modelpath))

    netF.eval()
    netC1.eval()
    netC2.eval()

    if args.dset=='VISDA-C':
        acc, acc_list = cal_acc(dset_loaders['test'], netF, netC1, netC2, True)
        log_str = '\nTask: {}, Accuracy = {:.2f}%'.format(args.name, acc) + '\n' + acc_list
    else:
        acc, _ = cal_acc(dset_loaders['test'], netF, netC1, netC2, False)
        log_str = '\nTask: {}, Accuracy = {:.2f}%'.format(args.name, acc)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT++')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=256, help="batch_size")
    parser.add_argument('--worker', type=int, default=8, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'VLCS'])
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet18, resnet34, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='src_train')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--SAM', action='store_true', default=False, help='Use Sharpness aware minimization')
    parser.add_argument('--optim', default='sgd', type=str, choices=['adam','sgd'],help='Optimizer for adaptation')
    parser.add_argument('--wd', default=0.0005, type=float, help='Weight decay')
    parser.add_argument('--rho', default=0.05, type=float, help='SAM rho')
    parser.add_argument('--oda_seed', default=2020,type=int,help='Known class selection seed for OoD scenario')
    args = parser.parse_args()

    args.worker = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print(args.worker)

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
        args.lr = 1e-3
    if args.dset == 'VLCS':
        names = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
        args.class_num = 5

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

    folder = './data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]
        if args.da == 'oda':
            args.class_num = 25
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]

    if args.dset == 'office':
        if args.da == 'pda':
            args.class_num = 31
            args.src_classes = [i for i in range(31)]
            args.tar_classes = [i for i in range(10)]
        if args.da == 'oda':
            args.class_num = 10
            args.src_classes = [i for i in range(10)]
            args.tar_classes = [i for i in range(31)]

    if args.dset == 'VISDA-C':
        if args.da == 'pda':
            args.class_num = 12
            args.src_classes = [i for i in range(12)]
            args.tar_classes = [i for i in range(6)]
        if args.da == 'oda':
            args.class_num = 6
            args.src_classes = [i for i in range(6)]
            args.tar_classes = [i for i in range(12)]

    args.output_dir_src = osp.join(args.output, args.da, args.dset, str(args.seed), names[args.s][0].upper())
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_source(args)

    #args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
    #for i in range(len(names)):
        #if i == args.s:
            #continue
        #args.t = i
        #args.name = names[args.s][0].upper() + names[args.t][0].upper()

        #args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        #args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        #if args.dset == 'office-home':
            #if args.da == 'pda':
                #args.class_num = 65
                #args.src_classes = [i for i in range(65)]
                #args.tar_classes = [i for i in range(25)]
            #if args.da == 'oda':
                #args.class_num = 25
                #args.src_classes = [i for i in range(25)]
                #args.tar_classes = [i for i in range(65)]

        #test_target(args)
        
    args.out_file.close()