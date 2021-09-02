"""
generate src data or feat and then train trg model
"""
import random
import time
import warnings
import sys
import argparse
import copy
import numpy as np
import os
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

sys.path.append('.')
from dalib.modules.classifier import ImageClassifier
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.dann import DomainAdversarialLoss
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss
from dalib.adaptation.sfda import DataSetPath, DataSetGen, mut_info_loss, save_src_imgs, MyDataParallel
import dalib.vision.datasets as datasets
import dalib.vision.models as models
from tools.utils import AverageMeter, ProgressMeter, accuracy, labels_to_one_hot
from tools.transforms import train_transform_aug0, train_transform_center_crop
from tools.transforms import val_transform
from tools.lr_scheduler import StepwiseLR


device = torch.device("cuda")
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


def main(args: argparse.Namespace):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    if args.center_crop:
        train_transform = train_transform_center_crop
    else:
        train_transform = train_transform_aug0

    # Data loading code    
    dataset = datasets.__dict__[args.data]
    # trainset
    trainset = dataset(root=args.root, task=args.target, download=True, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.workers, drop_last=True)
    # valset
    valset = dataset(root=args.root, task=args.target, download=True, transform=val_transform)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    # genset
    valset2 = dataset(root=args.root, task=args.target, download=True, transform=val_transform)
    valset2_path = DataSetPath(valset2)
    val_loader2 = DataLoader(valset2_path, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # create source model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=False)
    src_model = ImageClassifier(backbone, trainset.num_classes).to(device)

    # load pretrained source model
    print(f"=> loading source model: {args.data}_{args.source}_{args.arch}.pth")
    src_state_dict = torch.load(f'{args.model_dir}/{args.data}_{args.source}_{args.arch}.pth')
    src_model.load_state_dict(src_state_dict)

    # create target model
    trg_model = copy.deepcopy(src_model)

    # freeze source model
    for param in src_model.parameters():
        param.requires_grad = False
    src_model.eval()

    # freeze target classifer
    if args.fix_head:
        for param in trg_model.head.parameters():
            param.requires_grad = False

    # start training
    # ----------------- stage 1 --------------------- #
    print("=> start stage 1, source images synthesis")
    if args.synth_method == 'ce':
        data_list_file, gen_folder = src_img_synth_ce(val_loader2, src_model, args)
    elif args.synth_method == 'admm':
        data_list_file, gen_folder = src_img_synth_admm(val_loader2, src_model, args)
    
    genset = DataSetGen(data_list_file)
    gen_loader = DataLoader(genset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.workers, drop_last=True)
    
    # ----------------- stage 2 --------------------- #
    if args.uda_type == 'dann':
        domain_discri = DomainDiscriminator(in_feature=src_model.features_dim, hidden_size=1024).to(device)
        domain_adv = DomainAdversarialLoss(domain_discri).to(device)
    elif args.uda_type == 'cdan':
        domain_discri = DomainDiscriminator(src_model.features_dim * trainset.num_classes, hidden_size=1024).to(device)
        domain_adv = ConditionalDomainAdversarialLoss(domain_discri, entropy_conditioning=False, num_classes=trainset.num_classes, 
                        features_dim=src_model.features_dim, randomized=False).to(device)

    if torch.cuda.device_count() > 1:
        # src_model = MyDataParallel(src_model)
        trg_model = MyDataParallel(trg_model)
        # domain_discri = MyDataParallel(domain_discri)

    # define optimizer and lr scheduler
    optimizer = SGD(trg_model.get_parameters() + domain_discri.get_parameters(), 
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = StepwiseLR(optimizer, init_lr=args.lr, gamma=0.001, decay_rate=0.75)

    print("=> start stage 2, uda")
    eval_interval = args.epochs // args.num_eval
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train_uda(train_loader, gen_loader, trg_model, domain_adv, optimizer, lr_scheduler, epoch, args)

        # evaluate on validation set
        if epoch % eval_interval == 0:
            if args.data == "VisDA2017":
                acc1 = validate_per_class(val_loader, trg_model, args)
            else:
                acc1 = validate(val_loader, trg_model, args)

        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    shutil.rmtree(gen_folder)


def src_img_synth_ce(data_loader, src_model, args):

    gen_folder = 'gen_data_ce/'

    data_list_file = args.root
    data_list_file = data_list_file.replace('data/', gen_folder)
    data_list_file += '/image_list/'+args.source+'2'+args.target+'.txt'

    # return data_list_file
    
    dir = os.path.dirname(data_list_file)
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    for batch_idx, (images_t, _, path) in enumerate(data_loader):

        if batch_idx == 0 and os.path.exists(data_list_file):
            os.remove(data_list_file)

        images_t = images_t.to(device)
        # get pseudo labels
        y_t, _ = src_model(images_t)
        plabel_t = y_t.argmax(dim=1)

        # init src img
        images_s = images_t.clone()
        images_s.requires_grad_()
        optimizer_s = SGD([images_s], args.lr_img, momentum=args.momentum_img)

        for iter_i in range(args.iters_img):
            y_s, _ = src_model(images_s)
            loss = F.cross_entropy(y_s, plabel_t)
            
            optimizer_s.zero_grad()
            loss.backward()
            optimizer_s.step()

        # save src imgs
        save_src_imgs(images_s.detach_().cpu(), plabel_t, path, gen_folder, data_list_file, args)

    return data_list_file, gen_folder


def src_img_synth_admm(data_loader, src_model, args):

    gen_folder = 'gen_data_admm/'

    data_list_file = args.root
    data_list_file = data_list_file.replace('data/', gen_folder)
    data_list_file += '/image_list/'+args.source+'2'+args.target+'.txt'
    
    dir = os.path.dirname(data_list_file)
    if not os.path.exists(dir):
        os.makedirs(dir)

    # initialize
    for batch_idx, (images_t, _, path) in enumerate(data_loader):

        if batch_idx == 0 and os.path.exists(data_list_file):
            os.remove(data_list_file)

        images_t = images_t.to(device)
        # get pseudo labels
        y_t, _ = src_model(images_t)
        plabel_t = y_t.argmax(dim=1)

        save_src_imgs(images_t.cpu(), plabel_t, path, gen_folder, data_list_file, args)

    genset = DataSetGen(data_list_file)
    genset_path = DataSetPath(genset)
    gen_loader = DataLoader(genset_path, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    LAMB = torch.zeros_like(src_model.head.weight.data).to(device)

    for i in range(args.iters_admm):

        print(f'admm iter: {i}/{args.iters_admm}')

        # step1: update imgs
        for batch_idx, (images_s, labels_s, paths) in enumerate(gen_loader):

            images_s = images_s.to(device)
            labels_s = labels_s.to(device)

            # convert labels to one-hot
            plabel_onehot = labels_to_one_hot(labels_s, src_model.num_classes, device)

            # init src img
            images_s.requires_grad_()
            optimizer_s = SGD([images_s], args.lr_img, momentum=args.momentum_img)
            
            for iter_i in range(args.iters_img):
                y_s, f_s = src_model(images_s)
                loss = F.cross_entropy(y_s, labels_s)
                p_s = F.softmax(y_s, dim=1)
                grad_matrix = (p_s - plabel_onehot).t() @ f_s / p_s.size(0)
                new_matrix = grad_matrix + args.param_gamma * src_model.head.weight.data
                grad_loss = torch.norm(new_matrix, p='fro') ** 2
                loss += grad_loss * args.param_admm_rho / 2
                loss += torch.trace(LAMB.t() @ new_matrix)
                
                optimizer_s.zero_grad()
                loss.backward()
                optimizer_s.step()

            # update src imgs
            for img, path in zip(images_s.detach_().cpu(), paths):
                torch.save(img.clone(), path)

        # step2: update LAMB
        grad_matrix = torch.zeros_like(LAMB).to(device)
        for batch_idx, (images_s, labels_s, paths) in enumerate(gen_loader):
            images_s = images_s.to(device)
            labels_s = labels_s.to(device)

            # convert labels to one-hot
            plabel_onehot = labels_to_one_hot(labels_s, src_model.num_classes, device)

            y_s, f_s = src_model(images_s)
            p_s = F.softmax(y_s, dim=1)
            grad_matrix += (p_s - plabel_onehot).t() @ f_s

        new_matrix = grad_matrix / len(gen_loader.dataset) + args.param_gamma * src_model.head.weight.data
        LAMB += new_matrix * args.param_admm_rho

    return data_list_file, gen_folder


def train_uda(trg_loader: DataLoader, src_loader: DataLoader, trg_model: ImageClassifier, 
            domain_adv: DomainAdversarialLoss,
            optimizer: SGD, lr_scheduler: StepwiseLR, epoch: int, args: argparse.Namespace):
    
    batch_time = AverageMeter('Time', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    progress = ProgressMeter(
        len(trg_loader),
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    trg_model.train()
    src_iter = iter(src_loader)
    for i, (images_t, _) in enumerate(trg_loader):
        lr_scheduler.step()

        try:
            images_s, labels_s = next(src_iter)
        except StopIteration:
            src_iter = iter(src_loader)
            images_s, labels_s = next(src_iter)

        images_s = images_s.to(device)
        labels_s = labels_s.to(device)
        images_t = images_t.to(device)

        y_s, f_s = trg_model(images_s)
        y_t, f_t = trg_model(images_t)

        loss = 0.
        if args.param_cls_s > 0:
            cls_loss_s = F.cross_entropy(y_s, labels_s)
            loss += cls_loss_s * args.param_cls_s

        if args.uda_type == 'dann' and  args.param_dann > 0:
            dann_loss = domain_adv(f_s, f_t)
            loss += dann_loss * args.param_dann

        if args.uda_type == 'cdan' and  args.param_cdan > 0:
            cdan_loss = domain_adv(y_s, f_s, y_t, f_t)
            loss += cdan_loss * args.param_cdan

        if args.param_mi > 0:
            p_t = F.softmax(y_t, dim=1)
            mi_loss = mut_info_loss(p_t)
            loss += mi_loss * args.param_mi
        
        losses.update(loss.item(), images_t.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images_t, target) in enumerate(val_loader):
            images_t = images_t.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images_t)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images_t.size(0))
            top1.update(acc1.item(), images_t.size(0))
            top5.update(acc5.item(), images_t.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def validate_per_class(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time,],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    predicts = []
    targets = []
    with torch.no_grad():
        end = time.time()
        for i, (images_t, target) in enumerate(val_loader):
            images_t = images_t.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images_t)
            predict = output.argmax(dim=1)
            
            predicts.append(predict)
            targets.append(target)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        predicts = torch.cat(predicts)
        targets = torch.cat(targets)
        
        matrix = confusion_matrix(targets.cpu().float(), predicts.cpu().float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)

        print(f' * Acc@1: {aacc}, Accs: {acc}')

    return aacc


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=250, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay',default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--model_dir', default='source_models')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--num_eval', default=25, type=int)
    parser.add_argument('--center_crop', default=False, action='store_true')
    parser.add_argument('--fix_head', default=False, action='store_true')
    # args for data synthesis
    parser.add_argument('--lr_img', default=10., type=float)
    parser.add_argument('--momentum_img', default=0.9, type=float, metavar='M',
                        help='momentum of img optimizer')
    parser.add_argument('--iters_img', default=10, type=int, metavar='N',
                        help='number of total inner epochs to run')
    parser.add_argument('--param_gamma', default=0.01, type=float)
    parser.add_argument('--param_admm_rho', default=0.01, type=float)
    parser.add_argument('--iters_admm', default=3, type=int)
    parser.add_argument('--synth_method', default='admm')
    # param for semi-supervised learning
    parser.add_argument('--uda_type', default='cdan')
    parser.add_argument('--param_cdan', default=0., type=float)
    parser.add_argument('--param_dann', default=0., type=float)
    parser.add_argument('--param_cls_s', default=0., type=float)
    parser.add_argument('--param_mi', default=1., type=float)

    args = parser.parse_args()
    print(args)
    main(args)
