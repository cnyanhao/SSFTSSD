import sys
import torch
import os
import shutil
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
sys.path.append('.')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def create_exp_dir(path, scripts_to_save=None):
    os.makedirs(path, exist_ok=True)

    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        script_path = os.path.join(path, 'scripts')
        if os.path.exists(script_path):
            shutil.rmtree(script_path)
        os.mkdir(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            print(dst_file)
            shutil.copytree(script, dst_file)


class ForeverDataIterator:
    """A data iterator that will never stop producing data"""
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


def logits_to_probs(logits, labels):
    prob_vecs = F.softmax(logits, dim=1)
    probs = torch.gather(prob_vecs, 1, labels.view(-1,1))
    return probs


def logits_to_probs_unlabel(logits):
    prob_vecs = F.softmax(logits, dim=1)
    probs, _ = torch.max(prob_vecs, dim=1, keepdim=True)
    return probs


def labels_to_one_hot(labels, num_class, device):
    # convert labels to one-hot
    labels_one_hot = torch.FloatTensor(labels.shape[0], num_class).to(device)
    labels_one_hot.zero_()
    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return labels_one_hot


def subset_mut_info(logits, num_subsets, temp=1):
    eps = 1e-5
    num_per_subset = logits.shape[0] // num_subsets
    probs = F.softmax(logits, dim=1)
    mut_infos = []
    for i in range(num_subsets):
        probs_subset = probs[i*num_per_subset: (i+1)*num_per_subset]
        ent_y = torch.log(torch.tensor(logits.shape[1]))
        cond_ent = - torch.sum(probs_subset * torch.log(probs_subset+eps)) / num_per_subset
        mut_info = ent_y - cond_ent
        mut_infos.append(mut_info)
    mut_infos = torch.stack(mut_infos)
    mut_infos = torch.exp(mut_infos*temp) / (torch.exp(mut_infos*temp)).sum()
    mut_infos = mut_infos.repeat_interleave(num_per_subset)
    mut_infos.unsqueeze_(1)
    return mut_infos
