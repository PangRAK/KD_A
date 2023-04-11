from pytorch_metric_learning.regularizers import SparseCentersRegularizer
from pytorch_metric_learning.losses import ContrastiveLoss, NTXentLoss
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from my_dataloader import get_dataloader
from models import model_dict
import os
from utils import AverageMeter, accuracy
import numpy as np
from datetime import datetime
import random
from tqdm import tqdm
import loss 
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

from pytorch_metric_learning import losses, miners


from pytorch_metric_learning.distances import LpDistance

from loss.crossbatchmemory import CrossBatchMemory



parser = argparse.ArgumentParser()
parser.add_argument("--model_names", type=str, default="vgg16")


parser.add_argument("--root", type=str, default="../data/")
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--classes_num", type=int, default=100)
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar100",
    choices=["cifar100", "cifar10", "CUB", "tinyimagenet"],
    help="dataset",
)
parser.add_argument("--epsilon", default=0.03, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")


# parser.add_argument("--classes_num", type=int, default=100)

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epoch", type=int, default=240)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight-decay", type=float, default=5e-4)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--milestones", type=int,
                    nargs="+", default=[150, 180, 210])


parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--gpu-id", type=int, default=0)
parser.add_argument("--print_freq", type=int, default=5)
parser.add_argument("--aug_nums", type=int, default=2)
parser.add_argument("--exp_postfix", type=str, default="base")


parser.add_argument("--T", type=float)
parser.add_argument("--alpha", type=float)

# parser.add_argument('data', help='path to dataset')
parser.add_argument('-j', '--workers', default=2, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number')
# parser.add_argument('-b', '--batch-size', default=32, type=int,
#                     help='mini-batch size')
# parser.add_argument('--lr', default=0.05, type=float,
#                     help='initial model learning rate')
parser.add_argument('--centerlr', default=0.01, type=float,
                    help='initial center learning rate')
# parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
#                     help='weight decay', dest='weight_decay')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--eps', default=0.01, type=float,
                    help='epsilon for Adam')
parser.add_argument('--rate', default=0.1, type=float,
                    help='decay rate')
parser.add_argument('--dim', default=64, type=int,
                    help='dimensionality of embeddings')
parser.add_argument('--freeze_BN', action='store_true',
                    help='freeze bn')
parser.add_argument('--la', default=20, type=float,
                    help='lambda')
# parser.add_argument('--gamma', default=0.1, type=float,
#                     help='gamma')
parser.add_argument('--tau', default=0.2, type=float,
                    help='tau')
parser.add_argument('--margin', default=0.01, type=float,
                    help='margin')
parser.add_argument('-C', default=98, type=int,
                    help='C')
parser.add_argument('-K', default=10, type=int,
                    help='K')


args = parser.parse_args()
args.num_branch = len(args.model_names)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

exp_name = "_".join(args.model_names) + args.exp_postfix
exp_path = "./dlb/{}/{}".format(args.dataset, exp_name)
os.makedirs(exp_path, exist_ok=True)
print(exp_path)

######################
### from MoCo repo ###
######################
def copy_params(encQ, encK, m=None):
    if m is None:
        for param_q, param_k in zip(encQ.parameters(), encK.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    else:
        for param_q, param_k in zip(encQ.parameters(), encK.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)



######################
### from MoCo repo ###
######################
def batch_shuffle_single_gpu(x):
    """
    Batch shuffle, for making use of BatchNorm.
    """
    # random shuffle index
    idx_shuffle = torch.randperm(x.shape[0]).cuda()

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    return x[idx_shuffle], idx_unshuffle

######################
### from MoCo repo ###
######################
def batch_unshuffle_single_gpu(x, idx_unshuffle):
    """
    Undo batch shuffle.
    """
    return x[idx_unshuffle]



def create_labels(labels ,num_pos_pairs, previous_max_label):
    # create labels that indicate what the positive pairs are
    labels = torch.cat((labels, labels)).cuda()
    labels += previous_max_label + 1  ### queue 1024 사이즈 tracking
    enqueue_idx = torch.arange(num_pos_pairs, num_pos_pairs * 2)

    return labels, enqueue_idx


def tsne_plot(targets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(os.path.join('tsne.png'), bbox_inches='tight')
    print('done!')


def train_one_epoch(model, modelK, pair_loss, lm, paramK_momentum, optimizer,  train_loader, alpha, pre_data, pre_out, ep):
    model.train()
    acc_recorder = AverageMeter()
    loss_recorder = AverageMeter()
    

    tsne = TSNE()

    
    for i, data in enumerate(train_loader):

        iteration = ep * len(train_loader) + i
        


        imgs, label = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            label = label.cuda()
        _, out = model.forward(imgs[:, 0, ...])
    

        # 제외시킴
        if lm is None:
            # prev_max_label = pair_loss.label_memory
            previous_max_label = 0
            # previous_max_label = torch.max(pair_loss.label_memory)
        else:
            previous_max_label = torch.max(lm) 
        

        with torch.no_grad():
            copy_params(model, modelK, m=paramK_momentum)
            # shuffle index 
            imgK, idx_unshuffle = batch_shuffle_single_gpu(imgs[:, 1, ...])
            # imgK, idx_unshuffle = batch_shuffle_single_gpu(imgK)     
            # _, encK_out_shuffle = modelK.forward(imgK)
            _, encK_out = model.forward(imgK)
            # encK_proj, enck_out = modelK.forward(imgs[:, 1, ...])
            enck_out_unshuffle = batch_unshuffle_single_gpu(encK_out, idx_unshuffle)
            # encK_proj = batch_unshuffle_single_gpu(encK_proj, idx_unshuffle)




        # e_mem, l_mem, triplet_loss, filled = pair_loss(out, label)        
        all_enc = torch.cat((out, encK_out),dim=0)  
        loss_xbm, lm = pair_loss(
            # all_enc, labels, enqueue_idx=enqueue_idx)
            all_enc, torch.cat((label, label), dim=0))



        
        if pre_data != None:
            
            
                    
            pre_images, pre_label = pre_data
            if torch.cuda.is_available():
                pre_images = pre_images.cuda()
                pre_label = pre_label.cuda()
            _, out_pre = model.forward(pre_images[:, 1, ...])
            # _, out_pre_1 = modelK.forward(pre_images[:,0, ...])



            ce_loss = F.cross_entropy(
                torch.cat((out_pre, out), dim=0), 
                torch.cat((pre_label, label), dim=0)
            ) 
            


            dml_loss = (
                F.kl_div(
                    # [aug 1 , aug 0 ]
                    F.log_softmax(out_pre / args.T, dim=1),
                    F.softmax(pre_out.detach() / # [aug 0, aug 1]
                              args.T, dim=1),  # detach
                    reduction="batchmean",
                )
                * args.T
                * args.T
            )
        

            # pair_loss = nn.TripletMarginLoss(margin=0.2)(
            #     # pre_out.detach(), sibal, out_pre)
            #     out, out_2, out_pre) 

            
            # loss = ce_loss
            # loss = ce_loss + 0.1 * loss_xbm + dml_loss
            loss = ce_loss + 2 * loss_xbm + dml_loss


        else:
            ce_loss = F.cross_entropy(out, label)
            loss = ce_loss + 2 * loss_xbm
            

        loss_recorder.update(loss.item(), n=imgs.size(0))
        acc = accuracy(out, label)[0]
        acc_recorder.update(acc.item(), n=imgs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pre_data = data
        pre_out = out
        


    losses = loss_recorder.avg
    acces = acc_recorder.avg

    return losses, acces, pre_data, pre_out, lm


    
def evaluation(model, val_loader, ep):
    model.eval()
    acc_recorder = AverageMeter()
    loss_recorder = AverageMeter()




    with torch.no_grad():
        for img, label in val_loader:
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            targets_np = label.data.cpu().numpy()
            feat, out = model(img)
            outputs_np = out.data.cpu().numpy()
            
            acc = accuracy(out, label)[0]
            loss = F.cross_entropy(out, label)
            acc_recorder.update(acc.item(), img.size(0))
            loss_recorder.update(loss.item(), img.size(0))
            
    # targets = np.concatenate(targets_np, axis=0)
    # outputs = np.concatenate(outputs_np, axis=0).astype(np.float64)
    # if ep == 239:
    #     tsne_plot(targets_np, outputs_np.astype(np.float64))
    
    
    losses = loss_recorder.avg
    acces = acc_recorder.avg
    return losses, acces


def train(model, modelK, loss_fn, paramK_momentum, optimizer, train_loader, scheduler):

    best_acc = -1

    f = open(os.path.join(exp_path, "log_test.txt"), "w")
    pre_data, pre_out, lm = None, None, None
    
    for epoch in tqdm(range(args.epoch)):
        alpha = args.alpha

        # adjust_learning_rate(optimizer, epoch, args)   #decay 역할
        train_losses, train_acces, pre_data, pre_out, lm = train_one_epoch(
            model, modelK, loss_fn, lm,  paramK_momentum, optimizer, train_loader, alpha, pre_data, pre_out, epoch)

        val_losses, val_acces = evaluation(model, val_loader, epoch)

        if val_acces > best_acc:
            best_acc = val_acces
            state_dict = dict(
                epoch=epoch + 1, model=model.state_dict(), acc=val_acces)
            name = os.path.join(exp_path, args.model_names, "ckpt", "best.pth")
            os.makedirs(os.path.dirname(name), exist_ok=True)
            torch.save(state_dict, name)
            
            # if (epoch + 1) % args.print_freq == 0:
            msg = "epoch:{} model:{} train loss:{:.2f} acc:{:.2f}  val loss{:.2f} acc:{:.2f}\n".format(
            # msg = "epoch:{} model:{} train loss:{:.2f} acc:{:.2f} acc:{:.2f}\n".format(
                epoch,
                args.model_names,
                train_losses,
                train_acces,
                val_losses,
                val_acces,
            )
            print(msg)
            f.write(msg)
            f.flush()

        scheduler.step()



    msg_best = "model:{} best acc:{:.2f}".format(args.model_names, best_acc)
    print(msg_best)
    f.write(msg_best)
    f.close()


if __name__ == "__main__":
    train_set, val_set, train_loader, val_loader = get_dataloader(args)
    lr = args.lr
    model = model_dict[args.model_names](num_classes=args.classes_num)
    modelK = model_dict[args.model_names](num_classes=args.classes_num)

    copy_params(model, modelK)
    paramK_momentum = 0.99

    if torch.cuda.is_available():
        model = model.cuda()
        modelK = modelK.cuda()
        
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay,
    )

    mining_func = miners.TripletMarginMiner(
        margin=0.2, distance=LpDistance(power=2), type_of_triplets="semi-hard")
        # margin=0, distance=LpDistance(power=2), type_of_triplets="hard")

    # mining_func = miners.PairMarginMiner(pos_margin=0, neg_margin=1)


    loss_fn = CrossBatchMemory(
        # loss=NTXentLoss(temperature=0.1), embedding_size=512, memory_size=1024, miner=mining_func)
        # loss=nn.TripletMarginLoss(margin=0.2), embedding_size=100, memory_size=1024, miner=mining_func)
        # loss=losses.TripletMarginLoss(margin=0.2), embedding_size=100, memory_size=1024, miner=mining_func)
        # loss=losses.TripletMarginLoss(margin=0.2), embedding_size=100, memory_size=1024)
        loss=losses.TripletMarginLoss(margin=0.2), embedding_size=100, memory_size=256)



    scheduler = MultiStepLR(optimizer, args.milestones, args.gamma)

    train(model, modelK, loss_fn, paramK_momentum, optimizer, train_loader, scheduler)
