import torch, json, os, csv
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import math

def normalize_tensor(tensor, rescale=False):
    tmin = torch.min(tensor)
    if rescale or tmin < 0:
        tensor -= tmin
    tsum = tensor.sum()
    if tsum > 0:
        return tensor / tsum
    print("Zero tensor")
    tensor.fill_(1. / tensor.numel())
    return tensor

def read_csv(csv_file):
    if os.path.exists(csv_file):
        print('load: ', csv_file)
    else:
        print(csv_file, 'does not exist')
    data = []
    with open(csv_file,  encoding='UTF-8') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data

def write_json(data_input, file_name):
    json_data = json.dumps(data_input, indent=4, ensure_ascii=False)
    json_file = file_name
    with open(json_file, 'w', encoding='UTF-8') as fp:
        fp.write(json_data)
    print('write the:', json_file)

def read_json(json_file):
    with open (json_file, 'r') as f:
        res_data = json.load(f)
    return res_data


def corr_coeff(pred, target):
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    target = target.reshape(new_size)

    cc = []
    for x, y in zip(torch.unbind(pred, 0), torch.unbind(target, 0)):
        xm, ym = x - x.mean(), y - y.mean()
        ### for ym.sum ==0， r=0，can't cal loss and BP 
        if ym.sum() == 0:
            r = ym.sum()
        else:
            r_num = torch.mean(xm * ym)
            r_den = torch.sqrt(
                torch.mean(torch.pow(xm, 2)) * torch.mean(torch.pow(ym, 2)))
            r = r_num / r_den
        cc.append(r)

    cc = torch.stack(cc)
    cc = cc.reshape(size[:2])
    return cc  # 1 - torch.square(r)

def new_cc(pred, target):
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    target = target.reshape(new_size)

    cc = []
    for x, y in zip(torch.unbind(pred, 0), torch.unbind(target, 0)):
        x = (x/x.max())*255
        y = (y/y.max())*255
        xm, ym = x - x.mean(), y - y.mean()
        ### for ym.sum ==0， r=0，can't cal loss and BP 
        if ym.sum() == 0:
            r = ym.sum()
        else:
            r_num = torch.mean(xm * ym)
            r_den = torch.sqrt(
                torch.mean(torch.pow(xm, 2)) * torch.mean(torch.pow(ym, 2)))
            r = r_num / r_den
        cc.append(r)

    cc = torch.stack(cc)
    cc = cc.reshape(size[:2])
    return cc  # 1 - torch.square(r)

def new_kld(pred, target):
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    target = target.reshape(new_size)
    kld = []
    for x, y in zip(torch.unbind(pred, 0), torch.unbind(target, 0)):
        x = (x/x.max())*255
        y = (y/y.max())*255

        eps = torch.finfo(torch.float32).eps

        P = x / (eps + x.sum())
        Q = y / (eps + y.sum())
        k = (Q * torch.log(eps+ Q/(eps+P))).sum()

        kld.append(k)

    kld = torch.stack(kld)
    # kld = kld.reshape(size[:2])
    return kld  

def new_sim(pred, target):
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    target = target.reshape(new_size)
    sims = []
    for x, y in zip(torch.unbind(pred, 0), torch.unbind(target, 0)):
        # eps = torch.finfo(torch.float32).eps
        # x = (x-x.min()) / (x.max() - x.min()+eps)
        # y = (y-y.min()) / (x.max() - x.min()+eps)
        # sim loss under the condiction of x.sum()=1 and y.sum()=1
        x = x / x.sum()
        y = y / y.sum()
        sim = torch.min(x, y).sum()
        sims.append(sim)

    sims = torch.stack(sims)
    # sims = sims.reshape(size[:2])
    return sims

def kld_loss(pred, target):
    loss = F.kl_div(pred, target, reduction='none')
    loss = loss.sum(-1).sum(-1).sum(-1)
    return loss



def nss(pred, fixations):
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    fixations = fixations.reshape(new_size)

    pred_normed = (pred - pred.mean(-1, True)) / pred.std(-1, keepdim=True)
    results = []
    for this_pred_normed, mask in zip(torch.unbind(pred_normed, 0),
                                      torch.unbind(fixations, 0)):
        if mask.sum() == 0:
            print("No fixations.")
            results.append(torch.ones([]).float().to(fixations.device))
            continue
        nss_ = torch.masked_select(this_pred_normed, mask)
        nss_ = nss_.mean(-1)
        results.append(nss_)
    results = torch.stack(results)
    # results = results.reshape(size[:2])
    return results

def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    norm_s_map = (s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map)))
    return norm_s_map


def auc_judd(s_map, gt):
    # ground truth is discrete, s_map is continous and normalized
    s_map = normalize_map(s_map)
    assert np.max(gt) == 1.0,\
        'Ground truth not discretized properly max value > 1.0'
    assert np.max(s_map) == 1.0,\
        'Salience map not normalized properly max value > 1.0'

    # thresholds are calculated from the salience map,
    # only at places where fixations are present
    thresholds = s_map[gt > 0].tolist()

    num_fixations = len(thresholds)
    # num fixations is no. of salience map values at gt >0

    thresholds = sorted(set(thresholds))

    area = []
    area.append((0.0, 0.0))
    for thresh in thresholds:
        # in the salience map,
        # keep only those pixels with values above threshold
        temp = s_map >= thresh
        num_overlap = np.sum(np.logical_and(temp, gt))
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap
        # with gt / total number of non fixated pixels
        # this becomes nan when gt is full of fixations..this won't happen
        fp = (np.sum(temp) - num_overlap) / (np.prod(gt.shape[:2]) - num_fixations)

        area.append((round(tp, 4) ,round(fp, 4)))

    area.append((1.0, 1.0))
    area.sort(key=lambda x: x[0])
    tp_list, fp_list = list(zip(*area))
    return np.trapz(np.array(tp_list), np.array(fp_list))

def information_gain(y_true, y_pred, y_base, eps=np.spacing(1)):
    """
    Information gain. Assumes shape (b, h, w) for all tensors.

    :param y_true: groundtruth.
    :param y_pred: prediction.
    :param y_base: baseline.
    :param eps: regularization epsilon.
    :return: loss value (one symbolic value per batch element).
    """
    P = y_pred
    P = P / (eps + np.max(P, axis=(1, 2), keepdims=True))
    Q = y_true
    B = y_base

    Qb = np.round(Q)  # discretize at 0.5
    N = np.sum(Qb, axis=(1, 2), keepdims=True)
    IG = (np.log(eps + P) / np.log(2) - np.log(eps + B) / np.log(2))
    Temp = Qb *IG
    ig = np.sum(Qb *IG, axis=(1, 2), keepdims=True) / (eps + N)

    return ig

def auc_shuff_acl(s_map, gt, other_map, n_splits=100, stepsize=0.1):

    # If there are no fixations to predict, return NaN
    if np.sum(gt) == 0:
        print('no gt')
        return None

    # normalize saliency map
    s_map = normalize_map(s_map)

    S = s_map.flatten()
    F = gt.flatten()
    Oth = other_map.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)

    # for each fixation, sample Nsplits values from the sal map at locations
    # specified by other_map

    ind = np.where(Oth > 0)[0]  # find fixation locations on other images

    Nfixations_oth = min(Nfixations, len(ind))
    randfix = np.full((Nfixations_oth, n_splits), np.nan)

    for i in range(n_splits):
        # randomize choice of fixation locations
        randind = np.random.permutation(ind.copy())
        # sal map values at random fixation locations of other random images
        randfix[:, i] = S[randind[:Nfixations_oth]]

    # calculate AUC per random split (set of random locations)
    auc = np.full(n_splits, np.nan)
    for s in range(n_splits):

        curfix = randfix[:, s]

        allthreshes = np.flip(np.arange(0, max(np.max(Sth), np.max(curfix)), stepsize))
        tp = np.zeros(len(allthreshes) + 2)
        fp = np.zeros(len(allthreshes) + 2)
        tp[-1] = 1
        fp[-1] = 1

        for i in range(len(allthreshes)):
            thresh = allthreshes[i]
            tp[i + 1] = np.sum(Sth >= thresh) / Nfixations
            fp[i + 1] = np.sum(curfix >= thresh) / Nfixations_oth

        auc[s] = np.trapz(np.array(tp), np.array(fp))

    return np.mean(auc)


def similarity(s_map, gt):
    return np.sum(np.minimum(s_map, gt))


def getLabel(vid_index, frame_index):
    fixdatafile = ('./fixdata/fixdata' + str(vid_index) + '.mat')
    data = sio.loadmat(fixdatafile)

    fix_x = data['fixdata'][frame_index - 1][0][:, 3]
    fix_y = data['fixdata'][frame_index - 1][0][:, 2]
    mask = np.zeros((720, 1280), dtype='float32')

    for i in range(len(fix_x)):
        mask[fix_x[i], fix_y[i]] = 1
