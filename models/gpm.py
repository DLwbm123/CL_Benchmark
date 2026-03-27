import numpy as np
import torch
import logging
from models.utils.continual_model import ContinualModel
import time
from utils.args import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Gradient Episodic Memory.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--inputsize', type=int, default=256,
                        help='InputSize for GPM.')
    parser.add_argument('--example_bsz', type=int, default=16,
                        help='number of examples used to get the representation matrix.')
    parser.add_argument('--threshold', type=float, default=0.97,
                        help='threshold for PCA.')
    return parser


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=1, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))




def get_representation_matrix_pro(net, device, x, bszs):
    
    logger = logging.getLogger('base')
    example_data = x.to(device)
    logger.info("example here!")
    example_out = net.features(example_data)
    #batch_list = [8] + [bszs//4] * 4 + [bszs//2] *6 + [bszs] * 12
    batch_list = [bszs] + [bszs] * 4 + [bszs] *6 + [bszs] * 12
    logger.info("batch list: {}".format(batch_list))
    #batch_list = [2 * 12, 20, 20, 20, 20, 20,20, 20, 20, 20]
    mat_list = []
    act_key = list(net.act.keys())
    logger.info("netmap: {}".format(net.map))
    for i in range(len(net.map)):
        bsz = batch_list[i]
        k = 0
        ksz = net.ksize[i]
        #print(i,"netmap", net.map[i], "kernelsize", ksz, "actsize", net.act[act_key[i]].size())
        s = compute_conv_output_size(net.map[i], net.ksize[i])
        mat = np.zeros((net.ksize[i] * net.ksize[i] * net.in_channel[i], s * s * bsz))
        #print("mat size", s, mat.shape, bsz)
        act = net.act[act_key[i]].detach().cpu().numpy()
        #B,C,H,W = act.shape
        #shift_H = (H-(s-ksz+1))//2
        #shift_W = (W-(s-ksz+1))//2
        #print("activation size", act.shape)
        for kk in range(bsz):
            for ii in range(s-ksz+1):
                for jj in range(s-ksz+1):
                    #print("left", mat[:, k].shape, "right", act[kk, :, ii:ksz + ii, jj:ksz + jj].shape, jj, ksz + jj)
                    #shift_H = (H-(s-ksz+1))//2
                    mat[:, k] = act[kk, :, ii:ksz + ii, jj:ksz + jj].reshape(-1)
                    k += 1
        mat_list.append(mat)
        '''else:
            act = net.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)'''

    logger.info('-' * 30)
    logger.info('Representation Matrix')
    logger.info('-' * 30)
    for i in range(len(mat_list)):
        logger.info('Layer {} : {}'.format(i + 1, mat_list[i].shape))
    logger.info('-' * 30)
    return mat_list


def update_GPM(mat_list, threshold, feature_list=[], ):
    logger = logging.getLogger('base')
    if not feature_list:
        logger.info("may be for task 1 here")
        # After First Task
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1
            feature_list.append(U[:, 0:r])
    else:
        logger.info("may be for other here")
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1 ** 2).sum()
            # Projected Representation (Eq-8)
            act_hat = activation - np.dot(np.dot(feature_list[i], feature_list[i].transpose()), activation)
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            accumulated_sval = (sval_total - sval_hat) / sval_total

            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                logger.info('Skip Updating GPM for layer: {}'.format(i + 1))
                continue
            # update GPM
            Ui = np.hstack((feature_list[i], U[:, 0:r]))
            if Ui.shape[1] > Ui.shape[0]:
                feature_list[i] = Ui[:, 0:Ui.shape[0]]
            else:
                feature_list[i] = Ui

    logger.info('-' * 40)
    logger.info('Gradient Constraints Summary')
    logger.info('-' * 40)
    for i in range(len(feature_list)):
        logger.info('Layer {} : {}/{}'.format(i + 1, feature_list[i].shape[1], feature_list[i].shape[0]))
    logger.info('-' * 40)
    return feature_list

def get_basis(data):
    a = np.cov(data) * (1 / (data.shape[1] - 1))
    sigma, eig = np.linalg.eig(a)
    return eig, sigma

def update_GPM_PCA(mat_list, threshold, feature_list=[]):
    logger = logging.getLogger('base')
    logger.info('Threshold: {}'.format(threshold))
    if not feature_list:
        t0 = time.time()
        logger.info("may be for task 1 here")
        # After First Task
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U, S = get_basis(activation)
            # criteria (Eq-5)
            sval_total = (S).sum()
            sval_ratio = (S) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1
            feature_list.append(U[:, 0:r])
        t1 = time.time()
        logger.info("Time used: {}".format(t1-t0))
    else:
        logger.info("may be for other here")
        t0 = time.time()
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1, S1= get_basis(activation)
            sval_total = (S1).sum()
            act_hat = activation - np.dot(np.dot(feature_list[i], feature_list[i].transpose()), activation)
            U, S= get_basis(act_hat)
            sval_hat = (S).sum()
            sval_ratio = (S) / sval_total
            accumulated_sval = (sval_total - sval_hat) / sval_total

            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                logger.info('Skip Updating GPM for layer: {}'.format(i + 1))
                continue
            # update GPM
            Ui = np.hstack((feature_list[i], U[:, 0:r]))
            if Ui.shape[1] > Ui.shape[0]:
                feature_list[i] = Ui[:, 0:Ui.shape[0]]
            else:
                feature_list[i] = Ui
        t1 = time.time()
        logger.info("Time used: {}".format(t1-t0))

    logger.info('-' * 40)
    logger.info('Gradient Constraints Summary')
    logger.info('-' * 40)
    for i in range(len(feature_list)):
        logger.info('Layer {} : {}/{}'.format(i + 1, feature_list[i].shape[1], feature_list[i].shape[0]))
    logger.info('-' * 40)
    return feature_list


class Gpm(ContinualModel):
    NAME = 'gpm'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Gpm, self).__init__(backbone, loss, args, transform)
        #self.current_task = 0
        self.feature_list = []
        
    def begin_task(self, dataset):
        logger = logging.getLogger('base')
        if self.current_task == 0:
            pass
        else:
            logger.info('Generate projection matrix for each layer')
            self.feature_mat = []
            for k in range(len(self.net.act)):
                Uf = torch.Tensor(np.dot(self.feature_list[k], self.feature_list[k].transpose())).to(self.device)
                logger.info('Layer {} - Projection Matrix shape: {}'.format(k + 1, Uf.shape))
                self.feature_mat.append(Uf)
            logger.info('-' * 40)

    def end_task(self, dataset):
        #self.current_task += 1
        loader = dataset.not_aug_dataloader(self.args.example_bsz)
        xtrain, _ = next(iter(loader))
        mat_list = get_representation_matrix_pro(self.net, self.device, xtrain, self.args.example_bsz)
        threshold = np.array([self.args.threshold] * 19) + (self.current_task) * np.array([0.001] * 19)
        self.feature_list = update_GPM_PCA(mat_list, threshold, self.feature_list)
        
        
    def observe(self, inputs, labels, not_aug_inputs):


        # now compute the grad on the current data
        self.opt.zero_grad()
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        # check if gradient violates buffer constraints
        if self.current_task == 0:
            pass
        else:
            kk = 0
            '''for k, (m, params) in enumerate(self.net.named_parameters()):
                if "last" not in m and len(params.size()) == 4:
                    #print("check", k, m, params.shape, self.feature_mat[kk].size())
                    sz = params.grad.data.size(0)
                    random_direct = torch.rand(params.size()).to(self.device)
                    random_norm = random_direct.norm(p=2)
                    params.grad.data = 1.0/(random_norm + 1e-12)*random_direct
                    if torch.any(torch.isnan(params.grad.data)):
                        print('NAN!')
                        print(grad_norm)
                        print(random_norm)
                        sys.exit(0)
                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1), \
                                                                   self.feature_mat[kk]).view(params.size())
                    kk += 1'''
            for k, (m, params) in enumerate(self.net.named_parameters()):
                if "last" not in m and len(params.size()) == 4:
                    #print("check", k, m, params.shape, self.feature_mat[kk].size())
                    sz = params.grad.data.size(0)
                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1), \
                                                                   self.feature_mat[kk]).view(params.size())
                    kk += 1

        self.opt.step()

        return loss.item()
    
    def _grad_norm(self):
        shared_device = self.opt.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.opt.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
