import torch
import torch.nn as nn
import time
from torch.autograd import grad


class IFRU(nn.Module):
    def __init__(self, params, sys_params):
        super(IFRU, self).__init__()
        self.params = params
        self.sys_params = sys_params
        self.device = params['device']

    def forward_once_grad(self, model, user_indices, pos_item_indices, neg_item_indices, graph=None):
        if graph:
            cf_loss, reg_loss = model.bpr_loss_once(user_indices, pos_item_indices, neg_item_indices, graph)
        else:
            cf_loss, reg_loss = model.bpr_loss_once(user_indices, pos_item_indices, neg_item_indices)
        loss = cf_loss + 1e-4 * reg_loss
        model_params = [p for p in model.parameters() if p.requires_grad]
        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
        return loss, grad_all

    def if_approxi(self, model, grad_all, grad_unl):
        '''
        res_tuple == (grad_all, grad1)
        '''
        start_time = time.time()
        scale = self.sys_params.scale

        v = tuple(grad1 for grad1 in grad_unl)
        h_estimate = tuple(grad1 for grad1 in grad_unl)

        model_params = [p for p in model.parameters() if p.requires_grad]
        hv = self.hvps(grad_all, model_params, h_estimate)
        with torch.no_grad():
            h_estimate = [v1 + h_estimate1 - hv1 / scale
                          for v1, h_estimate1, hv1 in zip(v, h_estimate, hv)]

        params_change = [h_est / scale for h_est in h_estimate]
        params_esti = [p1 + p2 for p1, p2 in zip(params_change, model_params)]

        idx = 0
        for p in model.parameters():
            p.data = params_esti[idx]
            idx = idx + 1

        # return time.time() - start_time

    def hvps(self, grad_all, model_params, h_estimate):
        element_product = 0
        for grad_elem, v_elem in zip(grad_all, h_estimate):
            element_product += torch.sum(grad_elem * v_elem)

        return_grads = grad(element_product, model_params, create_graph=True)
        return return_grads
