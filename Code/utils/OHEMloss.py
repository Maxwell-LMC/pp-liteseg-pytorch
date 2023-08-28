import torch
import torch.nn as nn

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')
 
    def forward(self, logits, labels):
        N, C, H, W = logits.size()
 
        # OHEM here
        loss = self.criteria(logits, labels).view(-1)  # to 1-D
        loss, _ = torch.sort(loss, descending=True)  # sort
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
 
        return torch.mean(loss)