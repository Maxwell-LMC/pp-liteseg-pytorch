import torch
import torch.nn as nn
import torch.nn.functional as F
class OhemCrossEntropyLoss(nn.Module):
    """
    Implements the ohem cross entropy loss function.

    Args:
        thresh (float, optional): The threshold of ohem. Default: 0.7.
        min_kept (int, optional): The min number to keep in loss computation. Default: 10000.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, thresh=0.7, min_kept=10000, ignore_index=255):
        super(OhemCrossEntropyLoss, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.EPS = 1e-5

    def forward(self, logit, label):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        """
        if len(label.shape) != len(logit.shape):
            label = torch.unsqueeze(label, dim=1)

        # get the label after ohem
        n, c, h, w = logit.shape
        label = label.reshape((-1, )).long()
        valid_mask = (label != self.ignore_index).long()
        num_valid = valid_mask.sum()
        label = label * valid_mask

        prob = F.softmax(logit, dim=1)
        prob = prob.transpose(1,0).reshape((c, -1))

        if self.min_kept < num_valid and num_valid > 0:
            # let the value which ignored greater than 1
            prob = prob + (1 - valid_mask)

            # get the prob of relevant label
            label_onehot = F.one_hot(label, c)
            label_onehot = label_onehot.transpose(1,0)
            prob = prob * label_onehot
            prob = torch.sum(prob, dim=0)

            threshold = self.thresh
            if self.min_kept > 0:
                index = prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                threshold_index = int(threshold_index)
                if prob[threshold_index] > self.thresh:
                    threshold = prob[threshold_index]
                kept_mask = (prob < threshold).long()
                label = label * kept_mask
                valid_mask = valid_mask * kept_mask

        # make the invalid region as ignore
        label = label + (1 - valid_mask) * self.ignore_index

        label = label.reshape((n, h, w))
        valid_mask = valid_mask.reshape((n, h, w)).float()
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        loss = criterion(logit, label)
        loss = loss * valid_mask
        avg_loss = torch.mean(loss) / (torch.mean(valid_mask) + self.EPS)

        label.requires_grad = False
        valid_mask.requires_grad = False
        return avg_loss