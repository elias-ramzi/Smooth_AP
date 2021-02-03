# repo originally forked from https://github.com/Confusezius/Deep-Metric-Learning-Baselines

###################### LIBRARIES #################################################
import warnings
warnings.filterwarnings("ignore")

import torch, faiss


"""================================================================================================="""
############ LOSS SELECTION FUNCTION #####################
def loss_select(loss, opt, to_optim):
    """
    Selection function which returns the respective criterion while appending to list of trainable parameters if required.

    Args:
        loss:     str, name of loss function to return.
        opt:      argparse.Namespace, contains all training-specific parameters.
        to_optim: list of trainable parameters. Is extend if loss function contains those as well.
    Returns:
        criterion (torch.nn.Module inherited), to_optim (optionally appended)
    """
    if loss == 'smoothap':
        loss_params  = {'anneal':opt.sigmoid_temperature}
        criterion    = SmoothAP(**loss_params)
    else:
        raise Exception('Loss {} not available!'.format(loss))

    return criterion, to_optim


"""==============================================Smooth-AP========================================"""


def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


def compute_aff(x):
    """computes the affinity matrix between an input vector and itself"""
    return torch.mm(x, x.t())


class SmoothAP(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss.
    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.
    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:
        labels = ( A, A, A, B, B, B, C, C, C)
    (the order of the classes however does not matter)
    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.
    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings
    Shape:
        - Input (preds): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar
    Examples::
        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """

    def __init__(self, anneal):
        """
        Parameters
        ----------
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function
        """
        super(SmoothAP, self).__init__()
        self.anneal = anneal

    def forward(self, input, target):
        """
        Parameters
        ----------
        input: N x d
            tensor of descriptors
        target: NxN
            affinity matrix, target[i,j] indicates if samples i and j are similar
        """
        batch_size = target.size(0)
        device = input.device

        mask = 1.0 - torch.eye(batch_size)
        mask = mask.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        sim_all = compute_aff(input)
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, batch_size, 1)
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        # pass through the sigmoid
        sim_sg = sigmoid(sim_diff, temp=self.temperature) * mask.to(device)
        # compute the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1

        # ------ differentiable ranking of only positive set in retrieval set ------
        # compute the mask which only gives non-zero weights to the positive set
        pos_mask = (target - torch.eye(batch_size).to(device))
        # pos_mask = target.unsqueeze(1).repeat(1, batch_size, 1)
        sim_pos_repeat = sim_all.unsqueeze(dim=1).repeat(1, batch_size, 1)
        sim_pos_diff = sim_pos_repeat - (sim_pos_repeat * pos_mask).permute(0, 2, 1)
        # pass through the sigmoid
        sim_pos_sg = sigmoid(sim_pos_diff, temp=self.temperature) * pos_mask
        # compute the rankings of the positive set
        sim_pos_rk = (torch.sum(sim_pos_sg, dim=-1) + target) * target

        ap = ((sim_pos_rk / sim_all_rk).sum(1) * (1 / target.sum(1))).mean()

        return (1-ap)
