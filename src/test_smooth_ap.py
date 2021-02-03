import pytest
import torch
import torch.nn.functional as F

from losses import SmoothAP
from datasets import create_label_matrix


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


class PaperSmoothAP(torch.nn.Module):
    def __init__(self, anneal, batch_size, num_id, feat_dims):
        super().__init__()

        assert(batch_size % num_id == 0)

        self.anneal = anneal
        self.batch_size = batch_size
        self.num_id = num_id
        self.feat_dims = feat_dims

    def forward(self, preds):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """

        # ------ differentiable ranking of all retrieval set ------
        # compute the mask which ignores the relevance score of the query to itself
        mask = 1.0 - torch.eye(self.batch_size)
        mask = mask.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        sim_all = compute_aff(preds)
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        # pass through the sigmoid
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask.to(preds.device)
        # compute the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1

        # ------ differentiable ranking of only positive set in retrieval set ------
        # compute the mask which only gives non-zero weights to the positive set
        xs = preds.view(self.num_id, int(self.batch_size / self.num_id), self.feat_dims)
        pos_mask = 1.0 - torch.eye(int(self.batch_size / self.num_id))
        pos_mask = pos_mask.unsqueeze(dim=0).unsqueeze(dim=0).repeat(self.num_id, int(self.batch_size / self.num_id), 1, 1)
        # compute the relevance scores
        sim_pos = torch.bmm(xs, xs.permute(0, 2, 1))
        sim_pos_repeat = sim_pos.unsqueeze(dim=2).repeat(1, 1, int(self.batch_size / self.num_id), 1)
        # compute the difference matrix
        sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 1, 3, 2)
        # pass through the sigmoid
        sim_pos_sg = sigmoid(sim_pos_diff, temp=self.anneal) * pos_mask.to(preds.device)
        # compute the rankings of the positive set
        sim_pos_rk = torch.sum(sim_pos_sg, dim=-1) + 1

        # sum the values of the Smooth-AP for all instances in the mini-batch
        ap = torch.zeros(1).to(preds.device)
        group = int(self.batch_size / self.num_id)
        for ind in range(self.num_id):
            pos_divide = torch.sum(sim_pos_rk[ind] / (sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)]))
            ap = ap + ((pos_divide / group) / self.batch_size)

        return (1-ap)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_is_same_loss_easy(seed: int) -> None:
    paper_criterion = PaperSmoothAP(0.01, 90, 5, 128)
    criterion = SmoothAP(0.01)

    torch.manual_seed(seed)
    tens = torch.randn(90, 128)
    tens = F.normalize(tens, dim=1)

    target = []
    for i in range(5):
        target.extend([i]*18)
    target = torch.tensor(target)
    target = create_label_matrix(target).float()

    loss1 = paper_criterion(tens)
    loss2 = criterion(tens, target)

    assert loss1 == loss2


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_is_same_loss_hard(seed: int) -> None:
    paper_criterion = PaperSmoothAP(0.01, 90, 5, 128)
    criterion = SmoothAP(0.01)

    torch.manual_seed(seed)
    tens = torch.randn(90, 128)
    tens = F.normalize(tens, dim=1)

    target = []
    for i in range(5):
        target.extend([i]*18)
    target = torch.tensor(target)
    target = create_label_matrix(target).float()

    mix_tens = torch.zeros_like(tens)
    mix_target = torch.zeros_like(target)
    permute = torch.randperm(mix_tens.size(0))
    for i, j in enumerate(permute):
        mix_tens[i] = tens[j]
        mix_target[i] = target[j][permute]

    loss1 = paper_criterion(tens)
    loss2 = criterion(mix_tens, mix_target)

    torch.isclose(loss1, loss2)
