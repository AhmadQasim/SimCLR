import torch
import torch.nn as nn


class NT_Xent(nn.Module):
    """
    The normalized temperature-scaled cross entropy loss
    """
    def __init__(self, batch_size, temperature, device):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)
        self.device = device

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1)
        augmented examples within a minibatch as negative examples.
        """
        # doc: all the comments underneath are to be considered for a batch size of 128 unless specified otherwise
        p1 = torch.cat((z_i, z_j), dim=0)

        # doc: here the cosine similarity dim is 2. This works a bit differently from dimension-wise sum for example.
        # p1.shape = [256, 1, 64] and p2.shape = [1, 256, 64], when finding cosine similarity the first two dimensions
        # are iterated while taking the whole vector from the third dimension
        sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature

        # doc: suppose index for, p1 = [1, 2, 3, 4] where z_i = [1, 2] and z_j = [3, 4] and batch size = 2
        # then the similarity matrix will look like (in terms of indexes)
        # [11, 12, 13, 14]
        # [21, 22, 23, 24]
        # [31, 32, 33, 34]
        # [41, 42, 43, 44]
        # then torch.diag(sim, 2) = [13, 24] and torch.diag(sim, -2) = [31, 42] hence the positive samples
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # doc: concatenate the positive samples
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            self.batch_size * 2, 1
        )

        # doc: here the self.mask filters out the main diagonals which constitute the same samples
        # and also the minor diagonals of batch size and -batch size (look above)
        negative_samples = sim[self.mask].reshape(self.batch_size * 2, -1)

        labels = torch.zeros(self.batch_size * 2).to(self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)

        # doc: normalize the loss i.e. 1/2N
        loss /= 2 * self.batch_size
        return loss
