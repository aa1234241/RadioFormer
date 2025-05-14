import torch
import torch.nn as nn
import torch.nn.functional as F
class Codebook(nn.Module):
    def __init__(self):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = 512
        self.latent_dim = 384
        self.beta = 0.25

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        self.embedding.weight = torch.nn.Parameter(F.normalize(self.embedding.weight, dim=1))
        z = F.normalize(z, dim=-1)

        d = torch.sum(z**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
        # print(min_encoding_indices, torch.mean((z_q.detach() - z)**2))
        z_q = z + (z_q - z).detach()
        return z_q, min_encoding_indices, loss