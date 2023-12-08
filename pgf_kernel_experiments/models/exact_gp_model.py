import gpytorch
import torch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood, num_classes=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        self.num_classes = num_classes

        if self.num_classes is None:
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size((self.num_classes,)))

        self.covar_module = kernel

        self.likelihood = likelihood

    def setup(self, mode):
        if mode == 'train':
            self.train()
            self.likelihood.train()
        elif mode == 'test':
            self.eval()
            self.likelihood.eval()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
