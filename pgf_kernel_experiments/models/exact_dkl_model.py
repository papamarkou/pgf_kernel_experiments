import gpytorch
import torch

class ExactDKLModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, feature_extractor, kernel, likelihood, task='regression', num_classes=None):
        super(ExactDKLModel, self).__init__(train_x, train_y, likelihood)

        self.task = task
        self.num_classes = num_classes

        self.feature_extractor = feature_extractor

        if self.task == 'regression':
            self.mean_module = gpytorch.means.ConstantMean()
        elif self.task == 'classification':
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
        projected_x = self.feature_extractor(x)

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
