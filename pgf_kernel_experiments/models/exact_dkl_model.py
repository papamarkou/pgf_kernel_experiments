import gpytorch
import torch

class ExactDKLModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        feature_extractor,
        kernel,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        num_classes=None
    ):
        super(ExactDKLModel, self).__init__(train_x, train_y, likelihood)

        self.feature_extractor = feature_extractor

        if num_classes is None:
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size((num_classes,)))

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
