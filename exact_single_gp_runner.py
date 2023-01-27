import gpytorch
import torch

from exact_gp_model import ExactGPModel
from metrics import mae

class ExactSingleGPRunner:
    def __init__(self, train_x, train_y, kernel, likelihood=gpytorch.likelihoods.GaussianLikelihood(), metric=mae):
        self.model = ExactGPModel(train_x, train_y, kernel, likelihood=likelihood)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        self.metric = metric

    def step(self, optimizer, train_x, train_y):
        optimizer.zero_grad()

        output = self.model(train_x)

        loss = -self.mll(output, train_y)
        loss.backward()

        optimizer.step()

        return loss

    def train(self, optimizer, train_x, train_y, num_iters):
        self.model.setup('train')

        losses = torch.empty([num_iters], dtype=train_x.dtype, device=train_x.device)

        for i in range(num_iters):
            losses[i] = self.step(optimizer, train_x, train_y).item()

        return losses

    def predict(self, test_x):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(test_x))

        return predictions

    def test(self, test_x, test_y):
        self.model.setup('test')

        predictions = self.predict(test_x)

        error = self.metric(predictions.mean, test_y)

        return error, predictions
