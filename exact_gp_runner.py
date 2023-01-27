import gpytorch
import torch

from exact_gp_model import ExactGPModel

class ExactGPRunner:
    def __init__(self, train_x, train_y, kernel, likelihood=None):
        self.model = ExactGPModel(train_x, train_y, likelihood=likelihood)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

    def step(self, optimizer):
        optimizer.zero_grad()

        output = self.model(self.train_x)

        loss = -self.mll(output, self.train_y)
        loss.backward()

        optimizer.step()

        return loss

    def train(self, optimizer, num_iters):
        self.model.setup('train')

        for i in range(num_iters):
            loss = self.step(optimizer)

        return loss

    def predict(self, test_x):
        return self.likelihood(self.model(test_x))

    def test(self, test_x, test_y):
        self.model.setup('test')

        predictions = self.predict(test_x)

        error = torch.mean(torch.abs(predictions.mean - test_y))

        return predictions, error
