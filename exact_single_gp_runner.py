import gpytorch
import torch

from exact_gp_model import ExactGPModel

class ExactSingleGPRunner:
    def __init__(self, train_x, train_y, kernel, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        self.model = ExactGPModel(train_x, train_y, kernel, likelihood=likelihood)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def step(self, train_x, train_y, optimizer):
        optimizer.zero_grad()

        output = self.model(train_x)

        loss = -self.mll(output, train_y)
        loss.backward()

        optimizer.step()

        return loss

    def train(self, train_x, train_y, optimizer, num_iters, verbose=True):
        self.model.setup('train')

        losses = torch.empty([num_iters], dtype=train_x.dtype, device=train_x.device)

        if verbose:
            n = len(str(num_iters))
            msg = 'Iteration {:'+str(n)+'d}/{:'+str(n)+'d}, loss: {:.6f}'

        for i in range(num_iters):
            losses[i] = self.step(train_x, train_y, optimizer).item()

            if verbose:
                print(msg.format(i + 1, num_iters, losses[i]))

        return losses

    def predict(self, test_x):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.model.likelihood(self.model(test_x))

        return predictions

    def test(self, test_x):
        self.model.setup('test')

        predictions = self.predict(test_x)

        return predictions
