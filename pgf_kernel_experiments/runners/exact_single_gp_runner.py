import gpytorch
import torch

from pgf_kernel_experiments.models.exact_gp_model import ExactGPModel

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
        with torch.no_grad():
            predictions = self.model.likelihood(self.model(test_x))

        return predictions

    def assess(self, predictions, test_y, metrics, verbose=True):
        scores = torch.empty([len(metrics)], dtype=test_y.dtype, device=test_y.device)

        if verbose:
            msg = ', '.join(['{:.6f}']*len(metrics))

        for i in range(len(metrics)):
            scores[i] = metrics[i](predictions, test_y)

        if verbose:
            print(msg.format(*scores))

        return scores

    def test(self, test_x, metrics=None):
        self.model.setup('test')

        predictions = self.predict(test_x)

        return predictions
