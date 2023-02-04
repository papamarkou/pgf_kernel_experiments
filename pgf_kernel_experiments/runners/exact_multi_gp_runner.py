import gpytorch
import torch

from .exact_single_gp_runner import ExactSingleGPRunner

class ExactMultiGPRunner:
    def __init__(self, single_runners):
        self.single_runners = single_runners

    def num_gps(self):
        return len(self.single_runners)

    def step(self, train_x, train_y, optimizers):
        losses = torch.empty([self.num_gps()], dtype=train_x.dtype, device=train_x.device)

        for i in range(self.num_gps()):
            optimizers[i].zero_grad()

            output = self.single_runners[i].model(train_x)

            loss = -self.single_runners[i].mll(output, train_y)
            loss.backward()

            optimizers[i].step()

            losses[i] = loss

        return losses

    def train(self, train_x, train_y, optimizers, num_iters, verbose=True):
        for i in range(self.num_gps()):
            self.single_runners[i].model.setup('train')

        losses = torch.empty([num_iters, self.num_gps()], dtype=train_x.dtype, device=train_x.device)

        if verbose:
            n = len(str(num_iters))
            msg = 'Iteration {:'+str(n)+'d}/{:'+str(n)+'d}, loss: {:.6f}'
            for _ in range(self.num_gps() - 1):
                msg += ', {:.6f}'

        for i in range(num_iters):
            losses[i, :] = self.step(train_x, train_y, optimizers)

            if verbose:
                print(msg.format(i + 1, num_iters, *(losses[i, :])))

        return losses

    def predict(self, test_x):
        predictions = []

        with torch.no_grad():
            for i in range(self.num_gps()):
                predictions.append(self.single_runners[i].model.likelihood(self.single_runners[i].model(test_x)))

        return predictions

    def assess(self, predictions, test_y, metrics, verbose=True):
        scores = torch.empty([self.num_gps(), len(metrics)], dtype=test_y.dtype, device=test_y.device)

        if verbose:
            msg = ', '.join(['{:.6f}']*len(metrics))

        for i in range(self.num_gps()):
            for j in range(len(metrics)):
                scores[i, j] = metrics[j](predictions[i], test_y)

            if verbose:
                print(msg.format(*(scores[i, :])))

        return scores

    def test(self, test_x):
        for i in range(self.num_gps()):
            self.single_runners[i].model.setup('test')

        predictions = self.predict(test_x)

        return predictions

    @classmethod
    def generator(selfclass, train_x, train_y, kernels, likelihoods=None):
        if likelihoods is None:
            likelihoods = [gpytorch.likelihoods.GaussianLikelihood() for _ in range(len(kernels))]

        single_runners = []

        for i in range(len(kernels)):
            single_runners.append(ExactSingleGPRunner(train_x, train_y, kernels[i], likelihood=likelihoods[i]))

        return selfclass(single_runners)
