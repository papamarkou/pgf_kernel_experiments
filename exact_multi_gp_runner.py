import gpytorch
import torch

from exact_single_gp_runner import ExactSingleGPRunner

class ExactMultiGPRunner:
    def __init__(self, single_runners):
        self.single_runners = single_runners

    def num_gps(self):
        return len(self.single_runners)

    def step(self, optimizers, train_x, train_y):
        loss = torch.empty([self.num_gps()], dtype=train_x.dtype, device=train_x.device)

        for i in range(self.num_gps()):
            optimizers[i].zero_grad()

            output = self.single_runners[i].model(train_x)

            loss[i] = -self.mll(output, train_y)
            loss[i].backward()

            optimizers[i].step()

        return loss

    def train(self, optimizer, train_x, train_y, num_iters, verbose=True):
        for i in range(self.num_gps()):
            self.single_runners[i].model.setup('train')

        losses = torch.empty([num_iters, self.num_gps()], dtype=train_x.dtype, device=train_x.device)

        if verbose:
            n = len(str(num_iters))
            msg = 'Iteration {:'+str(n)+'d}/{:'+str(n)+'d}, loss: {:.6f}'
            for _ in range(self.num_gps() - 1):
                msg += ', {:.6f}'

        for i in range(num_iters):
            losses[i, :] = self.step(optimizer, train_x, train_y)

            if verbose:
                print(msg.format(i + 1, num_iters, *(losses[i, :])))

        return losses

    def predict(self, test_x):
        predictions = []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(self.num_gps()):
                predictions.append(self.model.likelihood(self.model(test_x)))

        return predictions

    def test(self, test_x):
        for i in range(self.num_gps()):
            self.model.setup('test')

        predictions = self.predict(test_x)

        return predictions

    # Function for constructing single GP runners
