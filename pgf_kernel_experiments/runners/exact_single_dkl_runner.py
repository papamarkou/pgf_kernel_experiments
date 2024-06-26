import gpytorch
import torch

from pgf_kernel_experiments.models.exact_dkl_model import ExactDKLModel

class ExactSingleDKLRunner:
    def __init__(
        self, train_x, train_y, feature_extractor, kernel, likelihood, task='regression', num_classes=None, use_cuda=True
    ):
        self.model = ExactDKLModel(
            train_x, train_y, feature_extractor, kernel, likelihood, task=task, num_classes=num_classes
        )
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        if use_cuda:
            self.model = self.model.cuda()
            self.mll = self.mll.cuda()

    def step(self, train_x, train_y, optimizer):
        optimizer.zero_grad()

        output, projected_x = self.model.run(train_x)

        if self.model.task == 'regression':
            loss = -self.mll(output, train_y)
        elif self.model.task == 'classification':
            loss = -self.mll(output, train_y).sum()

        loss.backward()

        optimizer.step()

        return loss, projected_x

    def train(self, train_x, train_y, optimizer, num_iters, scheduler=None, verbose=True):
        self.model.setup('train')

        losses = torch.empty([num_iters], dtype=train_x.dtype, device=train_x.device)

        if verbose:
            n = len(str(num_iters))
            msg = 'Iteration {:'+str(n)+'d}/{:'+str(n)+'d}, loss: {:.6f}'

        for i in range(num_iters):
            loss, projected_x = self.step(train_x, train_y, optimizer)
            losses[i] = loss.item()

            if scheduler is not None:
                scheduler.step()

            if verbose:
                print(msg.format(i + 1, num_iters, losses[i]))

        return losses, projected_x

    def predict(self, test_x):
        with torch.no_grad():
            distribution = self.model(test_x)

            if self.model.task == 'regression':
                predictions = self.model.likelihood(distribution)
            elif self.model.task == 'classification':
                predictions = distribution.loc

        return predictions

    def assess(self, predictions, test_y, metrics, dtype=torch.float64, verbose=True):
        scores = torch.empty([len(metrics)], dtype=dtype, device=test_y.device)

        if verbose:
            msg = ', '.join(['{:.6f}']*len(metrics))

        for i in range(len(metrics)):
            scores[i] = metrics[i](predictions, test_y)

        if verbose:
            print(msg.format(*scores))

        return scores

    def test(self, test_x):
        self.model.setup('test')

        predictions = self.predict(test_x)

        return predictions
