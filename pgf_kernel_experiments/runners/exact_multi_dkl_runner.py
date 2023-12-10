import gpytorch
import torch

from .exact_single_dkl_runner import ExactSingleDKLRunner

class ExactMultiDKLRunner:
    def __init__(self, single_runners):
        self.single_runners = single_runners

    def num_gps(self):
        return len(self.single_runners)

    def step(self, train_x, train_y, optimizers):
        losses = torch.empty([self.num_gps()], dtype=train_x.dtype, device=train_x.device)

        for i in range(self.num_gps()):
            losses[i] = self.single_runners[i].step(train_x, train_y, optimizers[i])

        return losses

    def train(self, train_x, train_y, optimizers, num_iters, schedulers=None, verbose=True):
        if schedulers is None:
            schedulers = [None for i in range(self.num_gps())]

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

            for j in range(self.num_gps()):
                if schedulers[j] is not None:
                    schedulers[j].step()

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
    def generator(
        selfclass, train_x, train_y, feature_extractors, kernels, likelihoods, tasks=None, num_classes=None, use_cuda=True
    ):
        single_runners = []

        if tasks is None:
            tasks = ['regression' for _ in range(len(kernels))]

        for i in range(len(kernels)):
            single_runners.append(ExactSingleDKLRunner(
                train_x,
                train_y,
                feature_extractors[i],
                kernels[i],
                likelihoods[i],
                task=tasks[i],
                num_classes=None if num_classes is None else num_classes[i],
                use_cuda=use_cuda)
            )

        return selfclass(single_runners)
