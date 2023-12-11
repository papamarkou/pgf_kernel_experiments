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
        return [self.single_runners[i].predict(test_x) for i in range(self.num_gps())]

    def assess(self, predictions, test_y, metrics, verbose=True):
        scores = torch.empty([self.num_gps(), len(metrics)], dtype=test_y.dtype, device=test_y.device)

        for i in range(self.num_gps()):
            scores[i, :] = self.single_runners[i].assess(predictions[i], test_y, metrics, verbose=verbose)

        return scores

    def test(self, test_x):
        return [self.single_runners[i].test(test_x) for i in range(self.num_gps())]

    @classmethod
    def generator(selfclass, train_x, train_y, kernels, likelihoods, tasks=None, num_classes=None, use_cuda=True):
        single_runners = []

        if tasks is None:
            tasks = ['regression' for _ in range(len(kernels))]

        for i in range(len(kernels)):
            single_runners.append(ExactSingleGPRunner(
                train_x,
                train_y,
                kernels[i],
                likelihoods[i],
                task=tasks[i],
                num_classes=None if num_classes is None else num_classes[i],
                use_cuda=use_cuda
            ))

        return selfclass(single_runners)
