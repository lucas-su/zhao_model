import torch

class confusion:
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, n_classes: int = 10):
        self._matrix = torch.zeros(n_classes * n_classes).to(self._device)
        self._n = n_classes

    def cpu(self):
        self._matrix.cpu()

    def cuda(self):
        self._matrix.cuda()

    def to(self, device: str):
        self._matrix.to(device)

    def __add__(self, other):
        if isinstance(other, ConfusionMatrix):
            self._matrix.add_(other._matrix)
        elif isinstance(other, tuple):
            self.update(*other)
        else:
            raise NotImplemented
        return self

    def update(self, prediction: torch.tensor, label: torch.tensor):
        conf_data = prediction * self._n + label
        conf = conf_data.bincount(minlength=self._n * self._n)
        self._matrix.add_(conf)

    @property
    def value(self):
        return self._matrix.view(self._n, self._n).T