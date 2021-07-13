import torch


class WindowLoss(torch.nn.Module):
    def __init__(self, metric, window_size=32, stride=16, batch_reduction='mean'):
        super(WindowLoss, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.metric = metric
        # self.metric.batch_reduction = 'mean'
        self.batch_reduction = batch_reduction
        self.unfold = torch.nn.Unfold(kernel_size=window_size, stride=stride)
        self.name = f"Windowed(w-{window_size}-{stride})-{self.metric.name}"

    def forward(self, x, y):
        b, c = x.size(0), x.size(1)
        # (batch, c,  p,  x p n_patches)
        x_uf = self.unfold(x).transpose(1, 2).reshape(b, -1, c, self.window_size, self.window_size)
        y_uf = self.unfold(y).transpose(1, 2).reshape(b, -1, c, self.window_size, self.window_size)

        results = torch.stack([self.metric(x_uf[i], y_uf[i]) for i in range(b)])
        if self.batch_reduction == 'mean':
            results = results.mean()

        return results
