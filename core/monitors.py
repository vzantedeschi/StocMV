import torch

from tensorboardX import SummaryWriter

class MonitorMV():

    def __init__(self, logdir, normalize=False):

        super(MonitorMV, self).__init__()

        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        self.normalize = normalize

    def write_all(self, it, alpha, grad, **metrics):

        if self.normalize:
            alpha_norm = torch.norm(alpha / alpha.sum(), p=2)
        else:
            alpha_norm = torch.norm(alpha, p=2)

        self.writer.add_scalars('variables/alpha', 
            { 
             "l2": float(alpha_norm),
             }, it)

        self.writer.add_scalars('variables/gradient', 
            { 
             "l2": float(torch.norm(grad, p=2)),
             }, it)

        self.write(it, **metrics)

    def write(self, it, **metrics):
        for key, item in metrics.items():
            self.writer.add_scalars(key, item, it)

    def close(self, logfile="monitor_scalars.json"):
        self.writer.export_scalars_to_json(self.logdir / logfile)
        self.writer.close()
