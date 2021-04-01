import torch

from tensorboardX import SummaryWriter

class MonitorMV():

    def __init__(self, logdir, normalize=False):

        super(MonitorMV, self).__init__()

        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        self.normalize = normalize

    def write_all(self, it, posterior, **metrics):

        if self.normalize:
            p = posterior / posterior.sum()    
        else:
            p = posterior

        self.writer.add_scalars('variables/posterior', 
            { 
             "l2": torch.norm(p, p=2),
             "l1": torch.norm(p, p=1),
             }, it)

        self.write(it, **metrics)

    def write(self, it, **metrics):
        for key, item in metrics.items():
            self.writer.add_scalars(key, item, it)

    def close(self, logfile="monitor_scalars.json"):
        self.writer.export_scalars_to_json(self.logdir / logfile)
        self.writer.close()
