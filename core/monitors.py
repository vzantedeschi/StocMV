import torch

from tensorboardX import SummaryWriter

class MonitorMV():

    def __init__(self, logdir):

        super(MonitorMV, self).__init__()

        self.logdir = logdir
        self.writer = SummaryWriter(logdir)

        self.it = 0

    def write_all(self, it, posterior, gradient, **metrics):

        self.it = it

        self.writer.add_scalars('variables/posterior', 
            { 
             "l2": torch.norm(posterior, p=2),
             "l1": torch.norm(posterior, p=1),
             }, it)

        self.writer.add_scalars('variables/post_grad', 
            { 
             "l2": torch.norm(gradient, p=2),
             }, it)

        self.write(it, **metrics)

    def write(self, it=None, **metrics):
        
        if it is None:
            it = self.it

        for key, item in metrics.items():
            self.writer.add_scalars(key, item, it)

        self.it += 1

    def close(self, logfile="monitor_scalars.json"):
        self.writer.export_scalars_to_json(self.logdir / logfile)
        self.writer.close()
