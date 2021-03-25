from tensorboardX import SummaryWriter

import jax.numpy.linalg as jlin
class MonitorMV():

    def __init__(self, logdir=None):

        super(MonitorMV, self).__init__()

        self.logdir = logdir
        self.writer = SummaryWriter(logdir)

    def write_all(self, it, alpha, grad, **metrics):

        self.writer.add_scalars('variables/alpha', 
            { 
             "l2": jlin.norm(alpha, ord=2),
             }, it)

        self.writer.add_scalars('variables/gradient', 
            { 
             "l2": jlin.norm(grad, ord=2),
             }, it)

        self.write(it, **metrics)

    def write(self, it, **metrics):
        for key, item in metrics.items():
            self.writer.add_scalars(key, item, it)

    def close(self, logfile="monitor_scalars.json"):
        self.writer.export_scalars_to_json(self.logdir / logfile)
        self.writer.close()