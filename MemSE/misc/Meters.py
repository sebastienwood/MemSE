import time
from .Timer import Timer

__all__ = ['AverageMeter', 'HistMeter', 'TimingMeter', 'ProgressMeter']

class AverageMeter(object):
	"""Computes and stores the average and current value
	https://github.com/pytorch/examples/blob/2c57b0011a096aef83da3b5265a14db2f80cb124/imagenet/main.py#L363
	"""
	__slots__ = ('name', 'fmt', 'val', 'avg', 'sum', 'count')
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)


class HistMeter(AverageMeter):
	__slots__ = ('hist', 'histed')
	def __init__(self, name, fmt=':f', histed='sum'):
		super().__init__(name, fmt)
		self.histed = histed
		self.hist = []

	def update(self, val, n=1):
		super().update(val, n)
		self.hist.append(getattr(self, self.histed))


class TimingMeter(HistMeter, Timer):
	def __init__(self, name, fmt=':f'):
		HistMeter.__init__(self, name, fmt)
		Timer.__init__(self)

	def __exit__(self):
		Timer.__exit__(self)
		HistMeter.update(self, Timer.__call__(self))


class ProgressMeter(object):
	def __init__(self, num_batches, meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.num_batches = num_batches
		self.start_time = None
		self.meters = meters
		self.prefix = prefix

	def display(self, batch):
		if self.start_time is None:
			self.start_time = time.time()
		ETA = (self.num_batches - batch) * (time.time() - self.start_time) / batch
		entries = [self.prefix + self.batch_fmtstr.format(batch) + f'({ETA=})']
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'