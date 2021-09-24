

########## Generated Code ###########
import torch


class GenModel(torch.nn.Module):
	
	def __init__(self):
		super().__init__()
		self.weight_3 = torch.nn.Parameter(torch.empty((16384, 1024)))
	
	def forward(self, data_1):
		tensor_4 = torch.nn.functional.linear(data_1, self.weight_3, None)
		tensor_6 = torch.nn.functional.gelu(tensor_4)
		tensor_8 = torch.nn.functional.dropout(tensor_6, 0.0, self.training, False)
		return tensor_8


########## Generated Code ###########
import torch
import cube

def _train_step(model, dataloader):
	tensor_21 = cube.runtime.temporal.forward(model, *(*next(dataloader), ))
	tensor_51 = cube.runtime.collectives.send_and_recv((tensor_21, ), [[1]], ([64, 16384],), [[1]])  # send: ([64, 16384],)
	cube.runtime.temporal.backward((), (tensor_21, ), (tensor_51, ))
	tensor_54 = cube.runtime.temporal.forward(model, *(*next(dataloader), ))
	tensor_84 = cube.runtime.collectives.send_and_recv((tensor_54, ), [[1]], ([64, 16384],), [[1]])  # send: ([64, 16384],)
	cube.runtime.temporal.backward((), (tensor_54, ), (tensor_84, ))
	tensor_87 = cube.runtime.temporal.forward(model, *(*next(dataloader), ))
	tensor_117 = cube.runtime.collectives.send_and_recv((tensor_87, ), [[1]], ([64, 16384],), [[1]])  # send: ([64, 16384],)
	cube.runtime.temporal.backward((), (tensor_87, ), (tensor_117, ))
	tensor_120 = cube.runtime.temporal.forward(model, *(*next(dataloader), ))
	tensor_150 = cube.runtime.collectives.send_and_recv((tensor_120, ), [[1]], ([64, 16384],), [[1]])  # send: ([64, 16384],)
	cube.runtime.temporal.backward((), (tensor_120, ), (tensor_150, ))
