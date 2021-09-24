

########## Generated Code ###########
import torch


class GenModel(torch.nn.Module):
	
	def __init__(self):
		super().__init__()
		self.weight_10 = torch.nn.Parameter(torch.empty((1024, 16384)))
		self.bias_11 = torch.nn.Parameter(torch.empty((1024,)))
		self.weight_14 = torch.nn.Parameter(torch.empty((1000, 1024)))
		self.bias_15 = torch.nn.Parameter(torch.empty((1000,)))
	
	def forward(self, tensor_8):
		tensor_12 = torch.nn.functional.linear(tensor_8, self.weight_10, self.bias_11)
		tensor_16 = torch.nn.functional.linear(tensor_12, self.weight_14, self.bias_15)
		tensor_17 = torch.sum(tensor_16)
		return tensor_17


########## Generated Code ###########
import torch
import cube

def _train_step(model, dataloader):
	tensor_21 = cube.runtime.collectives.recv(([64, 16384],), [[0]])
	tensor_23 = cube.runtime.temporal.forward(model, *(tensor_21, ))
	tensor_51 = cube.runtime.temporal.backward((tensor_21, ), (tensor_23, ), (None, ))
	tensor_54 = cube.runtime.collectives.send_and_recv((tensor_51, ), [[0]], ([64, 16384],), [[0]])  # send: ([64, 16384],)
	tensor_56 = cube.runtime.temporal.forward(model, *(tensor_54, ))
	tensor_84 = cube.runtime.temporal.backward((tensor_54, ), (tensor_56, ), (None, ))
	tensor_87 = cube.runtime.collectives.send_and_recv((tensor_84, ), [[0]], ([64, 16384],), [[0]])  # send: ([64, 16384],)
	tensor_89 = cube.runtime.temporal.forward(model, *(tensor_87, ))
	tensor_117 = cube.runtime.temporal.backward((tensor_87, ), (tensor_89, ), (None, ))
	tensor_120 = cube.runtime.collectives.send_and_recv((tensor_117, ), [[0]], ([64, 16384],), [[0]])  # send: ([64, 16384],)
	tensor_122 = cube.runtime.temporal.forward(model, *(tensor_120, ))
	tensor_150 = cube.runtime.temporal.backward((tensor_120, ), (tensor_122, ), (None, ))
	cube.runtime.collectives.send((tensor_150, ), [[0]])  # send: ([64, 16384],)
