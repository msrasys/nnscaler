

########## Generated Code ###########
import torch
import cube


class GenModel(torch.nn.Module):
	
	def __init__(self):
		super().__init__()
		self.weight_14 = torch.nn.Parameter(torch.empty((1024, 16384)))
		self.bias_15 = torch.nn.Parameter(torch.empty((1024,)))
		self.weight_21 = torch.nn.Parameter(torch.empty((1000, 1024)))
		self.bias_22 = torch.nn.Parameter(torch.empty((1000,)))
	
	def su3(self):
		tensor_49 = cube.runtime.collectives.recv([[64, 16384]], [[0]])
		return tensor_49
	
	def su4(self, tensor_49):
		tensor_58 = torch.nn.functional.linear(tensor_49, self.weight_14, self.bias_15)
		tensor_64 = torch.nn.functional.linear(tensor_58, self.weight_21, self.bias_22)
		tensor_68 = torch.sum(tensor_64)
		return tensor_68
	
	def su6(self, tensor_89):
		cube.runtime.collectives.send((tensor_89, ), [[0]])
		return 
	
	def su12(self):
		tensor_228 = cube.runtime.collectives.recv([[64, 16384]], [[0]])
		return tensor_228
	
	def su13(self, tensor_228):
		tensor_237 = torch.nn.functional.linear(tensor_228, self.weight_14, self.bias_15)
		tensor_243 = torch.nn.functional.linear(tensor_237, self.weight_21, self.bias_22)
		tensor_247 = torch.sum(tensor_243)
		return tensor_247
	
	def su15(self, tensor_268):
		cube.runtime.collectives.send((tensor_268, ), [[0]])
		return 
	
	def su21(self):
		tensor_407 = cube.runtime.collectives.recv([[64, 16384]], [[0]])
		return tensor_407
	
	def su22(self, tensor_407):
		tensor_416 = torch.nn.functional.linear(tensor_407, self.weight_14, self.bias_15)
		tensor_422 = torch.nn.functional.linear(tensor_416, self.weight_21, self.bias_22)
		tensor_426 = torch.sum(tensor_422)
		return tensor_426
	
	def su24(self, tensor_447):
		cube.runtime.collectives.send((tensor_447, ), [[0]])
		return 
	
	def su30(self):
		tensor_586 = cube.runtime.collectives.recv([[64, 16384]], [[0]])
		return tensor_586
	
	def su31(self, tensor_586):
		tensor_595 = torch.nn.functional.linear(tensor_586, self.weight_14, self.bias_15)
		tensor_601 = torch.nn.functional.linear(tensor_595, self.weight_21, self.bias_22)
		tensor_605 = torch.sum(tensor_601)
		return tensor_605
	
	def su33(self, tensor_626):
		cube.runtime.collectives.send((tensor_626, ), [[0]])
		return 


########## Generated Code ###########
import torch
import cube

def _train_step(model, dataloader):
	tensor_49 = cube.runtime.temporal.forward(model.su3, *())
	tensor_68 = cube.runtime.temporal.forward(model.su4, *(tensor_49, ))
	tensor_89 = cube.runtime.temporal.backward((tensor_49, ), (tensor_68, ), (None, ))
	cube.runtime.temporal.forward(model.su6, *(tensor_89, ))
	tensor_228 = cube.runtime.temporal.forward(model.su12, *())
	tensor_247 = cube.runtime.temporal.forward(model.su13, *(tensor_228, ))
	tensor_268 = cube.runtime.temporal.backward((tensor_228, ), (tensor_247, ), (None, ))
	cube.runtime.temporal.forward(model.su15, *(tensor_268, ))
	tensor_407 = cube.runtime.temporal.forward(model.su21, *())
	tensor_426 = cube.runtime.temporal.forward(model.su22, *(tensor_407, ))
	tensor_447 = cube.runtime.temporal.backward((tensor_407, ), (tensor_426, ), (None, ))
	cube.runtime.temporal.forward(model.su24, *(tensor_447, ))
	tensor_586 = cube.runtime.temporal.forward(model.su30, *())
	tensor_605 = cube.runtime.temporal.forward(model.su31, *(tensor_586, ))
	tensor_626 = cube.runtime.temporal.backward((tensor_586, ), (tensor_605, ), (None, ))
	cube.runtime.temporal.forward(model.su33, *(tensor_626, ))
