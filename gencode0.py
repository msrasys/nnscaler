

########## Generated Code ###########
import torch
import cube


class GenModel(torch.nn.Module):
	
	def __init__(self):
		super().__init__()
		self.weight_3 = torch.nn.Parameter(torch.empty((16384, 1024)))
	
	def su1(self, data_36):
		tensor_41 = torch.nn.functional.linear(data_36, self.weight_3, None)
		tensor_45 = torch.nn.functional.gelu(tensor_41)
		tensor_49 = torch.nn.functional.dropout(tensor_45, 0.0, self.training, False)
		return tensor_49
	
	def su2(self, tensor_49):
		cube.runtime.collectives.send((tensor_49, ), [[1]])
		return 
	
	def su7(self):
		tensor_89 = cube.runtime.collectives.recv([[64, 16384]], [[1]])
		return tensor_89
	
	def su10(self, data_215):
		tensor_220 = torch.nn.functional.linear(data_215, self.weight_3, None)
		tensor_224 = torch.nn.functional.gelu(tensor_220)
		tensor_228 = torch.nn.functional.dropout(tensor_224, 0.0, self.training, False)
		return tensor_228
	
	def su11(self, tensor_228):
		cube.runtime.collectives.send((tensor_228, ), [[1]])
		return 
	
	def su16(self):
		tensor_268 = cube.runtime.collectives.recv([[64, 16384]], [[1]])
		return tensor_268
	
	def su19(self, data_394):
		tensor_399 = torch.nn.functional.linear(data_394, self.weight_3, None)
		tensor_403 = torch.nn.functional.gelu(tensor_399)
		tensor_407 = torch.nn.functional.dropout(tensor_403, 0.0, self.training, False)
		return tensor_407
	
	def su20(self, tensor_407):
		cube.runtime.collectives.send((tensor_407, ), [[1]])
		return 
	
	def su25(self):
		tensor_447 = cube.runtime.collectives.recv([[64, 16384]], [[1]])
		return tensor_447
	
	def su28(self, data_573):
		tensor_578 = torch.nn.functional.linear(data_573, self.weight_3, None)
		tensor_582 = torch.nn.functional.gelu(tensor_578)
		tensor_586 = torch.nn.functional.dropout(tensor_582, 0.0, self.training, False)
		return tensor_586
	
	def su29(self, tensor_586):
		cube.runtime.collectives.send((tensor_586, ), [[1]])
		return 
	
	def su34(self):
		tensor_626 = cube.runtime.collectives.recv([[64, 16384]], [[1]])
		return tensor_626


########## Generated Code ###########
import torch
import cube

def _train_step(model, dataloader):
	data_36 = next(dataloader)
	tensor_49 = cube.runtime.temporal.forward(model.su1, *(data_36, ))
	cube.runtime.temporal.forward(model.su2, *(tensor_49, ))
	tensor_89 = cube.runtime.temporal.forward(model.su7, *())
	data_78 = cube.runtime.temporal.backward((data_36, ), (tensor_49, ), (tensor_89, ))
	data_215 = next(dataloader)
	tensor_228 = cube.runtime.temporal.forward(model.su10, *(data_215, ))
	cube.runtime.temporal.forward(model.su11, *(tensor_228, ))
	tensor_268 = cube.runtime.temporal.forward(model.su16, *())
	data_257 = cube.runtime.temporal.backward((data_215, ), (tensor_228, ), (tensor_268, ))
	data_394 = next(dataloader)
	tensor_407 = cube.runtime.temporal.forward(model.su19, *(data_394, ))
	cube.runtime.temporal.forward(model.su20, *(tensor_407, ))
	tensor_447 = cube.runtime.temporal.forward(model.su25, *())
	data_436 = cube.runtime.temporal.backward((data_394, ), (tensor_407, ), (tensor_447, ))
	data_573 = next(dataloader)
	tensor_586 = cube.runtime.temporal.forward(model.su28, *(data_573, ))
	cube.runtime.temporal.forward(model.su29, *(tensor_586, ))
	tensor_626 = cube.runtime.temporal.forward(model.su34, *())
	data_615 = cube.runtime.temporal.backward((data_573, ), (tensor_586, ), (tensor_626, ))
