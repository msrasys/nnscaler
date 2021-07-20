import torch
import os

torch.manual_seed(121)

### Checkpoint PyTorch Implementation (Skip un-deterministic scenario) ###
# Note this implementation can only work with a module that consists
# multiple operators. This will won't work for one OP because the output
# for this module will be saved in next op
def checkpoint_module_linear(input, weight, bias):

    class Checkpoint(torch.autograd.Function):
        """General class to wrapper op to enable checkpoint"""
        @staticmethod
        def forward(ctx, run_function, *args):
            ctx.run_function = run_function
            ctx.tensor_indices = []
            tensor_inputs = []
            for i, arg in enumerate(args):
                if torch.is_tensor(arg):
                    tensor_inputs.append(arg)
                    ctx.tensor_indices.append(i)
                    ctx.inputs.append(None)
                else:
                    ctx.inputs.append(arg)
            ctx.save_for_backward(*tensor_inputs)

            with torch.no_grad():
                outputs = run_function(*args)
            return outputs
        @staticmethod
        def backward(ctx, *args):
            # retrieve what need to regenerate tensors
            inputs = list(ctx.inputs)
            tensor_indices = ctx.tensor_indices
            tensors = ctx.saved_tensors
            # re-generate
            for i, idx in enumerate(tensor_indices):
                inputs[idx] = tensors[i]
            # detach inputs
            detached_inputs = list()
            for input in inputs:
                if torch.is_tensor(input):
                    x = input.detach()
                    x.requires_grad = input.requires_grad
                else:
                    x = input
                detached_inputs.append(x)
            detached_inputs = tuple(detached_inputs)
            # generate output tensor
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs)
            if torch.is_tensor(outputs):
                outputs = (outputs,)
            # run backward to tensors that require a grad
            outputs_with_grad = list()
            args_with_grad = list()
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
            torch.autograd.backward(outputs_with_grad, args_with_grad)
            grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None
                      for inp in detached_inputs)
            return (None, None) + grads

    output = Checkpoint.apply(torch._C._nn.linear, input, weight, bias)
    return output


### Swap linear ###
def swap_weight_grad_linear(input, weight, bias):
    ## Note pytorch tensor.to() will always return a copy

    ### Policy ###
    op_device_id = 0        # where to perform the device
    # output_swap = False     # whether output tensor needs swap
    weight_swap = True
    bias_swap = True
    gradient_swap = True

    ### Input swap-in (if needed) ###
    weight_locate = weight.get_device()
    if weight_locate == -1:
        weight.data = weight.cuda(op_device_id)
    bias_locate = bias.get_device()
    if bias_locate == -1:  # current on CPU
        bias.data = bias.cuda(op_device_id)
    
    ### Adatper to swap out gradient ###
    def swap_out_grad(grad):
        grad.data = grad.detach().cpu()
        return grad
    if gradient_swap:
        weight.register_hook(swap_out_grad)
        bias.register_hook(swap_out_grad)

    ### Compute ###
    output = torch._C._nn.linear(input, weight, bias)
    # inplacement swap
    # output.data = output.cpu()

    ### Swap out if needed ### TODO: swapout can be in any place
    if weight_swap:
        weight.data = weight.detach().cpu()
    if bias_swap:
        bias.data = bias.detach().cpu()
    # print(weight)
    # print(bias)

    return output



if __name__ == '__main__':

    torch.cuda.set_device(0)
    init_memory = torch.cuda.memory_allocated()

    # tensor definition
    batch_size = 32
    out_features = 10240
    in_features = 10240  ## 100 MB weight
    weight = torch.rand((out_features, in_features)).cuda().requires_grad_()
    # print('weight: ', weight)
    bias = torch.rand(out_features).cuda().requires_grad_()
    input = torch.rand((batch_size, in_features)).cuda()

    input_memory = (torch.cuda.memory_allocated() - init_memory) / 1024 / 1024
    
    # op compute
    print('======== Checkpointing Single Device =======')
    
    # swap out weight
    weight.data = weight.detach().cpu()
    bias.data = bias.detach().cpu()

    weight_swap_memory = (torch.cuda.memory_allocated() - init_memory) / 1024 / 1024

    output = swap_weight_grad_linear(input, weight, bias)
    loss = torch.mean(output) * 100
    loss.backward()
    
    finish_op_memory = (torch.cuda.memory_allocated() - init_memory) / 1024 / 1024
    
    # allocate tensor on gpu to see if swap workds
    tmp = torch.rand((out_features, in_features)).cuda()
    after_alloc_memory = (torch.cuda.memory_allocated() - init_memory) / 1024 / 1024

    print('memory consumption (MB): input-require: {:.2f} | after swap weight: {:.2f} | after op run {:.2f} | after allocate {:.2f}'.format(
        input_memory, weight_swap_memory, finish_op_memory, after_alloc_memory))

    # correctness verify
    print('weight grad: ', weight.grad.t())
    print('======== Checkpointing Single Device =======')
