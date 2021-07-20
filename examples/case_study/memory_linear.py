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
def swap_linear(input, weight, bias):
    ## Note pytorch tensor.to() will always return a copy

    ### Policy ###
    op_device_id = 0       # where to perform the device
    output_swap = True      # whether output tensor needs swap

    ### Additional Swap operator ###
    # Note autograd will not work in pytorch
    # as pytorch will record each input, even you do the inplacement-update
    # class SwapOutTensor(torch.autograd.Function):
    #     @staticmethod
    #     def forward(ctx, tensor):
    #         ctx.constants = tensor.get_device()
    #         cpu_tensor = tensor.cpu()
    #         tensor.data = cpu_tensor  # inplace-update
    #         return tensor
    #     @staticmethod
    #     def backward(ctx, grad_output):
    #         device_id = ctx.constants
    #         grad = grad_output.cuda(device_id)
    #         grad_output.data = grad
    #         return grad_output

    ### Input swap-in (if needed) ###
    input_swap = None
    if input.get_device() != op_device_id:
        input = input.cuda(op_device_id)
        input_swap = -1 # CPU
    weight_swap = None
    if weight.get_device() != op_device_id:
        weight_swap = -1 # CPU
        weight = weight.cuda(op_device_id)
    bias_swap = None
    if bias.get_device() != op_device_id:
        bias_swap = -1 # CPU
        bias = bias.cuda(op_device_id)

    ### Compute ###
    output = torch._C._nn.linear(input, weight, bias)
    print(output)
    # inplacement update
    output.data = output.cpu()
    print(output)

    # Here we need the backward to take back the intermediate tensor

    ### Swap out if needed ### TODO: swapout can be in any place
    # if output_swap:
    #     output = SwapOutTensor.apply(output)
    # if input_swap == -1:
    #     input_swap = SwapOutTensor.apply(input, )
    return output



if __name__ == '__main__':

    torch.cuda.set_device(0)

    # tensor definition
    batch_size = 32
    out_features = 1024
    in_features = 1024
    weight = torch.rand((out_features, in_features)).cuda().requires_grad_()
    # print('weight: ', weight)
    bias = torch.rand(out_features).cuda().requires_grad_()
    # print('bias: ', bias)
    input = torch.rand((batch_size, in_features)).cuda()
    # print('input: ', input)
    
    # op compute
    print('======== Checkpointing Single Device =======')
    output = swap_linear(input, weight, bias)
    print('output device: {}'.format(output.get_device()))
    print(output)
    output = output.cuda()
    print('output device: {}'.format(output.get_device()))
    print(output)
    loss = torch.mean(output)
    print(loss)
    loss.backward()
    print('weight grad: ', weight.grad.t())
    print('======== Checkpointing Single Device =======')
