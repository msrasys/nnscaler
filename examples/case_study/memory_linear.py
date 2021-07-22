import torch
import os

torch.manual_seed(121)

tensor_map = dict()

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


def swap_weight_grad_linear(input, weight, bias):

    ### Policy ###

    # op placement
    op_device = torch.device('cuda:0')

    # tensor placement: this should be set at tensor creation stage
    # note here if change this, we also need to change tensor init at main
    weight.host_device = torch.device('cpu')
    bias.host_device = torch.device('cpu')

    # grad placement: this can be set before running
    grad_device = torch.device('cuda:0')
    def grad_swap(grad):
        grad.data = grad.detach().to(grad_device)
        return grad
    weight.register_hook(grad_swap)
    bias.register_hook(grad_swap)

    ## Placement for a tensor swap in/out
    ##      where to swap in: op.device (op placement policy)
    ##      where to swap out: tensor.swap_to (policy)

    ## Timing when a tensor swapped in/out 
    ##      Basic Time block (each op is a slot?)
    ##      Event-driven (tesnor access? on-demand? | dynamic scenario?)

    # Policy description
    # op.device = torch.device('cuda:0')
    # ...
    
    #####

    class SwapLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):

            weight_id = id(weight)
            bias_id = id(bias)
            ctx.save_for_backward(input, torch.tensor(weight_id), torch.tensor(bias_id))
            tensor_map[weight_id] = weight
            tensor_map[bias_id] = bias

            # retrieve from cpu memory
            if weight.device != op_device:
                weight.data = weight.detach().to(op_device)
            if bias.get_device() != op_device:
                bias.data = bias.detach().to(op_device)

            # compute
            output = torch._C._nn.linear(input, weight, bias)

            # offload to CPU
            if weight.device != weight.host_device:
                weight.data = weight.detach().to(weight.host_device)
            if bias.device != bias.host_device:
                bias.data = bias.detach().to(bias.host_device)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight_id, bias_id = ctx.saved_tensors
            weight = tensor_map[weight_id.item()]
            bias = tensor_map[bias_id.item()]

            grad_input = grad_weight = grad_bas = None
            if ctx.needs_input_grad[0]:
                print('computing grad of input...')
                # retrieve weight
                if weight.device != op_device:
                    weight.data = weight.detach().to(op_device)
                grad_input = grad_output.matmul(weight)
                if weight.device != weight.host_device:
                    weight.data = weight.detach().to(weight.host_device)
            if ctx.needs_input_grad[1]:
                dim = grad_output.dim()
                if dim > 2:
                    grad_weight = grad\
                        .view(-1, grad_output.shape[-1])\
                        .t()\
                        .matmul(input.view(-1, input.shape[-1]))
                else:
                    grad_weight = grad_output.t().matmul(input)
            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0)

            ### Move gradient to it's tensor host device ###
            ### WARNING: there will be up to 2 redundant I/O if we require
            ### gradient to place differently with its tensor
            if grad_weight is not None:
                grad_weight.data = grad_weight.detach().to(weight.host_device)
            if grad_bias is not None:
                grad_bias.data = grad_bias.detach().to(bias.host_device)

            return grad_input, grad_weight, grad_bias
    
    output = SwapLinear.apply(input, weight, bias)
    return output


if __name__ == '__main__':

    torch.cuda.set_device(0)
    init_memory = torch.cuda.memory_allocated()

    # tensor definition
    batch_size = 32
    out_features = 10240
    in_features = 10240  ## 100 MB weight
    weight_1 = torch.rand((out_features, in_features)).requires_grad_()
    bias_1 = torch.rand(out_features).requires_grad_()
    input = torch.rand((batch_size, in_features)).cuda()
    weight_2 = torch.rand((out_features, in_features)).requires_grad_()
    bias_2 = torch.rand(out_features).requires_grad_()

    input_memory = (torch.cuda.memory_allocated() - init_memory) / 1024 / 1024
    
    # op compute
    print('======== Offloading Single Device =======')
    weight_swap_memory = (torch.cuda.memory_allocated() - init_memory) / 1024 / 1024

    output = swap_weight_grad_linear(input, weight_1, bias_1)
    output = swap_weight_grad_linear(output, weight_2, bias_2)
    loss = torch.mean(output) * 100
    print('loss: {}'.format(loss))
    loss.backward()
    
    finish_op_memory = (torch.cuda.memory_allocated() - init_memory) / 1024 / 1024
    max_allocated = (torch.cuda.max_memory_allocated() - init_memory) / 1024 / 1024
    
    # allocate tensor on gpu to see if swap workds
    tmp = torch.rand((out_features, in_features)).cuda()
    after_alloc_memory = (torch.cuda.memory_allocated() - init_memory) / 1024 / 1024

    print('Memory Consumption (MB):\n\t input-require: {:.2f}\n\t after swap weight: {:.2f}\n\t after op run {:.2f}\n\t max allocated: {:.2f}\n\t after allocate {:.2f}'.format(
        input_memory, weight_swap_memory, finish_op_memory, max_allocated, after_alloc_memory))

    # correctness verify
    print('weight grad: ', weight_1.grad.t())
    print('======== Offloading Single Device =======')
