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


def swap_weight_grad_linear_v2(input, weight, bias):

    class SwapLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias, swap_weight=True, swap_bias=True):

            weight_id = id(weight)
            bias_id = id(bias)
            ctx.save_for_backward(input, torch.tensor(weight_id), torch.tensor(bias_id))
            tensor_map[weight_id] = weight
            tensor_map[bias_id] = bias_id

            ctx.constants = (swap_weight, swap_bias)

            # retrieve from cpu memory
            if swap_weight:
                weight.data = weight.detach().cuda()
            if swap_bias:
                bias.data = bias.detach().cuda()

            # compute
            output = torch._C._nn.linear(input, weight, bias)

            # offload to CPU
            if swap_weight:
                weight.data = weight.detach().cpu()
            if swap_bias:
                bias.data = bias.detach().cpu()
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight_id, bias_id = ctx.saved_tensors
            weight = tensor_map[weight_id.item()]
            bias = tensor_map[bias_id.item()]
            swap_weight, swap_bias = ctx.constants

            grad_input = grad_weight = grad_bas = None
            if ctx.needs_input_grad[0]:
                print('computing grad of input...')
                # retrieve weight
                if swap_weight:
                    weight.data = weight.cuda()
                grad_input = grad_output.matmul(weight)
                if swap_weight:
                    weight.data = weight.detach().cpu()
            if ctx.needs_input_grad[1]:
                dim = grad_output.dim()
                if dim > 2:
                    grad_weight = grad\
                        .view(-1, grad_output.shape[-1])\
                        .t()\
                        .matmul(input.view(-1, input.shape[-1]))
                else:
                    grad_weight = grad_output.t().matmul(input)
                if swap_weight:
                    grad_weight.data = grad_weight.detach().cpu()
            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0)
                if swap_bias:
                    grad_bias.data = grad_bias.detach().cpu()
            print('here')
            return grad_input, grad_weight, grad_bias, None, None
    
    output = SwapLinear.apply(input, weight, bias,
                              True, True)
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
    weight_2 = torch.rand((out_features, in_features)).requires_grad_()
    bias_2 = torch.rand(out_features).requires_grad_()
    input = torch.rand((batch_size, in_features)).cuda()

    input_memory = (torch.cuda.memory_allocated() - init_memory) / 1024 / 1024
    
    # op compute
    print('======== Checkpointing Single Device =======')

    weight_swap_memory = (torch.cuda.memory_allocated() - init_memory) / 1024 / 1024

    output = swap_weight_grad_linear_v2(input, weight_1, bias_1)
    print('output: {}'.format(output))
    output = swap_weight_grad_linear_v2(output, weight_2, bias_2)
    loss = torch.mean(output) * 100
    loss.backward()
    
    finish_op_memory = (torch.cuda.memory_allocated() - init_memory) / 1024 / 1024
    
    # allocate tensor on gpu to see if swap workds
    tmp = torch.rand((out_features, in_features)).cuda()
    after_alloc_memory = (torch.cuda.memory_allocated() - init_memory) / 1024 / 1024

    max_allocated = (torch.cuda.max_memory_allocated() - init_memory) / 1024 / 1024
    print('memory consumption (MB): max allocated: {:.2f} | input-require: {:.2f} | after swap weight: {:.2f} | after op run {:.2f} | after allocate {:.2f}'.format(
        max_allocated, input_memory, weight_swap_memory, finish_op_memory, after_alloc_memory))

    # correctness verify
    print('weight grad: ', weight_2.grad.t())
    print('======== Checkpointing Single Device =======')
