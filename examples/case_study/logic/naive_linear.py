import torch
from torch.nn.parameter import Parameter
import math

torch.manual_seed(121)


def linear(input, weight, bias=None):
    output = torch._C._nn.linear(input, weight, bias)
    return output


def apply_adam(params, grads, exp_avgs, exp_avg_sqs, steps, beta1, beta2, lr):
    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = steps[-1]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)
        step_size = lr / bias_correction1
        param.addcdiv_(exp_avg, denom, value=-step_size)


if __name__ == '__main__':

    torch.cuda.set_device(0)

    # tensor definition
    batch_size = 128
    out_features = 1024
    in_features = 1024

    weight = torch.rand((out_features, in_features)).cuda().requires_grad_()
    bias = torch.rand(out_features).cuda().requires_grad_()
    input = torch.rand((batch_size, in_features)).cuda()
    # print('weight: ', weight)
    # print('bias: ', bias)
    # print('input: ', input)

    ## Adam optimizer states -- 2x more weights volume
    weight_exp_avg = torch.zeros_like(
        weight, memory_format=torch.preserve_format
    )
    weight_exp_avg_sq = torch.zeros_like(
        weight, memory_format=torch.preserve_format
    )
    bias_exp_avg = torch.zeros_like(
        bias, memory_format=torch.preserve_format
    )
    bias_exp_avg_sq = torch.zeros_like(
        bias, memory_format=torch.preserve_format
    )
    state_steps = list()
    lr = 0.01
    beta1 = 0.5
    beta2 = 0.5
    
    # iterations
    for _ in range(4):
        # ======= step1: forward ======= #
        output = linear(input, weight, bias)
        loss = torch.mean(output)
        print(loss)

        # ======= step2: backward ======= #
        loss.backward()
        # print('weight grad: ', weight.grad.t())

        # ======= step3: update ======= #
        params = [weight, bias]
        grads = [weight.grad, bias.grad]
        exp_avgs = [weight_exp_avg, bias_exp_avg]
        exp_avg_sqs = [weight_exp_avg_sq, bias_exp_avg_sq]
        state_steps.append(len(state_steps)+1)
        with torch.no_grad():
            apply_adam(
                params, grads, exp_avgs, exp_avg_sqs, state_steps,
                beta1, beta2, lr
            )
        # zero out grad
        weight.grad = None
        bias.grad = None
