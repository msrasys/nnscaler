from typing import List
import torch

_param_map = dict()


def get_swap_parameters() -> List[torch.nn.Parameter]:
    global _param_map
    return list(_param_map.values())


class _SwapEmbed(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight_id: int, fake: torch.nn.Parameter):
        # the fake parameter is preventing no grad fn
        ctx.save_for_backward(input, fake)
        ctx.weight_id = weight_id

        global _param_map
        weight = _param_map[weight_id]
        ctx.num_embeddings, ctx.embedding_dim = weight.size()
        ctx.weight_dtype = weight.dtype

        with torch.no_grad():
            # swap in
            weight.data = weight.detach().cuda()
            # compute
            output = torch.nn.functional.embedding(input, weight)
            # swap out
            weight.data = weight.detach().cpu()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        print(f'debug: >> {torch.distributed.get_rank()} embed backward here')
        (input, fake) = ctx.saved_tensors

        global _param_map
        weight = _param_map[ctx.weight_id]

        # swap in
        with torch.no_grad():
            weight = weight.data.cuda().requires_grad_()
        # compute
        with torch.enable_grad():
            output = torch.nn.functional.embedding(input, weight)
        torch.autograd.backward((output,), (grad_output,))
        # swap out
        assert weight.grad is not None
        with torch.no_grad():
            grad = weight.grad.data.cpu()
            weight = weight.data.cpu().requires_grad_()
            weight.grad = grad

        _param_map[ctx.weight_id] = weight
        fake_grad = torch.zeros_like(fake)
        return None, None, fake_grad


class SwapEmbed(torch.nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            assert padding_idx >= 0
            self.padding_idx = self.num_embeddings + padding_idx
        else:
            self.padding_idx = padding_idx

        _weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, requires_grad=True)
        )
        self.weight_id = id(_weight)
        # the fake parameter is preventing no grad fn
        self.fake = torch.nn.Parameter(torch.empty((1,), requires_grad=True))
        global _param_map
        _param_map[self.weight_id] = _weight

    def forward(self, input):
        return _SwapEmbed.apply(input, self.weight_id, self.fake)

    @property
    def weight(self):
        global _param_map
        return _param_map[self.weight_id]


if __name__ == '__main__':

    import cube
    from cube.profiler.memory import model_summary
    cube.init()

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            # self.model1 = torch.nn.Embedding(250000, 1024)
            self.model1 = SwapEmbed(250000, 1024)
            self.model2 = SwapEmbed(250000, 1024)
            # self.model2 = torch.nn.Embedding(250000, 1024)
            self.model3 = torch.nn.Embedding(250000, 1024)

        def forward(self, input_ids):
            out1 = self.model1(input_ids)
            # assert out1.grad_fn is not None
            out1 = out1 * 10
            # out2 = checkpoint.checkpoint(self.model2, input_ids)
            out2 = self.model2(input_ids)
            out2 = out2 / 10
            out3 = self.model3(input_ids)
            out3 = -out3
            return torch.sum(out1 + out2 + out3)

    model = Model().cuda()
    model.train()

    input_ids = torch.randint(
        0, 25000, (128, 1024),
        dtype=torch.int,
        device=torch.cuda.current_device(),
    )

    model_summary(model, (input_ids,))

    loss = model(input_ids)
    print(loss)
    loss.backward()

    print(model.model1.weight.grad)
