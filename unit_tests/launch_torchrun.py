import uuid

from torch.distributed.run import elastic_launch, LaunchConfig


def launch_torchrun(nproc_per_node, worker_fn, *args, **kwargs):
    launch_config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=nproc_per_node,
        rdzv_backend = "c10d",
        rdzv_endpoint = "localhost:29400",
        run_id = str(uuid.uuid4()),
        monitor_interval=0.1,
        max_restarts=0,
    )
    outputs = elastic_launch(launch_config, worker_fn)(*args, **kwargs)
    return outputs


def clone_to_cpu(tensor):
    # when you use launch_torchrun
    # you can't directly return a cuda tensor
    #   Error message: Producer process has been terminated before all shared CUDA tensors released. See Note [Sharing CUDA tensors]
    # So you can use this function to clone a tensor to cpu
    cloned_tensor = tensor.cpu().clone().detach().requires_grad_(tensor.requires_grad)
    if tensor.grad is not None:
        cloned_tensor.grad = tensor.grad.cpu().clone()
    return cloned_tensor
