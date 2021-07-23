import torch


class PhysicTensor(torch.Tensor):
    """
    Additional attributes on top of PyTorch Tensor:

    data_host_device:
        Tensor data placement. The device is responsible
        for managing the tensor data.
    
    grad_host_device:
        Gradient data placement. The device is responsible
        for managing the gradient data. If no grad required
        for this Tensor, this option won't have impact
    """
    @property
    def data_host_device(self):
        if not hasattr(self, '_data_host_device'):
            self._data_host_device = self.device
        return self._data_host_device

    @data_host_device.setter
    def data_host_device(self, device):
        if not isinstance(device, torch.device):
            raise TypeError('Expected torch.device')
        self._data_host_device = device
        # inplacement movement to host device
        if self.device != self.data_host_device:
            self.move_(self.data_host_device)

    @property
    def grad_host_device(self):
        if not hasattr(self, '_grad_host_hook'):
            self._grad_host_hook = None
        if not hasattr(self, '_grad_host_device'):
            self._grad_host_device = self._data_host_device
        return self._grad_host_device


    @grad_host_device.setter
    def grad_host_device(self, device):
        if not isinstance(device, torch.device):
            raise TypeError('Expected torch.device')
        self._grad_host_device = device
        # inplacement movement to host device
        if self.grad is not None:
            self.grad.data = self.grad.detach().to(self.grad_host_device)
        # modify hooks
        if self._grad_host_hook is not None:
            self._grad_host_hook.remove()
        def move_grad(grad):
            grad.data = grad.detach().to(self.grad_host_device)
            return grad
        self._grad_host_hook = self.register_hook(move_grad)


    def move_(self, device):
        """
        inplacement device movement 
        """
        if not isinstance(device, torch.device):
            raise TypeError('Expected torch.device')
        self.data = self.detach().to(device)

    def move_grad_(self, device):
        """
        inplacement device move on tensor grad
        """
        if not isinstance(device, torch.device):
            raise TypeError('Expected torch.device')
        if self.grad is not None:
            self.grad.data = self.grad.detach().to(device)