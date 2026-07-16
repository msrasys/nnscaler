import torch

from .muon_optimizer import MuonMixin


try:
    from dion import Muon as _Muon
except ImportError as e:
    _DION_IMPORT_ERROR = e

    class _Muon(torch.optim.Optimizer):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Dion is not installed. Install Dion to use its Muon optimizer."
            ) from _DION_IMPORT_ERROR


class Muon(MuonMixin, _Muon):
    momentum_buffer_name = 'momentum'
    momentum_buffer_aliases = ('momentum_buffer',)
