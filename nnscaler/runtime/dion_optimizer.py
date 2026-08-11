try:
    from dion import Muon as _Muon
except ModuleNotFoundError as e:
    if e.name != 'dion':
        raise
    raise ModuleNotFoundError(
        'Dion is not installed. Install Dion to use its Muon optimizer.',
        name='dion',
    ) from e

from .muon_optimizer import MuonMixin


class Muon(MuonMixin, _Muon):
    momentum_buffer_name = 'momentum'
