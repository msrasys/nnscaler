import warnings

from .muon_optimizer import MuonMixin


try:
    from dion import Muon as _Muon

    class Muon(MuonMixin, _Muon):
        momentum_buffer_name = 'momentum'

except ModuleNotFoundError as e:
    if e.name != 'dion':
        raise
    warnings.warn(
        'Dion is not installed. Dion Muon optimizers are unavailable.',
        stacklevel=2,
    )
