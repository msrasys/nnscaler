from .muon_optimizer import MuonMixin


try:
    from dion import Muon as _Muon

    class Muon(MuonMixin, _Muon):
        momentum_buffer_name = 'momentum'

except ImportError:
    pass
