from .muon_optimizer import MuonMixin


try:
    from dion import Muon as _Muon

    class Muon(MuonMixin, _Muon):
        is_dion = True

except ImportError:
    pass
