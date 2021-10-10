from cube.graph import parser
from cube.codegen.codegen import SScheduleCodeGen
from cube.sschedule.adapter import Adapter


class SpatialModule:

    def __init__(self, ir_graph):
        # the full semantic graph
        self._ir_graph = ir_graph
        # the spatial pytorch module for specific rank
        self._loaded_module = None

    def get_graph(self):
        return self._ir_graph

    def gen_module(self, seq, rank, outfile, attach=False) -> str:
        """
        Set the module
        """
        gener = SScheduleCodeGen(seq)
        code = gener.gen(device=rank, outfile=outfile, attach=attach)
        return code

    def load_module(self, filename: str):
        print(f'> loading generated spatial moduel from {filename}')
        import importlib.util
        spec = importlib.util.spec_from_file_location("GenModel", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._loaded_module = module.GenModel().cuda()

    def get_gen_module(self):
        return self._loaded_module

    def clear_module(self):
        self._loaded_module = None


def schedule(module, input_shapes, policy_fn=None):
    """
    Spatial schedule

    Returns:
        IRGraph
    """
    ir_graph = parser.convert(module, input_shapes=input_shapes)
    module = SpatialModule(ir_graph)
    if policy_fn:
        module._ir_graph = policy_fn(module.get_graph())
    module._ir_graph = Adapter.adapt(module._ir_graph)
    return module
