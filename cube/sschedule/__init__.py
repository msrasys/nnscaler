from cube.graph import parser
from cube.graph.ir_graph import IRGraph, IRAction
from cube.codegen.codegen import SScheduleCodeGen


class SpatialModule:

    def __init__(self, ir_graph):
        # the full semantic graph
        self._ir_graph = ir_graph
        # the spatial pytorch module for specific rank
        self._loaded_module = None

    def get_graph(self):
        return self._ir_graph

    def gen_module(self, rank, outfile, attach=False) -> str:
        """
        Set the module
        """
        # TODO: support multiple graph segments
        subnodes = [node for node in self._ir_graph.nodes() if node.on_device(rank)]
        # subgraph = self._ir_graph.subgraph(subnodes)
        action = IRAction(subnodes, self._ir_graph, devices=[rank])
        gener = SScheduleCodeGen(action)
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
    return module
