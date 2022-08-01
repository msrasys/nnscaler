from cube.graph import IRGraph
from cube.graph.function import IRConv2D, IRConv3D
from cube.graph.function import IRDimops, IRPad
from cube.ir.cten import IRTensor, IRCell
from cube.graph.function import IRSelect, IRSelectScatter, IRSlice, IRToTensor, IROnes, IRRand


def PAS(graph: IRGraph, resource):
    for node in graph.nodes():
        if isinstance(node, IRConv3D):
            sub_nodes = list()
            algo = node.algorithms('halo')
            Wnodes = graph.partition(node, algo, idx=0, dim=3, num=resource.ngpus // 2)
            for Wnode in Wnodes:
                algo = Wnode.algorithms('halo')
                Hnodes = graph.partition(Wnode, algo, idx=0, dim=2, num=2)
                sub_nodes += Hnodes
        else:
            sub_nodes = graph.replicate(node, times=resource.ngpus)
        # sub_nodes = graph.replicate(node, times=resource.ngpus)

        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
    print(graph.extra_repr())
    return graph

global opSigns

opSigns = []

def append_sign(sign: str):
    global opSigns
    if not sign in opSigns:
        opSigns.append(sign)
        
def PAS_ALL_TEST(graph: IRGraph, resource):
    for node in graph.nodes():
        sign = node.signature.split('.')[-1]
        append_sign(sign)
        if isinstance(node, IRConv3D):
            sub_nodes = list()
            algo = node.algorithms('halo')
            sub_nodes = graph.partition(node, algo, idx=0, dim=2, num=resource.ngpus)
        elif isinstance(node, IRDimops):
            sign = node.signature.split('.')[-1]
            if (sign == 'mul' or sign == 'add' or sign == 'sub' or sign == 'div') and (len(node.input(0).shape) == 5 or len(node.input(0).shape) == 3):
                algo = node.algorithms('dim')
                if len(node.input(0).shape) == 3:
                    sub_nodes = graph.partition(node, algo, idx=0, dim=1, num=resource.ngpus)
                    if sub_nodes == None:
                        sub_nodes = graph.replicate(node, times=resource.ngpus)
                elif len(node.input(0).shape) == 5:
                    sub_nodes = graph.partition(node, algo, idx=0, dim=3, num=resource.ngpus)
                    if sub_nodes == None:
                        sub_nodes = graph.replicate(node, times=resource.ngpus)
            elif sign == 'view':
                print('partition view')
                print(node)
                algo = node.algorithms('view_simp')
                sub_nodes = graph.partition(node, algo, idx=0, dimi=node.input(0).ndims-2, dimo=node.output(0).ndims-2, num=resource.ngpus)
                print(sub_nodes)
            else:
                sub_nodes = graph.replicate(node, times=resource.ngpus)
        elif isinstance(node, IRPad):
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(node, algo, dim=node.input(0).ndims-2, num=resource.ngpus)
        else:
            sub_nodes = graph.replicate(node, times=resource.ngpus)
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
    print(graph.extra_repr())
    return graph
  

def PAS_ALL_X(graph: IRGraph, resource):
    elewise_sign = ['mul', 'div', 'add', 'sub', 'multiref', 'neg', 'pow', 'cat', 'stack', 'sum', 'sin', 'gt']
    # elewise_sign = ['mul', 'div', 'add', 'sub']
    for node in graph.nodes():
        sign = node.signature.split('.')[-1]
        if isinstance(node, IRConv3D):
            sub_nodes = list()
            algo = node.algorithms('halo')
            sub_nodes = graph.partition(node, algo, idx=0, dim=3, num=resource.ngpus)
        elif isinstance(node, IRDimops):
            if sign in elewise_sign:
                ndims = node.input(0).ndims
                algo = node.algorithms('dim')
                append_sign(ndims)
                if ndims == 3 or ndims == 5 or ndims == 2 or ndims == 4:
                    sub_nodes = graph.partition(node, algo, idx=0, dim=ndims-1, num=resource.ngpus)
                    if sub_nodes == None:
                        sub_nodes = graph.replicate(node, times=resource.ngpus)
                else:
                    sub_nodes = graph.replicate(node, times=resource.ngpus)
            elif sign == 'view':
                algo = node.algorithms('view_simp')
                if node.input(0).ndims >= 2 and node.output(0).ndims >= 3:
                    sub_nodes = graph.partition(node, algo, idx=0, dimi=node.input(0).ndims-1, dimo=node.output(0).ndims-1, num=resource.ngpus)
                else:
                    sub_nodes = graph.replicate(node, times=resource.ngpus)
            else:
                sub_nodes = graph.replicate(node, times=resource.ngpus)
        # FIXME: Check 'circular' padding, should not be splitted easily
        elif isinstance(node, IRSelect) or isinstance(node, IRPad) or isinstance(node, IRSlice) or isinstance(node, IRToTensor):
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(node, algo, dim=node.input(0).ndims-1, num=resource.ngpus)
        elif isinstance(node, IRSelectScatter):
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(node, algo, diml=node.input(0).ndims-1, dimr=node.input(1).ndims-1, num=resource.ngpus)
        elif isinstance(node, IROnes) and node.output(0).ndims >= 3:
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(node, algo, dim=node.output(0).ndims-1, num=resource.ngpus)
        # elif isinstance(node, IRRand) and node.output(0).ndims >= 3:
        #     algo = node.algorithms('dim')
        #     sub_nodes = graph.partition(node, algo, dim=node.output(0).ndims-1, num=resource.ngpus)
        else:
            sub_nodes = graph.replicate(node, times=resource.ngpus)
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
    print(graph.extra_repr())
    print(opSigns)
    return graph

def PAS_ALL_Y(graph: IRGraph, resource):
    elewise_sign = ['mul', 'div', 'add', 'sub', 'multiref', 'neg', 'pow', 'cat', 'stack', 'sum', 'sin', 'gt']
    # elewise_sign = ['mul', 'div', 'add', 'sub']
    for node in graph.nodes():
        sign = node.signature.split('.')[-1]
        if isinstance(node, IRConv3D):
            sub_nodes = list()
            algo = node.algorithms('halo')
            sub_nodes = graph.partition(node, algo, idx=0, dim=2, num=resource.ngpus)
            assert sub_nodes != None
        elif isinstance(node, IRDimops):
            if sign in elewise_sign:
                ndims = node.input(0).ndims
                algo = node.algorithms('dim')
                append_sign(ndims)
                if ndims == 3 or ndims == 5 or ndims == 2 or ndims == 4:
                    sub_nodes = graph.partition(node, algo, idx=0, dim=ndims-2, num=resource.ngpus)
                    if sub_nodes == None:
                        sub_nodes = graph.replicate(node, times=resource.ngpus)
                else:
                    sub_nodes = graph.replicate(node, times=resource.ngpus)
            elif sign == 'view':
                algo = node.algorithms('view_simp')
                if node.input(0).ndims >= 2 and node.output(0).ndims >= 3:
                    print(node.input(0).shape, node.output(0).shape)
                    sub_nodes = graph.partition(node, algo, idx=0, dimi=node.input(0).ndims-2, dimo=node.output(0).ndims-2, num=resource.ngpus)
                    assert sub_nodes != None
                else:
                    sub_nodes = graph.replicate(node, times=resource.ngpus)
            else:
                sub_nodes = graph.replicate(node, times=resource.ngpus)
        elif isinstance(node, IRSelect) or isinstance(node, IRPad) or isinstance(node, IRSlice) or isinstance(node, IRToTensor):
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(node, algo, dim=node.input(0).ndims-2, num=resource.ngpus)
            assert sub_nodes != None
        elif isinstance(node, IRSelectScatter):
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(node, algo, diml=node.input(0).ndims-2, dimr=node.input(1).ndims-2, num=resource.ngpus)
            assert sub_nodes != None
        elif isinstance(node, IROnes) and node.output(0).ndims >= 3:
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(node, algo, dim=node.output(0).ndims-2, num=resource.ngpus)
            assert sub_nodes != None
        # elif isinstance(node, IRRand) and node.output(0).ndims >= 3:
        #     algo = node.algorithms('dim')
        #     sub_nodes = graph.partition(node, algo, dim=node.output(0).ndims-1, num=resource.ngpus)
        else:
            sub_nodes = graph.replicate(node, times=resource.ngpus)
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
    print(graph.extra_repr())
    print(opSigns)
    return graph
