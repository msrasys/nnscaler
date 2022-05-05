from enum import Enum
import sys

# class NodeType(Enum):
#     UNKNOWN = 0
#     DATALOADER = 1
#     FORWARD = 2
#     BACKWARD_A = 3
#     BACKWARD_W = 4
#     OPTIMIZER = 5

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


nodeList = []
global_node_id = -1


def new_node_id():
    global global_node_id
    global_node_id += 1
    return global_node_id


def last_node(last_step=1):
    assert len(nodeList) >= last_step
    return nodeList[-last_step]




class AlgorithmMgr:
    batch_split: str
    replica: str

    def __init__(self):
        self.batch_split = 'batch_split'
        self.replica = 'replica'
        self.split = 'split'
        self.tensor_split = 'tensor_split'

class Operator:
    algo: AlgorithmMgr
    def __init__(self):
        self.algo = AlgorithmMgr()

class Node:
    id: int
    inputs: []
    outputs: []
    removed: bool
    op: Operator

    def __init__(self):
        super().__init__()
        self.removed = False
        self.id = new_node_id()
        self.inputs = []
        self.outputs = []
        self.op = Operator()
        nodeList.append(self)

    def spawn(self, portion: str=None):
        node = self.__class__() #create same type
        node.inputs = [t.spawn() for t in self.inputs]
        node.outputs = [t.spawn() for t in self.outputs]
        return node

    def __str__(self):
        return "Node({}), {}\tinput:{}\toutput:{} ".format(
            self.id, str(type(self)).lstrip('<class \'__main__.').rstrip('\'>'),
            '\t'.join([str(x) for x in self.inputs] if len(self.inputs) > 0 else ""),
            '\t'.join([str(x) for x in self.outputs] if len(self.outputs) > 0 else ""))


    def slim(self):
        return "Node({}), {}".format(
            self.id, str(type(self)).lstrip('<class \'__main__.').rstrip('\'>'))

class NodeData(Node):
    def __init__(self):
        super(NodeData, self).__init__()
        # self.type = NodeType.DATALOADER


class NodeFwd(Node):
    def __init__(self):
        super(NodeFwd, self).__init__()
        # self.type = NodeType.FORWARD


class NodeBwd(Node):
    def __init__(self):
        super(NodeBwd, self).__init__()


class NodeBwdA(NodeBwd):
    def __init__(self):
        super(NodeBwd, self).__init__()
        # self.type = NodeType.BACKWARD_A


class NodeBwdW(NodeBwd):
    def __init__(self):
        super(NodeBwd, self).__init__()
        # self.type = NodeType.BACKWARD_W


class NodeOpt(Node):
    def __init__(self):
        super(NodeOpt, self).__init__()
        # self.type = NodeType.OPTIMIZER


# for logic tensor
class TensorType(Enum):
    UNKNOWN = 0
    WEIGHT = 1
    WEIGHT_UPDATED = 2
    ACTIVATION = 3
    GRADIENT_A = 4
    GRADIENT_W = 5
    OPTIMIZER_STATE = 6
    LOSS = 7


logicTensorList = []
global_logic_tensor_id = -1


def new_logic_tensor_id():
    global global_logic_tensor_id
    global_logic_tensor_id += 1
    return global_logic_tensor_id


def last_logic_tensor(last_step=1):
    assert len(logicTensorList) >= last_step
    return logicTensorList[-last_step]


class LogicTensor:
    id: int
    type: TensorType

    def __init__(self, tensor_type=TensorType.UNKNOWN):
        super().__init__()
        self.id = new_logic_tensor_id()
        self.type = tensor_type


tensorList = []
global_tensor_id = -1


def new_tensor_id():
    global global_tensor_id
    global_tensor_id += 1
    return global_tensor_id


def last_tensor(last_step=1):
    assert len(tensorList) >= last_step
    return tensorList[-last_step]


class Tensor:
    id: int
    logic: LogicTensor
    portion: str

    def new(self):
        pass

    def __init__(self, tensor_type=TensorType.UNKNOWN, exist_tensor=None, portion=None):
        super().__init__()
        self.id = new_tensor_id()
        if exist_tensor is None:
            self.logic = LogicTensor(tensor_type)
            self.portion = 'full'
        else:
            self.logic = exist_tensor.logic
            self.portion = exist_tensor.portion
            if portion is not None:
                self.portion += '>' + portion
        tensorList.append(self)

    def __getattr__(self, attr):
        if(attr == 'type'):
            return self.logic.type
        else:
            return self.attr

    def __str__(self):
        return ("Tensor({}), {} of ({} {})".format(
            self.id,
            self.portion,
            self.logic.id,
            str(self.type).lstrip('TensorType.')))

    def spawn(self, portion:str=None):
        return Tensor(exist_tensor=self, portion=portion)

class Graph:
    nodes: []

    def find_input(self, node: Node, tensor_type: TensorType):
        ret = list(filter(lambda x: x.type == tensor_type, node.inputs))
        assert len(ret) > 0
        return ret[0]

    def create_sample_graph(self):
        op_num = 2

        for idx in range(1):  # sample data loader
            node = NodeData()  # Node(NodeType.DATALOADER)
            node.outputs.append(Tensor(TensorType.ACTIVATION))
            self.nodes.append(node)

        for idx in range(op_num):  # forward ops
            node = NodeFwd()  # Node(NodeType.FORWARD)
            node.inputs.append(last_tensor())
            node.inputs.append(Tensor(TensorType.WEIGHT))
            node.outputs.append(Tensor(TensorType.ACTIVATION))
            self.nodes.append(node)

        for idx in range(1):  # label data loader
            node = NodeData()  # Node(NodeType.DATALOADER)
            node.outputs.append(Tensor(TensorType.ACTIVATION))
            self.nodes.append(node)

        for idx in range(1):  # loss
            node = NodeFwd()  # Node(NodeType.FORWARD)
            node.inputs.append(last_tensor())
            node.outputs.append(Tensor(TensorType.LOSS))
            self.nodes.append(node)

        for fwd_node in list(filter(lambda x: type(x) is NodeFwd, self.nodes))[::-1]:  # backward ops
            out_gradient = last_tensor()
            if len(fwd_node.inputs) == 2:
                # computing weight's gradient
                node = NodeBwdW()  # Node(NodeType.BACKWARD_W)
                node.inputs.append(out_gradient)  # out_g_act
                node.inputs.append(self.find_input(fwd_node, TensorType.WEIGHT))
                node.outputs.append(Tensor(TensorType.GRADIENT_W))
                self.nodes.append(node)
            if len(fwd_node.inputs) >= 1:
                # computing activation's gradient
                node = NodeBwdA()  # Node(NodeType.BACKWARD_A)
                node.inputs.append(out_gradient)
                node.inputs.append(self.find_input(fwd_node, TensorType.ACTIVATION))
                node.outputs.append(Tensor(TensorType.GRADIENT_A))
                self.nodes.append(node)
            else:
                assert False

        for bwd_w_node in list(filter(lambda x: type(x) is NodeBwdW, self.nodes)):  # optimizer
            node = NodeOpt()  # Node(NodeType.OPTIMIZER)
            node.inputs.append(self.find_input(bwd_w_node, TensorType.WEIGHT))  # WEIGHT
            node.inputs.append(bwd_w_node.outputs[0])
            node.inputs.append(Tensor(TensorType.OPTIMIZER_STATE))
            node.outputs.append(Tensor(TensorType.WEIGHT_UPDATED))
            self.nodes.append(node)

    def __init__(self, create_sample=False):
        super().__init__()
        self.nodes = []

        if create_sample:
            self.create_sample_graph()

    def __str__(self):
        # for node in self.nodes:
        return '\n'.join([str(x) if not x.removed else "DEL "+str(x) for x in self.nodes])


graph = Graph(create_sample=True)
print('graph = \n{}'.format(graph))
global_new_graph = Graph()

# print('nodeList[{}] = \n{}'.format(len(nodeList), nodeList))
# print('tensorList[{}] = \n{}'.format(len(tensorList), tensorList))


class Config:
    num: int


class Device:
    pass


class Parallelizer:
    def run(self, g: Graph, config: Config) -> Graph:
        return None


def trans(node: Node, algo, num: int) -> [Node]:
    node.removed = True
    nodes = [node.spawn() for i in range(num)]
    if algo == 'replica':
        global_new_graph.nodes.extend(nodes)
        return nodes
    elif algo == 'batch_split':
        for idx, nd in enumerate(nodes):
            for ts in nd.inputs + nd.outputs:
                ts.portion += '>batch-{}/{}'.format(idx, num)
        global_new_graph.nodes.extend(nodes)
        return nodes
    elif algo == 'split': #elementwise split
        for idx, nd in enumerate(nodes):
            for ts in nd.inputs + nd.outputs:
                ts.portion += '>flat-{}/{}'.format(idx, num)
        global_new_graph.nodes.extend(nodes)
        return nodes
    elif algo == 'tensor_split':
        for idx, nd in enumerate(nodes):
            for ts in nd.inputs + nd.outputs:
                ts.portion += '>tensor-{}/{}'.format(idx, num)
        global_new_graph.nodes.extend(nodes)
        return nodes
    else:
        assert False


def sched_s(node: Node, dev: Device) -> None:
    print("{}sched_s...{} @ {}{}".format(bcolors.OKGREEN, node.slim(), dev, bcolors.ENDC))
    pass


def sched_t_pair(node_before: Node, node_after: Node) -> bool:
    print("{}sched_t...{}-> {}{}".format(bcolors.OKBLUE, node_before.slim(), node_after.slim(), bcolors.ENDC))
    #TODO legal check
    return True


def sched_t(nodes: [Node]) -> bool:
    for i in range (len(nodes) - 1):
        if not sched_t_pair(nodes[i], nodes[i+1]):
            return False
    return True


def set_affinity(producer_node, consumer_node):
    print("{}affinity...{}-> {}{}".format(bcolors.OKCYAN, producer_node.slim(), consumer_node.slim(), bcolors.ENDC))
    pass


from collections import namedtuple
def idxzip(list: []):
    Entry = namedtuple('Entry', ['idx', 'item'])
    # return [{'idx': i, 'item': x} for i, x in enumerate(list)]
    return [Entry(i, x) for i, x in enumerate(list)]


def xmap(func, iterables):
    if list(map(func, iterables)) is None:
        print('xmap ERROR')


### TODO how about Tx in flexflow GSPMD etc.?

# traditional data-parallel process:
# tx start
# 1. replicated graph g -> g' * N
# 2. change batch-size of g'
# 3. insert gradient allreduce (manually, can auto-gen in our sys)
# tx end


class DataParallelParallelizer(Parallelizer):
    def run(self, g: Graph, config: Config) -> Graph:
        global global_new_graph
        global_new_graph.nodes.clear()

        # ----------------
        for node in g.nodes:
            if isinstance(node, (NodeData, NodeFwd, NodeBwd)):
                nodes = trans(node, node.op.algo.batch_split, config.num)  # by batch-dim-split
                xmap(lambda x: sched_s(node=x.item, dev=x.idx), idxzip(nodes))
            elif isinstance(node, (NodeOpt)):
                nodes = trans(node, node.op.algo.replica, config.num) #replicated optimizers
                xmap(lambda x: sched_s(node=x.item, dev=x.idx), idxzip(nodes))
            else:
                print(node)
                print(type(node))
                assert False
        # ----------------

        global_new_graph.nodes[:0] = [nd for nd in graph.nodes if not nd.removed]
        return global_new_graph


class DataParallelZeROParallelizer(Parallelizer):
    def run(self, g: Graph, config: Config) -> Graph:
        global global_new_graph
        global_new_graph.nodes.clear()

        # ----------------
        for node in g.nodes:
            if isinstance(node, (NodeData, NodeFwd, NodeBwd)):
                nodes = trans(node, node.op.algo.batch_split, config.num)  # by batch-dim-split
                map(lambda x: sched_s(node=x.item, dev=x.idx), idxzip(nodes))
            elif isinstance(node, (NodeOpt)):
                nodes = trans(node, node.op.algo.split, config.num) #split optimizers
                map(lambda x: sched_s(node=x.item, dev=x.idx), idxzip(nodes))
            else:
                assert False
        # ----------------

        global_new_graph.nodes[:0] = [nd for nd in graph.nodes if not nd.removed]
        return global_new_graph


class GradientAccumulationParallelizer(Parallelizer):
    def run(self, g: Graph, config: Config) -> Graph:
        global global_new_graph
        global_new_graph.nodes.clear()

        # ----------------
        for node in g.nodes:
            if isinstance(node, (NodeData, NodeFwd, NodeBwd)):
                nodes = trans(node, node.op.algo.batch_split, config.num)  # by batch-dim-slit
                sched_t(nodes)  # sequential order
            elif isinstance(node, (NodeOpt)):
                pass
            else:
                assert False
        # ----------------

        global_new_graph.nodes[:0] = [nd for nd in graph.nodes if not nd.removed]
        return global_new_graph


def node_to_stage(g: Graph, config: Config) -> {}:  # return node->stage mapping
    ret = {}
    nodes = g.nodes  # TODO topo forward traversal
    fwd_node = list(filter(lambda x: type(x) is NodeFwd, nodes))

    per_stage_size = len(nodes) // config.stages
    for node in nodes:
        # TODO replace dummy assignment
        ret[node] = 0

    return ret


class GPipeParallelizer(Parallelizer):
    def run(self, g: Graph, config: Config) -> Graph:
        global global_new_graph
        global_new_graph.nodes.clear()

        # ----------------
        n2stage = node_to_stage(g, config)
        for node in g.nodes:
            device = n2stage[node]
            if isinstance(node, (NodeData, NodeFwd, NodeBwd)):
                nodes = trans(node, node.op.algo.batch_split, config.num)  # by batch-dim-slit
                sched_t(nodes)  # sequential order
                xmap(lambda x: sched_s(node=x, dev=device), nodes)  # assign same stage device
            elif isinstance(node, (NodeOpt)):
                sched_s(node, device)
            else:
                assert False
        # ----------------

        global_new_graph.nodes[:0] = [nd for nd in graph.nodes if not nd.removed]
        return global_new_graph


class TensorParallelParallelizer(Parallelizer):
    def run(self, g: Graph, config: Config) -> Graph:
        global global_new_graph
        global_new_graph.nodes.clear()

        # ----------------
        for node in g.nodes:
            if isinstance(node, (NodeFwd, NodeBwd, NodeOpt)):
                nodes = trans(node, node.op.algo.tensor_split, config.num)  # by tensor-dim-slit
                xmap(lambda x: sched_s(node=x.item, dev=x.idx), idxzip(nodes))
            elif isinstance(node, (NodeData)):
                nodes = trans(node, node.op.algo.replica, config.num)
                xmap(lambda x: sched_s(node=x.item, dev=x.idx), idxzip(nodes))
            else:
                assert False
        # ----------------

        global_new_graph.nodes[:0] = [nd for nd in graph.nodes if not nd.removed]
        return global_new_graph


def find_consumers(graph: Graph, tensor: Tensor):
    ret = []
    for node in graph.nodes:
        if any([input_tensor.logic == tensor.logic for input_tensor in node.inputs]):
            ret.append(node)
    return ret

def find_producers(graph: Graph, tensor: Tensor):
    ret = []
    for node in graph.nodes:
        if any([output_tensor.logic == tensor.logic for output_tensor in node.outputs]):
            ret.append(node)
    return ret


class Recompute(Parallelizer):
    def run(self, g: Graph, config: Config) -> Graph:
        global global_new_graph
        global_new_graph.nodes.clear()

        # ----------------
        for node in g.nodes:
            if isinstance(node, (NodeFwd)):
                origin_fwd, recompute_fwd = trans(node, node.op.algo.replica, 2)
                consumers = find_consumers(g, origin_fwd.outputs[0])
                for consumer in consumers:
                    if isinstance(consumer, NodeFwd):
                        set_affinity(origin_fwd, consumer)  # break dependencies op0.fwd -> op1.fwd; op0.fwd' -> op0.bwd
                    else:
                        set_affinity(recompute_fwd, consumer)  # break dependencies op0.fwd -> op1.fwd; op0.fwd' -> op0.bwd
                        producers = list(filter(lambda x: isinstance(x, NodeBwd), find_producers(g, consumer.inputs[0])))
                        for producer in producers:
                            sched_t_pair(producer, recompute_fwd)
        # ----------------

        global_new_graph.nodes[:0] = [nd for nd in graph.nodes if not nd.removed]
        return global_new_graph

class ActivationSwap(Parallelizer):
    def run(self, g: Graph, config: Config) -> Graph:
        pass

# para = DataParallelParallelizer()
# para = DataParallelZeROParallelizer()
# para = GradientAccumulationParallelizer()
# para = GPipeParallelizer()
# para = TensorParallelParallelizer()
para = Recompute()


config = Config()
config.num = 2
config.stages = 2
global_new_graph = para.run(graph, config)
print('new_graph = \n{}'.format(global_new_graph))
