from graph_manipulation import *

'''
[dataflow graph]: the logic of one training iteration
input: samples, weight tensors, optimizer state tensors
output: (updated) weight tensors, (updated) optimizer state tensors

(assumption) a DL model description compatible with different batch-size
    ref 1 ONNX: https://github.com/onnx/onnx/issues/2182
    ref 2 ...
 
'''

'''
graph manipulation as data-parallel 
option 1: deep copy graph and manually decide adjustment for each node/tensor
option 2: manually decide manipulation for each node/tensor
option 3: using node-role, e.g., DataNode/Fwd/Bwd split; Optimizer replicate; weight gradient all-reduce before apply
option 4: using tensor info. e.g., tensors with  
'''
def data_parallel_raw(g: Graph, dev_num: int, option: int):
    print(g)

    if option == 1: #per node manipulation following oracle
        def magic_func(node) -> bool:
            pass

        # 1. multiply operators
        for node in g.nodes:
            if magic_func(node) == 'split op':
                new_nodes = []
                for i in range(dev_num):
                    new_node_inputs = []
                    for ts in node.inputs:
                        if magic_func(ts) == 'split tensor':
                            new_node_inputs.append(split(ts, dev_num))
                        elif magic_func(ts) == 'replicate tensor':
                            new_node_inputs.append(ts)
                    #TODO connect split tensor
                    new_nodes.append(Node(type=node.type, inputs=new_node_inputs)) #insert new node
            elif magic_func(node) == 'replicate op':
                new_nodes = [node] * dev_num

            g.replace(node, new_nodes)  # TODO connect

        #2. inserting gradient averaging
        for node in g.nodes:
            new_allreduce_node = None
            gradient = None
            for ts in node.inputs:
                if magic_func(ts):
                    new_allreduce_node = Node(type=allreduce, inputs=ts)
                    gradient = ts
                    break
            new_node = Node(node.type, node.inputs - gradient + new_allreduce_node.output)
            g.replace(node, [new_allreduce_node, new_node])

    elif option == 2: #replicate entire graph and adjust
        def magic_func(node) -> bool:
            pass

        # deep copy graph
        graphs = [g.deepcopy() for i in range(dev_num)]
        # reset batch size for each new graph, leveraging resizable DFG description (<-assumption)
        for graph in graphs:
            graph.batch_size = g.batch_size // dev_num
        # inserting gradient averaging
        for graph in graphs:
            for node in graph.nodes:
                new_allreduce_node = None
                gradient = None
                for ts in node.inputs:
                    if magic_func(ts):
                        new_allreduce_node = Node(type=allreduce, inputs=ts)
                        gradient = ts
                        break
                new_node = Node(type=node.type, inputs=node.inputs - gradient + new_allreduce_node.output)
                graph.replace(node, [new_allreduce_node, new_node])
    elif option == 3: #node role based manipulation
        pass

    elif option == 4: #tensor dimention info based manipulation
        pass





data_parallel_raw(graph)