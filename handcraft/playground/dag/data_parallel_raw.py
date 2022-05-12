from graph_manipulation import *

'''
[dataflow graph]: the logic of one training iteration
DFG input: samples, weight tensors, optimizer state tensors
DFG output: (updated) weight tensors, (updated) optimizer state tensors

data tensors as edges, produced by one operator and consumed by one or more operator(s) 
operators as nodes, consumes one or more tensor(s), (mostly) producing one tensor 

(assumption resizable batch) same DL model description for different batch-sizes (batch-size as variable)
    ref 1 ONNX: https://github.com/onnx/onnx/issues/2182
    ref 2 ...
 
///////////////

graph manipulation as data-parallel, 4 method options
option 1: manually decide manipulation for each node/tensor following an oracle that knows everything
option 2: deep copy graph and manually decide adjustment for each node/tensor
option 3: using node-role, e.g., DataNode/Fwd/Bwd split; Optimizer replicate; weight's gradient all-reduce before used by Optimizers
option 4: using tensor info. e.g., tensors with batch-dim will split, operators and other tensors adapt accordingly 
'''
def data_parallel_raw(g: Graph, device_num: int, method: int):
    def oracle_func(*args) -> bool:
        pass

    if method == 'raw graph manipulation': #per node manipulation following oracle's instruction
        # 1. multiply operators for ``parallel'' in data-parllelism
        for node in g.nodes:
            new_nodes = []
            for device_id in range(device_num):
                new_node_inputs = []
                for ts in node.inputs:
                    # find corresponding input tensor, which is another new operator's (sliced/replicated...) output
                    new_input = oracle_func(node, ts, device_id, device_num).query("find_new_input")
                    new_node_inputs.append(new_input)

                new_node_outputs = []
                for ts in node.outputs:
                    # new out tensor of the same shape (if replicate) or 1/N (if slice on certain dim)
                    new_output_shape = oracle_func(node, ts, device_id, device_num).query("new_output_shape")
                    new_output = Tensor(new_output_shape) # create new tensor as output (will be another operator(s)'s input)
                    new_node_outputs.append(new_output)

                new_node_type = oracle_func(node).query("new_node_type")
                # create new node, with device info
                new_node = Node(type=new_node_type, inputs=new_node_inputs, outputs=new_node_outputs,
                                device=device_id)
                new_nodes.append(new_node)

            g.replace(node, new_nodes) #replacing with new nodes

        # 2. inserting gradient averaging
        for node in g.nodes:
            new_allreduce_node = None
            input_to_replace = None
            for ts in node.inputs:
                if oracle_func(ts).query('insert allreduce here'):
                    new_allreduce_node = Node(type='allreduce', inputs=ts)
                    input_to_replace = ts
                    break

            new_node = Node(type=node.type, inputs=node.inputs - input_to_replace + new_allreduce_node.output,
                            outputs=node.outputs)
            g.replace(node, [new_allreduce_node, new_node])

    elif method == 'replicate graph and adjust': #replicate entire graph and adjust, similar to approaches of Horovod and PyTorch DDP
        # 1. deep copy graph
        graphs = [g.deepcopy() for i in range(device_num)]

        # 2. reset batch size for each new graph, leveraging model description resizable batch (<-assumption)
        #    input or output shape inferred from shape_inference, representing split (1/N shape) or replicated (unchanged shape)
        for index, graph in enumerate(graphs):
            graph.arguments.batch_size = g.arguments.batch_size // device_num
            graph.to_device(device=index)

        # 3. inserting gradient averaging
        for graph in graphs:
            for node in graph.nodes:
                new_allreduce_node = None
                input_to_replace = None
                for ts in node.inputs:
                    if oracle_func(ts).query('insert allreduce here'):
                        new_allreduce_node = Node(type=allreduce, inputs=ts, outputs=Tensor(node.outputs.shape))
                        input_to_replace = ts
                        break

                new_node = Node(type=node.type, inputs=node.inputs - input_to_replace + new_allreduce_node.output,
                                outputs=node.outputs)
                graph.replace(node, [new_allreduce_node, new_node])

    elif method == 3: #node role based manipulation
        for node in g.nodes:
            if isinstance(node, (NodeData)):
                new_nodes = [
                    Node(type=node.type, inputs=None,
                         config=node.config.reset_batch_size(node.config.batch_size // device_num),
                         outputs=node.outputs.shape[0] // device_num + node.outputs.shape[1:]) for
                    device_id in range(device_num)]
            elif isinstance(node, (NodeFwd, NodeBwdA)):
                new_nodes = [
                    # assume inputs[0] as activation (for NodeFwd) or activation's gradient (for NodeBwdA) and inputs[1] as weight
                    Node(type=node.type,
                         inputs=[oracle_func(node, node.inputs[0], device_id, device_num).query("find_new_input"), #batch-split
                                 oracle_func(node, node.inputs[1], device_id, device_num).query("find_new_input")], #replicated
                         outputs=Tensor(node.outputs.shape[0] // device_num + node.outputs.shape[1:])) #output batch-split
                    for device_id in range(device_num)]
            elif isinstance(node, (NodeBwdW)): #backward that computing weight's gradient
                # assume inputs[0] as activation's gradient and inputs[1] as activation, both with batch-dim
                new_nodes = [
                    Node(type=node.type,
                         inputs=[oracle_func(node, node.inputs[0], device_id, device_num).query("find_new_input"), #batch-split
                                 oracle_func(node, node.inputs[1], device_id, device_num).query("find_new_input")], #batch-split
                         outputs=[Tensor(node.outputs.shape[0])]) #shape unchanged, but only 1/N value
                    for device_id in range(device_num)]
            elif isinstance(node, (NodeOpt)):
                new_nodes = trans(node, algo.replica, device_num)  # replicated optimizers
                [sched_s(node=x.item, dev=x.idx) for x in index_enumerate(nodes)]

            g.replace(node, new_nodes)

        #omit device assign and allreduce insertion
    elif method == 4: #tensor dimention info based manipulation
        pass





data_parallel_raw(graph)