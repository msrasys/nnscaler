from typing import Dict, Any


class DistAlgorithmFactory:

    class __DistAlgorithmFactory:

        def __init__(self):
            # [LogicOp][tag] = algorithm
            self._algos: Dict[Any, Dict[str, Any]] = dict()

    instance = None
    
    def __init__(self):
        if not DistAlgorithmFactory.instance:
            DistAlgorithmFactory.instance = DistAlgorithmFactory.__DistAlgorithmFactory()
            self._load_predefined_algos()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def exist(self, op, tag=None):
        """
        Check if the factory has op's algorithm recorded

        Returns:
            True if have, False if not
        """
        if tag is None:
            return op in self.instance._algos
        else:
            return op in self.instance._algos and tag in self.instance._algos[op]

    def register(self, op, algorithm, tag: str):
        """
        Register a holistic op (class) as one of the anchors 
        """
        if op not in self.instance._algos:
            self.instance._algos[op] = dict()
        self.instance._algos[op][tag] = algorithm

    def algorithms(self, op, tag = None):
        """
        Get op tranformed algorithms

        Args:
            op (IRFwOperation): index for the holist op factory
            args, kwargs: (logical) tensor inputs

        Returns:
            algorithm class
        """
        if op not in self.instance._algos:
            raise KeyError("Op {op} is not registered in factory")
        if tag:
            return self.instance._algos[op][tag]
        else:
            return self.instance._algos[op].values()

    def _load_predefined_algos(self):

        import cube.algorithm.ops.dataloader as dataloader
        self.register(dataloader.IRDataOperation, dataloader.DPDataLoader, tag='data')

        import cube.algorithm.ops.dimops as dimops
        self.register(dimops.IRDimops, dimops.DimSplitEinops, tag='dim')
        self.register(dimops.IRDimops, dimops.SimpleViewSplitEinops, tag='view_simp')

        import cube.algorithm.ops.conv as conv
        self.register(conv.IRConv2D, conv.DimSplitConv2D, tag='dim')
        self.register(conv.IRConv2D, conv.HaloSplitConv2D, tag='halo')
        self.register(conv.IRConv3D, conv.HaloSplitConv3D, tag='halo')

        import cube.algorithm.ops.pad as pad
        self.register(pad.IRPad, pad.DimSplitPad, tag='dim')
        
        import cube.algorithm.ops.select as select
        self.register(select.IRSelect, select.DimSplitSelect, tag='dim')
        self.register(select.IRSlice, select.DimSplitSlice, tag='dim')
        
        import cube.algorithm.ops.scatter as scatter
        self.register(scatter.IRSelectScatter, scatter.DimSplitScatter, tag='dim')
        
        import cube.algorithm.ops.creators as creators
        self.register(creators.IRToTensor, creators.DimSplitTo, tag='dim')
        self.register(creators.IROnes, creators.DimSplitOnes, tag='dim')
        self.register(creators.IRRand, creators.DimSplitRand, tag='dim')
        # import cube.algorithm.ops.elementwise as elew
        # self.register(elew.ElementWise, elew.ElementWiseDimParallel, tag='dim')
        # self.register(elew.Add, elew.AddDimParallel, tag='dim')

        # import cube.algorithm.ops.layernorm as ln
        # self.register(ln.LayerNorm, ln.LayerNormDimParallel, tag='dim')

        # import cube.algorithm.ops.activation as activation
        # self.register(activation.Activation, activation.ActivationDimParallel, tag='dim')
        # self.register(activation.Dropout, activation.DropoutDimParallel, tag='dim')
        # self.register(activation.Softmax, activation.SoftmaxDimParallel, tag ='dim')

        # import cube.algorithm.ops.reduce as reduce
        # self.register(reduce.Sum, reduce.SumDimParallel, tag='dim')

        # import cube.algorithm.ops.memory as mem
        # self.register(mem.Transpose, mem.TransposeDimParallel, tag='dim')
