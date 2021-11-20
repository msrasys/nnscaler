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

        import cube.algorithm.ops.linear as linear
        self.register(linear.Linear, linear.LinearDataParallel, tag='data')
        self.register(linear.Linear, linear.LinearColumnWeight, tag='column')
        self.register(linear.Linear, linear.LinearRowWeight, tag='row')

        import cube.algorithm.ops.bmm as bmm
        self.register(bmm.BatchLinear, bmm.BatchLinearDataParallel, tag='data')
        self.register(bmm.BatchLinear, bmm.BatchLinearNParallel, tag='n')
        self.register(bmm.BatchLinear, bmm.BatchLinearNParallel, tag='m')
        self.register(bmm.BatchLinear, bmm.BatchLinearNParallel, tag='p')

        import cube.algorithm.ops.elementwise as elew
        self.register(elew.ElementWise, elew.ElementWiseDimParallel, tag='dim')
        self.register(elew.Add, elew.AddDimParallel, tag='dim')

        import cube.algorithm.ops.activation as activation
        self.register(activation.Activation, activation.ActivationDimParallel, tag='dim')
        self.register(activation.Dropout, activation.DropoutDimParallel, tag='dim')
        self.register(activation.Softmax, activation.SoftmaxDimParallel, tag ='dim')

        import cube.algorithm.ops.reduce as reduce
        self.register(reduce.Reduce, reduce.ReduceDataParallel, tag='data')

        import cube.algorithm.ops.complex as complex
        self.register(complex.CubeComplexToQKV, complex.CubeToQKVDataParallel, tag='data')
        self.register(complex.CubeComplexToQKV, complex.CubeToQKVHeadParallel, tag='head')

        self.register(complex.CubeComplexTrilMask, complex.CubeTrilMaskDataParallel, tag='data')
        self.register(complex.CubeComplexTrilMask, complex.CubeTrilMaskHeadParallel, tag='head')
    
        self.register(complex.CubeComplexAttnView, complex.CubeAttnViewDataParallel, tag='data')
        self.register(complex.CubeComplexAttnView, complex.CubeAttnViewHeadParallel, tag='head')
