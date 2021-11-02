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

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def register(self, op, algorithm, tag: str):
        """
        Register a holistic op (class) as one of the anchors 
        """
        if op not in self.instance._algos:
            self.instance._algos[op] = dict()
        self._algos[op][tag] = algorithm

    def algorithms(self, op, tag = None):
        """
        Get op tranformed algorithms

        Args:
            op (IROperation): index for the holist op factory
            args, kwargs: (logical) tensor inputs

        Returns:
            algorithm class
        """
        if tag:
            return self.instance._algos[op][tag]
        else:
            return self.instance._algos[op].values()
