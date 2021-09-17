from typing import List


class DistAlgorithmFactory:

    def __init__(self):

        self.algorithms = list()

    def __len__(self):
        """
        Return the number of holistic op registered
        """
        return len(self.algorithms)

    def register(self, algorithm):
        """
        Register a holistic op (class) as one of the anchors 
        """
        self.algorithms.append(algorithm)

    def get_op(self, idx, outputs, *args, **kwargs):
        """
        Get holistic operator based on idx

        The holistic operator will be initialized with shapes

        Args:
            idx (int): index for the holist op factory
            args, kwargs: (logical) tensor inputs

        Returns:
            HolisticOp instance
        """
        return self.algorithms[idx](outputs, *args, **kwargs)


class GenericLogicalOp:

    def __init__(self, signature: str):
        """
        Generic logical operator

        signature (str):
            Framework implementation signature,
                e.g., 'torch.nn.functional.linear'
        """
        if not isinstance(signature, str):
            raise TypeError("Expect signature to be a string")
        # factory 
        self.factory = DistAlgorithmFactory()
        # torch impl signature
        self.signature = signature
    
    @staticmethod
    def shape_infer(*args, **kwargs):
        """
        Output shape inference according to inputs

        Args:
            Operator input

        Returns:
            shapes tuple(list[int]): shape for each output tensor
        """
        raise NotImplementedError("Expected a shape infer engine")
    
    def register_algorithm(self, algorithm):
        """
        Register a distributed algoritm description
        """
        self.factory.register(algorithm)

    def translate(self, config):
        """
        Translate the algorithm to implementation
        """
        raise NotImplementedError("Expected a tranlation for operator")


class ElementSameInputOp(GenericLogicalOp):

    def __init__(self):
        """
        Elementwise Operator
        """
        super().__init__()

    @staticmethod
    def shape_infer(input: List[int], *args, **kwargs):
        """
        Element-wise single input op
        """
        return [input]
