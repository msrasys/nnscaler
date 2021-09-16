
class IDGenerator:
    """
    Tensor / Operator manager. To guarantee that each IRTensor / IROperator id
    is unique and progressively increases.
    
    This class is designed in singleton pattern.
    """
    class __IDGenerator:
        def __init__(self):

            self._tensor_id = 0
            self._op_id = 0

    instance = None

    def __init__(self):
        if not IDGenerator.instance:
            IDGenerator.instance = IDGenerator.__IDGenerator()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def gen_tensor_id(self):
        self.instance._tensor_id += 1
        return self.instance._tensor_id

    def gen_op_id(self):
        self.instance._op_id += 1
        return self.instance._op_id

    def clear(self):
        self.instance._tensor_id = 0
        self.instance._op_id = 0
