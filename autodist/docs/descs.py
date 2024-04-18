@dataclass
class AutoDistConfig:
    recompute: bool = False
    mem_granularity_mb: bool = 1


@dataclass
class NodePartitionDesc:
    # list element: (idx, dim, num), the order matters
    desc: List[Tuple[int, int, int]]


@dataclass
class DeviceDesc:
    dev_num: int
    peak_mem_gb: int = 30
    connection: str = 'NV3'


@dataclass
class TensorParallelDesc:
    partition_descs: List[NodePartitionDesc]
    recompute_groups: List[List[int]]
    logical_desc: DeviceDesc


@dataclass
class ParallelDesc:
    stages: List[Tuple[TensorParallelDesc, DeviceDesc]]


class TensorParallelDPSolver:

    # resource is a logical mesh
    def __init__(graph: IRGraph, resource: DeviceDesc, config: AutoDistConfig):
        pass

    def solver():
        pass

    # temp design
    def get_optimal_plan(
            start_desc: NodePartitionDesc, end_desc: NodePartitionDesc
    ) -> Tuple[TensorParallelDesc, float, int]:
        pass


class PipelineDPSolver:

    # resource is a physical mesh
    def __init__(graph: IRGraph, resource: DeviceDesc, config: AutoDistConfig):
        pass

    def solver():
        pass

    def get_optimal_plan() -> Tuple[ParallelDesc, float]:
        pass
