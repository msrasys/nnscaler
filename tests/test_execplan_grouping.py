# run tests: 
# pytest ./tests/test_execplan_grouping.py

from typing import Dict, List

import pytest
from cube.execplan.planpass import grouping
from cube.execplan.planpass.grouping import Grouping
from cube.ir.cten import IRCell
from cube.ir.operator import IRDataOperation, IRFwOperation

# Stub object for 'cube.execplan.ExecPlan'
# Commonly the devices are like [0,1,2,...]
class StubExecPlan():
    def __init__(self, devices:List[int], seq:Dict[int, List[IRCell]]) -> None:
        assert all(devid in seq for devid in devices)
        self._devices = devices
        self._seq = seq

    def devices(self):
        return self._devices
    def seq(self, devid:int):
        return self._seq[devid]

# With these settings, all tests here are run twice, with 'grouping._get_new...algo' returning True or False, respectively.
# And all the setting ups and the recovery of this flag happen in the background.
#
# By runninng tests in both environments, we can check the consistency of the old and new algorithms.
@pytest.fixture(params=[True, False], autouse=True)
def setup_and_cleanup(request:pytest.FixtureRequest) -> None:
    flag = grouping._get_use_new_grouping_algo()
    grouping._set_use_new_grouping_algo(request.param)
    yield
    grouping._set_use_new_grouping_algo(flag)


def test_grouping_forward_single_group():
    execplan = StubExecPlan([0], {0: [IRFwOperation(f"op{i}", f"sign{i}", i, i) for i in range(1, 10)] })
    # each type: Dict[DeviceIdInt, List[List[IRCell]] ]
    fwgroups, bpgroups = Grouping.group(execplan)

    assert len(fwgroups) == 1 # one device
    assert len(fwgroups[0]) == 1 # one group
    assert all(fnode.name == f"op{i+1}" for i, fnode in enumerate(fwgroups[0][0]))

    assert len(bpgroups) == 1
    assert len(bpgroups[0]) == 1
    assert bpgroups[0][0] is None
    

def test_grouping_forward_interleaving_excluded_nodes():
    execplan = StubExecPlan([0], {0: [
            IRFwOperation(f"op{i}", f"sign{i}", i, i) if i % 2 == 0
            else IRDataOperation(i, (2,)*i) # IRDataOperation is the IRCell to exclude from the group
            for i in range(1, 9)  # [1,2,...,8]
        ] })
    # each type: Dict[DeviceIdInt, List[List[IRCell]] ]
    fwgroups, bpgroups = Grouping.group(execplan)

    assert len(fwgroups) == 1
    assert len(fwgroups[0]) == 4
    assert all(len(fwgroup) == 1 and fwgroup[0].name == f"op{i}" for fwgroup, i in zip(fwgroups[0], [2,4,6,8]))

    assert len(bpgroups) == 1
    assert len(bpgroups[0]) == 4
    assert all(bpgroup is None for bpgroup in bpgroups[0])