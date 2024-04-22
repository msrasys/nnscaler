import cppimport.import_hook
import nnscaler.autodist.dp_solver as dp_solver

# use a naive ffn to test the dynamic programming solver
# the ffn has 3 layers
# - linear layer
# - relu layer
# - linear layer
# each operator has 2 partition options

def test_dp_solver():
    solver = dp_solver.DPSolver(True, 0, 80 * 1024, 1, 1)
    solver.add_interval(0, 2)

    solver.add_node(0, 0, [0], [], 2)
    solver.add_partition(0, 0, 1, 1, 1, 1, 1, 0, [[]])
    solver.add_partition(0, 1, 2, 2, 2, 2, 2, 1, [[]])

    solver.add_node(1, 1, [1], [0], 2)
    solver.add_partition(1, 0, 0.5, 1, 1, 1, 1, 0, [[0.1, 1]])
    solver.add_partition(1, 1, 1, 2, 2, 2, 2, 1, [[1, 0]])

    solver.add_node(2, 2, [2], [1], 2)
    solver.add_partition(2, 0, 1, 1, 1, 1, 1, 0, [[0.2, 1]])
    solver.add_partition(2, 1, 2, 2, 2, 2, 2, 1, [[1, 0]])

    solver.solve()

    ans = solver.get_results(0, 2)

    best = ans[0]

    # optimal all time 1 + 0.5 + 0.1 + 1 + 0.2 = 2.8
    assert best.all_time == 2.8
    # optimal inner time 1 + 0.5 + 1 = 2.5
    assert best.inner_time == 2.5
    # the optimal plan is each operator's first partition
    assert best.path == [(0, 0), (1, 0), (2, 0)]
