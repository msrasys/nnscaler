from cube.graph.adapter.layout import GridLayout
from cube.graph.tensor import IRFullTensor

def test_grid():

    tensor = IRFullTensor(shape=[8192,8192], name='src')
    
    src = GridLayout.grid(tensor, r=2, v=2, dims=[1, 1])
    dst = GridLayout.grid(tensor, r=2, v=1, dims=[2, 1])
    
    path, prims = src.path(dst)
    for grid in path:
        print(grid)
    for prim in prims:
        print(prim)


if __name__ == '__main__':
    test_grid()
