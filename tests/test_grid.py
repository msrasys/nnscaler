from cube.graph.adapter.layout import GridLayout
from cube.graph.tensor import IRFullTensor

def test_grid():

    tensor = IRFullTensor(shape=[8192,8192], name='src')
    
    src = GridLayout.grid(tensor, r=2, v=2, dims=[0, 0])
    dst = GridLayout.grid(tensor, r=4, v=1, dims=[0, 0])
    
    path = src.path(dst)
    for grid in path:
        print(grid)


if __name__ == '__main__':
    test_grid()
