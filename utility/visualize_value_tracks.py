import argparse
import matplotlib.pyplot as plt
from nnscaler.graph import IRGraph
from matplotlib.patches import FancyArrowPatch
from nnscaler.ir.cten import IR, IRTensor, IRObject


class Visualizer:
    NUM_ROWS_PER_OP = 3
    TEXT_HEIGHT_IN_INCH = 0.4
    PER_OP_GAP_IN_INCH = 0.2
    PER_ROW_HEIGHT_IN_INCH = TEXT_HEIGHT_IN_INCH * 1.1
    PER_OP_HEIGHT_IN_INCH = PER_ROW_HEIGHT_IN_INCH * NUM_ROWS_PER_OP
    PER_INOUT_GAP = 0.01

    INIT_Y = 0.001
    INIT_X = 0.001

    def __init__(self, graph):
        self.graph = graph
        self.value_loc = {}
        self.ops = [node for node in self.graph.nodes() if node.isfw()]

        self.fig_heigth_in_inch = (
            self.PER_OP_HEIGHT_IN_INCH + self.PER_OP_GAP_IN_INCH
        ) * (len(self.ops) + 1)
        self.coord_per_inch = 1.0 / self.fig_heigth_in_inch
        self.per_op_height = self.PER_OP_HEIGHT_IN_INCH * self.coord_per_inch
        self.per_row_height = self.per_op_height / self.NUM_ROWS_PER_OP
        self.per_op_gap = self.PER_OP_GAP_IN_INCH * self.coord_per_inch

        self.fig, self.ax = plt.subplots(figsize=(30, self.fig_heigth_in_inch))
        self.ax.axis('off')
        self.ax.invert_yaxis()

    def draw_value(self, value, value_track, cur_x, cur_y, previous_value_loc):
        t = self.ax.text(cur_x, cur_y, str(value),
            fontsize=14, ha="left", va="top")
        bbox = t.get_window_extent()
        bbox = bbox.transformed(self.ax.transData.inverted())
        if value_track is not None:
            if value_track.value_id in previous_value_loc:
                prev_x, prev_y = previous_value_loc[value_track.value_id]
                arrow = FancyArrowPatch(
                    (prev_x, prev_y),
                    (cur_x + bbox.width/2, cur_y),
                    arrowstyle="Simple,tail_width=0.25,head_width=1,head_length=1",
                    mutation_scale=6,
                    color="#2c7bb6",
                    linewidth=0.02,
                    connectionstyle="arc3,rad=0",
                    alpha=0.5,
                    zorder=4
                )
                self.ax.add_patch(arrow)
            self.value_loc[value_track.value_id] = (cur_x + bbox.width/2, cur_y)

        cur_x += bbox.width + self.PER_INOUT_GAP/2
        return cur_x

    def draw_obj(self, obj, cur_x, cur_y, previous_value_loc):
        if isinstance(obj, IRTensor):
            cur_x = self.draw_value('T(', None, cur_x, cur_y, previous_value_loc)
            for i, d in enumerate(obj.shape):
                if i > 0:
                    cur_x = self.draw_value(',', None, cur_x, cur_y, previous_value_loc)
                cur_x = self.draw_value(str(d), obj.dim_tracks[i], cur_x, cur_y, previous_value_loc)
            cur_x = self.draw_value(')', None, cur_x, cur_y, previous_value_loc)
        else:
            assert isinstance(obj, IRObject)
            cur_x = self.draw_value('O(', None, cur_x, cur_y, previous_value_loc)
            cur_x = self.draw_value(str(obj.value), obj.value_track, cur_x, cur_y, previous_value_loc)
            cur_x = self.draw_value(')', None, cur_x, cur_y, previous_value_loc)
        cur_x += self.PER_INOUT_GAP
        return cur_x

    def draw_objs(self, objs, cur_x, cur_y):
        previous_value_loc = dict(self.value_loc)
        for inp in objs:
            cur_x = self.draw_obj(inp, cur_x, cur_y, previous_value_loc)

    def draw_graph_inputs(self, g, cur_x, cur_y):
        label = "GRAPH IN: "
        t = self.ax.text(cur_x, cur_y, label,
                fontsize=14, fontweight="bold", ha="left", va="top")

        bbox = t.get_window_extent()
        bbox = bbox.transformed(self.ax.transData.inverted())
        cur_x = cur_x + bbox.width + self.PER_INOUT_GAP

        ir_objs = []
        for inp in g.inputs():
            if isinstance(inp, (IRObject, IRTensor)):
                ir_objs.append(inp)
            elif isinstance(inp, IRObject):
                sub_objs = IR.get_objects(inp.value)
                if sub_objs:
                    ir_objs.extend(sub_objs)
                else:
                    ir_objs.append(inp)

        self.draw_objs(ir_objs, cur_x, cur_y)

    def draw_inout(self, node, cur_y, is_in):
        if is_in:
            ir_objs = node.iobjs()
            label = "IN: "
            cur_y += self.per_row_height
        else:
            ir_objs = node.oobjs()
            label = "OU: "
            cur_y += self.per_row_height * 2

        t = self.ax.text(self.INIT_X, cur_y, label,
                fontsize=14, fontweight="bold", ha="left", va="top")

        bbox = t.get_window_extent()
        bbox = bbox.transformed(self.ax.transData.inverted())
        cur_x = self.INIT_X + bbox.width + self.PER_INOUT_GAP

        self.draw_objs(ir_objs, cur_x, cur_y)

    def visualize(self):
        self.draw_graph_inputs(self.graph, self.INIT_X, self.INIT_Y)
        cur_y = self.INIT_Y + (self.per_op_height + self.per_op_gap)/2

        for node in self.ops:
            op_name = node.name
            self.ax.text(self.INIT_X, cur_y, op_name + ":",
                    fontsize=16, fontweight="bold", ha="left", va="top")

            self.draw_inout(node, cur_y, is_in=True)
            self.draw_inout(node, cur_y, is_in=False)

            cur_y += self.per_op_height + self.per_op_gap


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'graphfile',
        type=str,
        help="Graph dump file"
    )
    parser.add_argument(
        'imagefile',
        type=str,
        nargs='?',
        default=None,
        help="Save generated image to file"
    )
    args = parser.parse_args()
    g = IRGraph.load(args.graphfile)
    visualizer = Visualizer(g)
    visualizer.visualize()
    if args.imagefile:
        plt.savefig(args.imagefile, bbox_inches='tight', dpi=100)
    plt.show()
