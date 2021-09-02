
class ExecutionPlan:

    def __init__(self, seq, ndevice):
        """
        Seq: action sequence
        ndevice: device number
        """
        self.seq = seq
        self.ndevice = ndevice
        self.device_timeline = None
        self.device_actions = None
    
    def gen(self):
        """
        Generate execution plan
        """
        # timeline: [(start_time, end_time)]
        self.device_timeline = [list() for _ in range(self.ndevice)]
        self.device_actions = [list() for _ in range(self.ndevice)]

        for action in self.seq:
            if action.device == -1 or action.device >= self.ndevice:
                raise RuntimeError("action {} device not assigned or out of boundary".format(action))
            if len(self.device_timeline[action.device]) == 0:
                start_time = 1
            else:
                start_time = self.device_timeline[action.device][-1][1]
            for dev_id, (timeline, dev_actions) in enumerate(zip(self.device_timeline, self.device_actions)):
                if dev_id == action.device:
                    continue
                # go through to check if the action has dependencies
                for (_, end_time), dev_action in zip(timeline[::-1], dev_actions[::-1]):
                    if action.depends_on(dev_action):
                        # print('find dependency {} -> {}, end time: {}'.format(action, dev_action, end_time))
                        start_time = max(start_time, end_time)
                        break
                    elif dev_action.depends_on(action):
                        raise RuntimeError("Action happened before")
            # update timeline
            self.device_timeline[action.device].append((start_time, start_time + action.est_latency))
            self.device_actions[action.device].append(action)
    
    def actions(self, device_id):
        """
        Get action sequence for the specific device id
        """
        if device_id >= self.ndevice:
            raise ValueError(f"device id out of boundary ({device_id} >= {self.ndeivce})")
        if self.device_actions is None:
            self.gen()
        return self.device_actions[device_id]
    
    def timeline(self, device_id):
        """
        Get action timeline for the specific device id
        """
        if device_id >= self.ndevice:
            raise ValueError(f"device id out of boundary ({device_id} >= {self.ndeivce})")
        if self.device_timeline is None:
            self.gen()
        return self.device_timeline[device_id]

    def get_time(self):
        if self.device_timeline is None:
            self.gen()
        return max(
            [timeline[-1][1] for timeline in self.device_timeline if len(timeline) != 0]
        )
    
    def get_memory(self):
        if self.device_timeline is None:
            self.gen()

        def device_memory(actions):
            max_mem = 0
            cur_mem = 0
            for action in actions:
                cur_mem += action.est_memory
                max_mem = max(cur_mem, max_mem)
            return max_mem

        return max(
            [device_memory(actions) for actions in self.device_actions]
        )

    def draw(self, outfile='./execplan.png'):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        plt.rcParams['figure.figsize'] = (12.0, 4.0)

        if self.device_actions is None:
            self.gen()

        fig, ax = plt.subplots()
        plan_time = self.get_time()

        # xaxis
        ax.set_xlim((1, plan_time))
        plt.xticks(list(range(1, plan_time+1, 1)))
        ax.xaxis.grid(True, linestyle='--')
        plt.xlabel('time')

        # yaxis
        ax.set_ylim((0.5, self.ndevice+0.5))
        plt.yticks(list(range(1, self.ndevice+1, 1)))
        ax.invert_yaxis()
        plt.ylabel('device id')

        ax.set_aspect('equal')

        for devid in range(len(self.device_actions)):
            timeline = self.device_timeline[devid]
            actions = self.device_actions[devid]
            for action, (start, end) in zip(actions, timeline):
                # draw 
                color = 'blue' if (end - start) == 1 else 'orange'
                rec = Rectangle((start, devid + 0.5), end-start, 1,
                                         color=color, ec='black', lw=1.5)
                ax.add_artist(rec)
                rx, ry = rec.get_xy()
                cx = rx + rec.get_width() / 2.0
                cy = ry + rec.get_height() / 2.0
                anno = action.name if action.fid is None else action.fid
                ax.annotate(anno, (cx, cy), color='w', weight='bold',
                            fontsize=10, ha='center', va='center')
        # plt.grid()
        plt.savefig(outfile)
    
    def to_json(self):
        return [repr(action) for action in self.seq]
