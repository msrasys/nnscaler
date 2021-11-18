import time
import sys

import torch


def print_each_rank(msg, rank_only=None, outfile=''):
    myrank = torch.distributed.get_rank()
    outfile = sys.stdout if outfile == '' else outfile
    for rank in range(torch.distributed.get_world_size()):
        if rank_only is None:
            if myrank == rank:
                f = open(outfile, 'a') if outfile != sys.stdout else sys.stdout
                f.write('rank [{}]: {}\n'.format(rank, msg))
                if outfile != sys.stdout:
                    f.close()
        else:
            if myrank == rank_only and rank_only == rank:
                f = open(outfile, 'a') if outfile != sys.stdout else sys.stdout
                f.write('rank [{}]: {}\n'.format(rank, msg))
                if outfile != sys.stdout:
                    f.close()
        torch.distributed.barrier()


class CudaTimer:
    r"""
    Singleton Timer
    """
    class __CudaTimer:

        def __init__(self):
            self.start_t = None
            self.stop_t = None
            self.field = dict()
            self.field_data = dict()
    
    instance = None

    def __init__(self):
        if not CudaTimer.instance:
            CudaTimer.instance = CudaTimer.__CudaTimer()
    
    def start(self, field_name='default'):
        """
        Start recording time on the the field

        Note `start` and `stop` on the same field can be called nestly
        """
        torch.cuda.synchronize()
        if field_name not in CudaTimer.instance.field:
            CudaTimer.instance.field[field_name] = list()
            CudaTimer.instance.field_data[field_name] = 0
        CudaTimer.instance.field[field_name].append(time.time())
    
    def stop(self, field_name='default'):
        """
        Return the time span from last `start` on the smae field name to now

        Returns:
            float (ms)
        """
        if field_name not in CudaTimer.instance.field:
            raise RuntimeError("Missing start on the field")
        torch.cuda.synchronize()
        stop_time = time.time()
        start_time = CudaTimer.instance.field[field_name].pop(-1)
        span = stop_time - start_time # in seconds
        CudaTimer.instance.field_data[field_name] += span
        return span

    def duration(self, times, field_name='default'):
        if field_name not in CudaTimer.instance.field:
            raise RuntimeError(f"Missing start on the field {field_name}")
        if len(CudaTimer.instance.field[field_name]) != 0:
            raise RuntimeError(f"timer for field {field_name} not stopped")
        return CudaTimer.instance.field_data[field_name] / times * 1000  # in ms

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def clear(self):
        CudaTimer.instance = CudaTimer.__CudaTimer()

    def print_all(self, times):
        msg = list()
        comm_span = 0
        for field_name in CudaTimer.instance.field_data:
            span = self.duration(times, field_name)
            if 'send' in field_name or 'recv' in field_name:
                comm_span += span
            msg.append('{} : {:.2f} ms'.format(field_name, span))
        msg.append('{} : {:.2f} ms'.format('communication', comm_span))
        msg = ' | '.join(msg)

        print_each_rank(msg)

    def warmup(self, seconds=1.0):
        """
        Warm up GPU for `span` seconds.
        """
        print('> warming up for 1 second')
        data1 = torch.randn((4096, 4096), device=torch.cuda.current_device())
        data2 = torch.randn((4096, 4096), device=torch.cuda.current_device())
        # warm up 1s
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        start = time.time()
        while time.time() - start < seconds:
            out = torch.matmul(data1, data2)
            # if torch.distributed.is_initialized():
            #     torch.distributed.all_reduce(out)
            torch.cuda.synchronize()
