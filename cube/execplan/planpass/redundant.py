from cube.execplan import ExectuionPlan
from cube.schedule.su import SUType
from cube.execplan.planpass.planpass import PlanPass


class RemoveRedundantAdapters(PlanPass):

    @staticmethod
    def apply(execplan: ExectuionPlan) -> ExectuionPlan:
        """
        Remove redundant adapters

        A redundant adapter is sending / recving tensors on the same deivce
        """
        for devid in execplan.devices():
            seq = execplan.sequence(devid)
            comms = [su for su in seq if su.stype == SUType.Comm]
            for comm in comms:
                send_ranks = set([devid])
                recv_ranks = set([devid])
                for node in comm.nodes():
                    send_ranks.update(node.send_ranks)
                    recv_ranks.update(node.recv_ranks)
                if list(send_ranks) != [devid]:
                    continue
                if list(recv_ranks) != [devid]:
                    continue
                # remove
                execplan.at(devid).remove(comm)
        return execplan
