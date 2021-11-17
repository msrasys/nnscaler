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
        # remove identity comm
        for devid in execplan.devices():
            seq = execplan.sequence(devid)
            comms = [su for su in seq if su.stype == SUType.P2P]
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
        # remove redundant comm e.g., recving same tensor from other ranks
        for devid in execplan.devices():
            all_outs = list()
            seq = execplan.sequence(devid)
            for su in seq:
                # zero-output SU will not be removed
                removable = len(su.outputs()) != 0
                for output in su.outputs():
                    if output not in all_outs:
                        removable = False
                        all_outs.append(output)
                if removable:
                    # only recv has output
                    execplan.at(devid).remove(su)
                    if su.stype == SUType.P2P:
                        # remove all the paired send
                        ranks = su.nodes(0).recv_ranks
                        if len(ranks) > 1:
                            raise NotImplementedError
                        rank = ranks[0]
                        if su.mirror not in execplan.at(rank):
                            raise RuntimeError("Recv Op not found!")
                        execplan.at(rank).remove(su.mirror)
        return execplan
