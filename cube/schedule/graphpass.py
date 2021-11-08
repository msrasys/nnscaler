from cube.schedule.sugraph import SUGraph
from cube.schedule.su import SUType, ScheduleUnit


class SUGraphPass:

    @staticmethod
    def remove_redundant_adapters(sugraph: SUGraph) -> SUGraph:
        """
        Remove redundant adapters

        A redundant adapter is sending and recving
        on the same device
        """
        redundant_adapters = list()
        for su in sugraph.sus():
            if su.stype != SUType.Comm:
                for idx in range(len(su.outputs())):
                    send_adapters, recv_adapters = su.out_adapters(idx)
                    for sadapter, radapter in zip(send_adapters, recv_adapters):
                        # indicate a tensor selection in-device
                        if sadapter.device == radapter.device:
                            if len(sadapter.inputs()) != 1:
                                raise NotImplementedError
                            # indicate identity op:
                            if sadapter.inputs(0).shape == su.outputs(idx).shape:
                                redundant_adapters.append(sadapter)
                                redundant_adapters.append(radapter)

        all_sus = sugraph.sus()
        for adapter in redundant_adapters:
            if adapter in all_sus:
                all_sus.remove(adapter)
        
        sugraph = SUGraph(all_sus)
        return sugraph
                
    @staticmethod
    def merge_small_sus(sugraph: SUGraph) -> SUGraph:
        """
        Merge SU to a larger one if possible
        """
        devices = set()
        for su in sugraph.sus():
            devices.update(set(su.device))
        for device in devices:
            dev_sus = [su for su in sugraph.fsus() if device in su.device]
            merged_su = dev_sus[0]
            for su in dev_sus[1:]:
                merged_su = sugraph.merge(merged_su, su)
                if merged_su is None:
                    merged_su = su
        return sugraph
