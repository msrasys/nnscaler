"""Equal-byte batch-size tuning; all transfer mechanisms are unchanged."""
import json
import sys
import experiment as e

if __name__=='__main__':
    nn,llm=e.source_paths()
    for count,size,batch in ((16,16,16),(16,16,4),(128,2,128),(128,2,32),(16,64,16),(16,64,4)):
        name=f'transport-n{count}-mib{size}-numa0-batch{batch}'
        result=e.ROOT/(name+'.json')
        if result.exists():continue
        command=['/usr/bin/numactl','--cpunodebind=0','--membind=0',sys.executable,str(e.ROOT/'standalone-benchmark.py'),'--output',str(result),
            '--tensors',str(count),'--tensor-mib',str(size),'--batch-tensors',str(batch),
            '--iterations','16','--warmup','4','--inflight-mib','1024','--prefetch-mib','1024','--cpu-budget-mib','2048',
            '--compute-repeats','400','--compute-dim','4096']
        code,guard=e.protected(name,command,cwd=llm,env=dict(e.environment(),CUDA_VISIBLE_DEVICES='0'),timeout=600,expected_gib=16,gpu_reserve_mib=16384)
        assert code==0 and not guard['stop_reason']
        row=json.loads(result.read_text())
        print(name,{key:row['results'][key]['median_wall_ms'] for key in ('compute_only','store_roundtrip','overlap_compute')},flush=True)
