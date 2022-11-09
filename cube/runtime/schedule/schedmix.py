"""
Schedule Plan designed for Interplaced Pipeline
"""

from typing import Callable, Iterable, List, Optional
import torch

from cube.runtime.schedule.strategy import ScheduleABC

def debug_msg(msg: str, ranks):
    myrank = torch.distributed.get_rank()
    if myrank in ranks:
        print(f'rank [{myrank}]: {msg}')


class ScheduleMix(ScheduleABC):
    """
    Emb -> Encoder -> Demb -> Decoder
    
    All communication will start at begining of each step and
    finish at the end of step. No communication will happen cross
    step, i.e., send from the previous step and recv at the next step.
    """
    @staticmethod
    def run(encoder_barrier: Callable,
            decoder_barrier: Callable,
            segment: Callable,
            rfadapter: Optional[Callable], # segment adapter
            sfadapter: Optional[Callable], # segment adapter
            rbadapter: Optional[Callable], # segment adapter
            sbadapter: Optional[Callable], # segment adapter
            enc_badapter: Optional[Callable], # sharding encoder gradient input prepare adapter
            dec_fadapter: Optional[Callable], # sharding decoder input prepare adapter
            dec_badapter: Optional[Callable], # sharding decoder gradient input prepare adapter
            dataloader: Iterable,
            stage_id: int,
            num_stages: int,
            num_microbatch: int,
            reducers: List[Callable],
            recompute: bool = False):
        
        assert num_stages >= 4, f"Only support for stage number >= 4."

        enc_emb_stage = 0
        dec_emb_stage = num_stages // 2
        
        fw_ofst = -(stage_id // 2)
        bw_ofst = -(num_stages - 1 - (stage_id // 2))

        # sharding encoder embed inputs / outputs
        shard_enc_inputs, shard_enc_outputs = (None,), (None,)
        shard_enc_input_grads, shard_enc_output_grads = (None,), (None,)
        # sharding decoder embed inputs / outputs
        shard_dec_inputs, shard_dec_outputs = (None,), (None,)
        shard_dec_input_grads, shard_dec_output_grads = (None,), (None,)
        # segement inputs / outputs
        segment_inputs, segment_outputs = (None,), (None,)
        segment_input_grads, segment_output_grads = (None,), (None,)

        for step in range(num_microbatch + num_stages - 1):
            fmid, bmid = step + fw_ofst, step + bw_ofst
            encoder_fw_mid = step
            decoder_fw_mid = step - num_stages // 2 // 2
            encoder_bw_mid = step + 1 - num_stages // 2 * 2
            decoder_bw_mid = step + 1 - int(num_stages // 2 * 1.5)
            do_forward = 0 <= fmid and fmid < num_microbatch
            do_backward = 0 <= bmid and bmid < num_microbatch

            # step1: sharding encoder forward
            if 0 <= encoder_fw_mid and encoder_fw_mid < num_microbatch:
                data = ScheduleMix.dataloader_step(dataloader)
                shard_enc_outputs = ScheduleMix.forward_step(encoder_barrier, *data)
                ScheduleMix.push_tail('shard_enc_inputs', data)
                ScheduleMix.push_tail('shard_enc_outputs', shard_enc_outputs)
                shard_enc_outputs = tuple(t.detach().requires_grad_() for t in shard_enc_outputs)

            # step2: sharding decoder forward
            if 0 <= decoder_fw_mid and decoder_fw_mid < num_microbatch:
                if stage_id == dec_emb_stage - 1:
                    shard_dec_inputs = tuple(t.detach().requires_grad_() for t in segment_outputs)
                    ScheduleMix.adapter_step(dec_fadapter, True, *shard_dec_inputs)
                else:
                    shard_dec_inputs = ScheduleMix.adapter_step(dec_fadapter, True)
                shard_dec_outputs = ScheduleMix.forward_step(decoder_barrier, *shard_dec_inputs)
                ScheduleMix.push_tail('shard_dec_inputs', shard_dec_inputs)
                ScheduleMix.push_tail('shard_dec_outputs', shard_dec_outputs)
                shard_dec_outputs = tuple(t.detach().requires_grad_() for t in shard_dec_outputs)

            # step3: forward then backward
            if stage_id % 2 == 0:

                # After barrier communication: send backward recv forward =========>
                if segment_input_grads != (None,):
                    ScheduleMix.adapter_step(sbadapter, False, *segment_input_grads)
                    segment_input_grads = (None,)
                if do_forward:
                    if stage_id == enc_emb_stage:
                        segment_inputs = shard_enc_outputs
                    elif stage_id == dec_emb_stage:
                        segment_inputs = shard_dec_outputs
                    else:
                        segment_inputs = ScheduleMix.adapter_step(rfadapter, True)
                # <===============================================================

                segment_outputs = (None,)
                if do_forward:
                    ScheduleMix.push_tail('segment_inputs', segment_inputs)
                    if recompute:
                        with torch.no_grad():
                            segment_outputs = ScheduleMix.forward_step(segment, *segment_inputs)
                        ScheduleMix.push_tail('segment_outputs', None)
                    else:
                        segment_outputs = ScheduleMix.forward_step(segment, *segment_inputs)
                        ScheduleMix.push_tail('segment_outputs', segment_outputs)
                    
                # recompute
                if recompute:
                    inputs = ScheduleMix.pop_head('segment_inputs', inputs)
                    ScheduleMix.pop_head('segment_outputs', outputs)
                    outputs = ScheduleMix.forward_step(segment, *inputs)
                    ScheduleMix.push_head('segment_inputs', inputs)
                    ScheduleMix.push_head('segment_outputs', outputs)

                # Inter barrier communication: recv backward send forward ======>
                if do_backward:
                    segment_output_grads = ScheduleMix.adapter_step(rbadapter, False)
                if segment_outputs != (None,):
                    ScheduleMix.adapter_step(sfadapter, True, *segment_outputs)
                # <===============================================================

                segment_input_grads = (None,)
                if do_backward:
                    inputs = ScheduleMix.pop_head('segment_inputs')
                    outputs = ScheduleMix.pop_head('segment_outputs')
                    segment_input_grads = ScheduleMix.backward_step(inputs, outputs, segment_output_grads)

            # step3: backward then forward
            if stage_id % 2 == 1:
                
                # After barrier communication: recv backward send forward =========>
                if do_backward:
                    if stage_id == dec_emb_stage - 1:
                        segment_output_grads = shard_dec_input_grads
                    else:
                        segment_output_grads = ScheduleMix.adapter_step(rbadapter, False)
                if segment_outputs != (None,):
                    segment_input_grads = ScheduleMix.adapter_step(sfadapter, True, *segment_outputs)
                # <===============================================================

                segment_input_grads = (None,)
                if do_backward:
                    inputs = ScheduleMix.pop_head('segment_inputs')
                    outputs = ScheduleMix.pop_head('segment_outputs')
                    segment_input_grads = ScheduleMix.backward_step(inputs, outputs, segment_output_grads)

                # Inter barrier communication: send backward recv forward ========>
                if segment_input_grads != (None,):
                    ScheduleMix.adapter_step(sbadapter, False, *segment_input_grads)
                if do_forward:
                    segment_inputs = ScheduleMix.adapter_step(rfadapter, True)
                # <===============================================================

                segment_outputs = (None,)
                if do_forward:
                    ScheduleMix.push_tail('segment_inputs', segment_inputs)
                    if recompute:
                        with torch.no_grad():
                            segment_outputs = ScheduleMix.forward_step(segment, *segment_inputs)
                            ScheduleMix.push_tail('segment_outputs', None)
                    else:
                        segment_outputs = ScheduleMix.forward_step(segment, *segment_inputs)
                        ScheduleMix.push_tail('segment_outputs', segment_outputs)

                # recompute
                if recompute:
                    inputs = ScheduleMix.pop_head('segment_inputs', inputs)
                    ScheduleMix.pop_head('segment_outputs', outputs)
                    outputs = ScheduleMix.forward_step(segment, *inputs)
                    ScheduleMix.push_head('segment_inputs', inputs)
                    ScheduleMix.push_head('segment_outputs', outputs)

            # step 4: sharding decoder backward
            if 0 <= decoder_bw_mid and decoder_bw_mid < num_microbatch:
                if stage_id == dec_emb_stage:
                    assert segment_input_grads != (None,)
                    shard_dec_output_grads = segment_input_grads
                    ScheduleMix.adapter_step(dec_badapter, False, *shard_dec_output_grads)
                else:
                    shard_dec_output_grads = ScheduleMix.adapter_step(dec_badapter, False)
                
                inputs = ScheduleMix.pop_head('shard_dec_inputs')
                outputs = ScheduleMix.pop_head('shard_dec_outputs')
                shard_dec_input_grads = ScheduleMix.backward_step(
                    inputs, outputs, shard_dec_output_grads)

            # step 5: sharding encoder backward
            if 0 <= encoder_bw_mid and encoder_bw_mid < num_microbatch:
                if stage_id == enc_emb_stage:
                    assert segment_input_grads != (None,)
                    shard_enc_output_grads = segment_input_grads
                    ScheduleMix.adapter_step(enc_badapter, False, *shard_enc_output_grads)
                else:
                    shard_enc_output_grads = ScheduleMix.adapter_step(enc_badapter, False)
                
                inputs = ScheduleMix.pop_head('shard_enc_inputs')
                outputs = ScheduleMix.pop_head('shard_enc_outputs')
                shard_enc_input_grads = ScheduleMix.backward_step(
                    inputs, outputs, shard_enc_output_grads)

        for reducer in reducers:
            reducer()
        
        ScheduleMix.assert_empty()
