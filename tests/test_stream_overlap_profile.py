from nnscaler.profiler.stream_overlap import analyze, duration, intersection


def test_union_prevents_double_counting_concurrent_communication():
    def kernel(name, start, end, stream):
        return dict(name=name, ts=start, dur=end-start, ph='X', cat='kernel',
                    args=dict(device=0, stream=stream))
    events = [dict(name='stream_overlap/step/1', ts=0, dur=100, ph='X'),
              kernel('intranode::dispatch', 10, 40, 7),
              kernel('intranode::combine', 30, 50, 7),
              kernel('gemm', 20, 45, 8),
              kernel('ncclAllReduce', 10, 50, 9),
              kernel('intranode::dispatch', 110, 150, 7)]
    result = analyze(events)['0']
    assert result['communication_fraction'] == .4
    assert result['uncovered_fraction'] == .15
    assert result['hidden_fraction'] == .625
    assert result['wall_ms'] == .1


def test_intersection_handles_disjoint_windows_and_empty_inputs():
    assert intersection([(0, 10), (15, 20)], [(5, 17)]) == [(5, 10), (15, 17)]
    assert duration([(0, 10), (5, 20)]) == 20
    assert intersection([], [(0, 1)]) == []


def test_dispatch_combine_phases_follow_launches_not_gpu_execution_time():
    def cpu(name, start, end, external):
        return dict(name=name, ts=start, dur=end-start, ph='X', cat='cpu_op',
                    tid=1, args={'External id': external})

    def kernel(name, start, end, external=None):
        return dict(name=name, ts=start, dur=end-start, ph='X', cat='kernel',
                    args=dict(device=0, **({'External id': external} if external else {})))

    events = [dict(name='stream_overlap/step/1', ts=0, dur=100, ph='X'),
              cpu('FusedDispatch', 0, 5, 1),
              cpu('autograd::engine::evaluate_function: FusedCombineBackward', 6, 15, 2),
              cpu('aten::empty', 7, 9, 3),
              kernel('deep_ep::get_dispatch_layout', 15, 20, 1),
              kernel('deep_ep::dispatch', 20, 40, 1),
              kernel('deep_ep::dispatch', 50, 70, 3),
              kernel('deep_ep::combine', 80, 90),
              kernel('shared_gemm', 30, 60),
              kernel('ncclAllReduce', 60, 90)]
    report = analyze(events, compute_pattern='shared_gemm')['0']
    assert report['operations']['dispatch']['uncovered_communication_ms'] == .02
    assert report['operations']['combine']['uncovered_communication_ms'] == .01
    assert report['operations']['layout']['communication_ms'] == .005
    assert report['phases']['forward']['operations']['dispatch']['communication_ms'] == .02
    assert report['phases']['backward']['operations']['dispatch']['communication_ms'] == .02
    assert report['phases']['unknown']['operations']['combine']['communication_ms'] == .01
    assert report['operations']['dispatch']['selected_compute_overlap_ms'] == .02
