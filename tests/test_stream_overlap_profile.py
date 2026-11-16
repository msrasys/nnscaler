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
