from chatlas import StreamController


def test_stream_controller_initial_state():
    ctrl = StreamController()
    assert ctrl.cancelled is False
    assert ctrl.reason is None


def test_stream_controller_cancel():
    ctrl = StreamController()
    ctrl.cancel()
    assert ctrl.cancelled is True
    assert ctrl.reason == "cancelled"


def test_stream_controller_cancel_with_reason():
    ctrl = StreamController()
    ctrl.cancel(reason="timeout")
    assert ctrl.cancelled is True
    assert ctrl.reason == "timeout"


def test_stream_controller_reset():
    ctrl = StreamController()
    ctrl.cancel(reason="timeout")
    ctrl.reset()
    assert ctrl.cancelled is False
    assert ctrl.reason is None
