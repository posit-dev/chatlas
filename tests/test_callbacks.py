import time

from chatlas._callbacks import CallbackManager


def test_callbacks_lifo_order():
    callbacks = CallbackManager()
    res1 = None
    res2 = None

    def cb1(value):
        nonlocal res1
        res1 = {"value": value, "time": time.time()}

    def cb2(*args):
        nonlocal res2
        res2 = {"value": args, "time": time.time()}

    callbacks.add(cb1)
    callbacks.add(cb2)

    assert callbacks.count() == 2

    # Callbacks don't return a value
    assert callbacks.invoke({"x": 1, "y": 2}) is None

    # Callbacks receive expected arguments
    assert res1["value"] == {"x": 1, "y": 2}
    assert res2["value"] == ({"x": 1, "y": 2},)

    # Callbacks are invoked in reverse order
    assert res1["time"] > res2["time"]


def test_add_callback_removal():
    callbacks = CallbackManager()
    res1 = None
    res2 = None

    def cb1():
        nonlocal res1
        res1 = time.time()

    def cb2():
        nonlocal res2
        res2 = time.time()

    rm_cb1 = callbacks.add(cb1)
    callbacks.add(cb2)

    assert callbacks.count() == 2
    callbacks.invoke()

    # Unregistering a callback
    res1_first = res1
    res2_first = res2
    rm_cb1()
    assert callbacks.count() == 1
    callbacks.invoke()
    assert res1 == res1_first  # first callback result hasn't changed
    assert res2 > res2_first  # second callback was evaluated

    # Unregistering callbacks are idempotent
    cb_list = callbacks.get_callbacks()
    rm_cb1()  # should not raise an error
    # Callback list hasn't changed
    assert callbacks.get_callbacks() == cb_list
