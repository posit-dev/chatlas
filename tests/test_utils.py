import pytest

from chatlas._utils import run_coro_sync


async def _forty_two():
    return 42


def test_run_coro_sync_outside_loop():
    assert run_coro_sync(_forty_two, what="do a thing") == 42


@pytest.mark.asyncio
async def test_run_coro_sync_inside_loop_errors():
    with pytest.raises(RuntimeError, match="running event loop"):
        run_coro_sync(_forty_two, what="do a thing")
