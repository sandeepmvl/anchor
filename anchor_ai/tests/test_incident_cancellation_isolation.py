import asyncio

import anyio
import pytest

from anchor_ai.streaming import cancel_tasks


def test_session_cleanup_does_not_cancel_owner_or_unrelated_voice_future():
    async def exercise():
        owner = asyncio.current_task()
        voice_future = asyncio.get_running_loop().create_future()
        stopped = asyncio.Event()

        async def child():
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                stopped.set()

        task = asyncio.create_task(child())
        await asyncio.sleep(0)
        with anyio.CancelScope() as scope:
            scope.cancel()
            await cancel_tasks([task, owner, task, None])
            assert stopped.is_set()
            assert not owner.cancelling()
            assert not voice_future.cancelled()
        voice_future.set_result('voice remains available')
        assert await voice_future == 'voice remains available'
        assert task.cancelled()

    asyncio.run(exercise())


def test_direct_caller_cancellation_is_not_swallowed():
    async def exercise():
        draining = asyncio.Event()
        finished = asyncio.Event()

        async def child():
            try:
                await asyncio.Event().wait()
            finally:
                draining.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    finished.set()

        task = asyncio.create_task(child())
        await asyncio.sleep(0)
        cleanup = asyncio.create_task(cancel_tasks([task]))
        await draining.wait()
        cleanup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cleanup
        assert finished.is_set()
        assert task.done()

    asyncio.run(exercise())


def test_incident_validation_leaves_voice_future_alone(tmp_path):
    from anchor_ai.runtime import IncidentRuntime, corrected_incident_policy

    async def exercise():
        future = asyncio.get_running_loop().create_future()
        policy = corrected_incident_policy()
        policy['retry_owners'] = ['resilience', 'sdk']
        with pytest.raises(ValueError, match='conflicting_retry_ownership'):
            IncidentRuntime(tmp_path, policy)
        assert not future.cancelled()
        future.set_result('unchanged')
        assert await future == 'unchanged'

    asyncio.run(exercise())
