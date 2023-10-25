import asyncio
import logging as log


class QueueCancelError(asyncio.QueueEmpty):
    pass


class CancelQueue(asyncio.Queue):
    def __init__(self, *args, sentinel=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sentinel = sentinel
        self._cancelled = False

    async def get(self):
        if self._cancelled:
            raise QueueCancelError
        return await super().get()

    def _put(self, item):
        if self._cancelled:
            log.warning("put to cancelled queue")
            return
        return super()._put(item)

    def cancel(self):
        self._cancelled = True
        for _ in range(len(self._getters)):
            self.put_nowait(self.sentinel)
