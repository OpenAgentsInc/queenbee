import typing
from queue import Queue
from typing import Iterable

T = typing.TypeVar(name="T")


class UniqueQueue(Queue):
    """A simple multi-producer, single-consumer queue that only stores a given element once.

    This derives from Queue, and inherits much of the safety/functionality of Queue.

    There is a potential race condition during iteration--when entries are being added, the consumer may miss the
    very latest of them if there is contention. This class should not be used in cases where this matters.
    """

    def __init__(self, maxsize=0, key=None):
        self.key = key
        self.queue = set() if key is None else {}
        super().__init__(maxsize)

    def has(self, key):
        return key in self.queue

    def _init(self, maxsize):
        self.queue = set() if self.key is None else {}

    def _qsize(self):
        return len(self.queue)

    def _put(self, item):
        if self.key is None:
            self.queue.add(item)
        else:
            self.queue[self.key(item)] = item

    def clear(self):
        """Remove all elements from the queue"""
        with self.mutex:
            self.queue.clear()
            self.not_full.notify_all()

    def _get(self):
        try:
            if self.key is None:
                return self.queue.pop()
            else:
                k = next(iter(self.queue))
                return self.queue.pop(k)
        except (KeyError, StopIteration):
            raise IndexError

    def __iter__(self) -> Iterable[T]:
        """Iterate over the elements of the queue.

        This will return as soon as there are no more values. It may thus be desirable to use it in a loop, as producer
        threads can actively be adding content that will be missed otherwise.
        """
        while self.queue:
            yield self.get()

    def wait_for_value(self):
        """Block until the set has at least one element."""
        # We don't need to lock because we're only going to support *one* consumer at once
        with self.not_empty:
            self.not_empty.wait_for(lambda: bool(self.queue))
