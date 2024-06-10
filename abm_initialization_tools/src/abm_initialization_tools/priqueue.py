"""Priority queue implemented using a heap."""

from typing import Any, Tuple

import numba as nb
import numpy as np


class PriorityQueue:
    """A priority queue implemented using a heap."""

    def __init__(self, capacity, dtype=np.uint32):
        self.payloads = np.zeros(capacity, dtype=dtype)
        self.priority = np.zeros(capacity, dtype=np.uint32)
        self.capacity = capacity
        self.size = 0

    def push(self, payload, priority):
        # _pq_push(self, payload, priority)
        if self.size >= self.capacity:
            raise IndexError("Priority queue is full")
        _pq_push_nb(self.payloads, self.priority, self.size, payload, priority)
        self.size += 1
        return

    def peek(self) -> Tuple[Any, np.uint32]:
        if self.size == 0:
            raise IndexError("Priority queue is empty")
        return (self.payloads[0], self.priority[0])

    def pop(self) -> Tuple[Any, np.uint32]:
        ret = self.peek()
        _pq_pop_nb(self.payloads, self.priority, self.size)
        self.size -= 1
        return ret

    def __len__(self):
        return self.size


@nb.njit(
    (nb.uint32[:], nb.uint32[:], nb.uint32, nb.uint32, nb.uint32),
    nogil=True,
    cache=True,
)  # parallel n/a
def _pq_push_nb(
    payloads: np.ndarray,
    priorities: np.ndarray,
    size: np.uint32,
    payload: np.uint32,
    priority: np.uint32,
) -> None:
    """Push an item onto the priority queue."""
    if size >= len(payloads):
        raise IndexError("Priority queue is full")

    payloads[size] = payload
    priorities[size] = priority

    index = size
    while index > 0:
        parent_index = (index - 1) // 2
        if priorities[index] < priorities[parent_index]:
            payloads[index], payloads[parent_index] = (
                payloads[parent_index],
                payloads[index],
            )
            priorities[index], priorities[parent_index] = (
                priorities[parent_index],
                priorities[index],
            )
            index = parent_index
        else:
            break

    return


def _pq_push(pq: PriorityQueue, payload: Any, priority: np.uint32):
    """Push an item onto the priority queue."""
    if (index := pq.size) >= len(pq.payloads):
        raise IndexError("Priority queue is full")

    pays = pq.payloads
    pris = pq.priority

    pays[index] = payload
    pris[index] = priority

    while index > 0:
        parent_index = (index - 1) // 2
        if pris[index] < pris[parent_index]:
            pays[index], pays[parent_index] = pays[parent_index], pays[index]
            pris[index], pris[parent_index] = pris[parent_index], pris[index]
            index = parent_index
        else:
            break

    pq.size += 1

    return


@nb.njit(
    (nb.uint32[:], nb.uint32[:], nb.uint32), nogil=True, cache=True
)  # parallel n/a
def _pq_pop_nb(
    payloads: np.ndarray,
    priorities: np.ndarray,
    size: np.uint32,
) -> None:
    """Remove the item with the highest priority from the priority queue."""

    size -= 1
    payloads[0] = payloads[size]
    priorities[0] = priorities[size]

    index = 0
    while index < size:
        left_child_index = 2 * index + 1
        right_child_index = 2 * index + 2
        smallest = index

        if (
            left_child_index < size
            and priorities[left_child_index] < priorities[smallest]
        ):
            smallest = left_child_index

        if (
            right_child_index < size
            and priorities[right_child_index] < priorities[smallest]
        ):
            smallest = right_child_index

        if smallest != index:
            payloads[index], payloads[smallest] = payloads[smallest], payloads[index]
            priorities[index], priorities[smallest] = (
                priorities[smallest],
                priorities[index],
            )
            index = smallest
        else:
            break

    return


def _pq_pop(pq: PriorityQueue) -> Tuple[Any, np.uint32]:
    """Remove the item with the highest priority from the priority queue."""
    if pq.size == 0:
        raise IndexError("Priority queue is empty")

    payload, priority = pq.peek()

    pq.size -= 1
    size = pq.size
    pq.payloads[0] = pq.payloads[size]
    pq.priority[0] = pq.priority[size]

    pays = pq.payloads
    pris = pq.priority

    index = 0
    while index < size:
        left_child_index = 2 * index + 1
        right_child_index = 2 * index + 2
        smallest = index

        if left_child_index < size and pris[left_child_index] < pris[smallest]:
            smallest = left_child_index

        if right_child_index < size and pris[right_child_index] < pris[smallest]:
            smallest = right_child_index

        if smallest != index:
            pays[index], pays[smallest] = pays[smallest], pays[index]
            pris[index], pris[smallest] = pris[smallest], pris[index]
            index = smallest
        else:
            break

    return payload, priority
