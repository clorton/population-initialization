"""Priority queue implemented using a heap."""

from typing import Any, Tuple

import numpy as np

class PriorityQueue:
    """A priority queue implemented using a heap."""

    def __init__(self, capacity, dtype=np.uint32):
        self.payloads = np.zeros(capacity, dtype=dtype)
        self.priority = np.zeros(capacity, dtype=np.uint32)
        self.size = 0

    def push(self, payload, priority):
        _pq_push(self, payload, priority)

    def peek(self) -> Tuple[Any, np.uint32]:
        if self.size == 0:
            raise IndexError("Priority queue is empty")
        return (self.payloads[0], self.priority[0])

    def pop(self) -> Tuple[Any, np.uint32]:
        return _pq_pop(self)

    def __len__(self):
        return self.size


def _pq_push(pq: PriorityQueue, payload: Any, priority: np.uint32):
    """Push an item onto the priority queue."""
    if (index := pq.size) >= len(pq.payloads):
        raise IndexError("Priority queue is full")
    pq.payloads[index] = payload
    pq.priority[index] = priority

    pays = pq.payloads
    pris = pq.priority

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
