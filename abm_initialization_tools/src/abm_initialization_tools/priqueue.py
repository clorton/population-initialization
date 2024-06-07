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
    """Push an item with a priority into the priority queue."""
    if pq.size >= len(pq.payloads):
        raise IndexError("Priority queue is full")
    pq.payloads[pq.size] = payload
    pq.priority[pq.size] = priority
    pq.size += 1
    index = pq.size - 1
    while index > 0:
        parent_index = (index - 1) // 2
        if pq.priority[index] < pq.priority[parent_index]:
            pq.payloads[index], pq.payloads[parent_index] = pq.payloads[parent_index], pq.payloads[index]
            pq.priority[index], pq.priority[parent_index] = pq.priority[parent_index], pq.priority[index]
            index = parent_index
        else:
            break

    return


def _pq_pop(pq: PriorityQueue) -> Tuple[Any, np.uint32]:
    """Remove the item with the highest priority from the priority queue."""
    if pq.size == 0:
        raise IndexError("Priority queue is empty")

    payload, priority = pq.peek()

    pq.payloads[0] = pq.payloads[pq.size - 1]
    pq.priority[0] = pq.priority[pq.size - 1]
    pq.size -= 1

    index = 0
    while index < pq.size:
        left_child_index = 2 * index + 1
        right_child_index = 2 * index + 2
        smallest = index

        if left_child_index < pq.size and pq.priority[left_child_index] < pq.priority[smallest]:
            smallest = left_child_index

        if right_child_index < pq.size and pq.priority[right_child_index] < pq.priority[smallest]:
            smallest = right_child_index

        if smallest != index:
            pq.payloads[index], pq.payloads[smallest] = pq.payloads[smallest], pq.payloads[index]
            pq.priority[index], pq.priority[smallest] = pq.priority[smallest], pq.priority[index]
            index = smallest
        else:
            break

    return payload, priority
