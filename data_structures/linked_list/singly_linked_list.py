from __future__ import annotations
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar, overload

T = TypeVar("T")


@dataclass(slots=True)
class Node(Generic[T]):
    """
    A node in a singly linked list.

    >>> Node(20)
    Node(20)
    >>> Node("Hello, world!")
    Node(Hello, world!)
    >>> Node(None)
    Node(None)
    >>> Node(True)
    Node(True)
    """

    data: T
    next_node: Optional[Node[T]] = None

    def __repr__(self) -> str:
        """
        Get the string representation of this node.

        >>> Node(10).__repr__()
        'Node(10)'
        >>> repr(Node(10))
        'Node(10)'
        >>> str(Node(10))
        'Node(10)'
        >>> Node(10)
        Node(10)
        """
        return f"Node({self.data})"


class LinkedList(Generic[T]):
    """
    Singly linked list with efficient length tracking and robust error handling.

    Supports iteration, indexing (read/write), insertion at head/tail, deletion at
    head/tail, search, reverse, and string representation.

    - O(1) len() via cached length
    - O(1) tail append via cached tail pointer
    - Type-safe via generics
    - Clear exceptions with messages
    """

    __slots__ = ("_head", "_tail", "_len")

    def __init__(self, iterable: Optional[Iterable[T]] = None) -> None:
        self._head: Optional[Node[T]] = None
        self._tail: Optional[Node[T]] = None
        self._len: int = 0
        if iterable is not None:
            for item in iterable:
                self.insert_tail(item)

    # ---------------------------- basic protocol ----------------------------
    def __len__(self) -> int:  # O(1)
        return self._len

    def __iter__(self) -> Iterator[T]:
        current = self._head
        while current is not None:
            yield current.data
            current = current.next_node

    def __repr__(self) -> str:
        return f"LinkedList([{', '.join(repr(x) for x in self)}])"

    def __str__(self) -> str:
        return " -\u003e ".join(
            (repr(x) if not isinstance(x, Node) else str(x)) for x in self
        )

    # ----------------------------- node helpers -----------------------------
    def _node_at(self, index: int) -> Node[T]:
        """Return node at index or raise IndexError with clear message.

        Optimized by scanning from head only (singly list), but validates bounds
        up-front using cached length.
        """
        if index < 0 or index >= self._len:
            raise IndexError(f"index out of range: {index}")
        cur = self._head
        for _ in range(index):
            assert cur is not None  # for type checkers
            cur = cur.next_node
        assert cur is not None
        return cur

    # ------------------------------- indexing -------------------------------
    @overload
    def __getitem__(self, index: int) -> T: ...

    def __getitem__(self, index: int) -> T:
        return self._node_at(index).data

    def __setitem__(self, index: int, value: T) -> None:
        self._node_at(index).data = value

    # ------------------------------- mutation -------------------------------
    def clear(self) -> None:
        self._head = None
        self._tail = None
        self._len = 0

    def insert_head(self, value: T) -> None:
        node: Node[T] = Node(value, self._head)  # type: ignore[arg-type]
        # dataclass with slots won't accept positional for next_node unless declared,
        # so assign explicitly:
        node = Node(value)
        node.next_node = self._head
        self._head = node
        if self._tail is None:
            self._tail = node
        self._len += 1

    def insert_tail(self, value: T) -> None:  # O(1)
        node = Node(value)
        if self._tail is None:
            self._head = self._tail = node
        else:
            self._tail.next_node = node
            self._tail = node
        self._len += 1

    def delete_head(self) -> T:
        if self._head is None:
            raise IndexError("delete_head from empty LinkedList")
        node = self._head
        self._head = node.next_node
        if self._head is None:
            self._tail = None
        self._len -= 1
        return node.data

    def delete_tail(self) -> T:
        if self._head is None:
            raise IndexError("delete_tail from empty LinkedList")
        if self._head.next_node is None:
            # single element
            data = self._head.data
            self._head = self._tail = None
            self._len = 0
            return data
        # find penultimate
        prev = self._head
        cur = self._head.next_node
        while cur and cur.next_node is not None:
            prev = cur
            cur = cur.next_node
        assert cur is not None
        data = cur.data
        prev.next_node = None
        self._tail = prev
        self._len -= 1
        return data

    # -------------------------------- search --------------------------------
    def find(self, value: T) -> int:
        """Return first index of value or -1 if not found.

        Early-terminates naturally; stable O(n).
        """
        idx = 0
        cur = self._head
        while cur is not None:
            if cur.data == value:
                return idx
            cur = cur.next_node
            idx += 1
        return -1

    def contains(self, value: T) -> bool:
        return self.find(value) != -1

    # -------------------------------- other ---------------------------------
    def reverse(self) -> None:
        prev: Optional[Node[T]] = None
        cur = self._head
        self._tail = self._head
        while cur is not None:
            nxt = cur.next_node
            cur.next_node = prev
            prev = cur
            cur = nxt
        self._head = prev

    def print_list(self) -> None:
        # Maintain existing behavior for scripts; avoid printing Node objects oddly
        print(self)


# ------------------------------- unit tests ---------------------------------
import unittest


class TestLinkedList(unittest.TestCase):
    def test_basic_insert_and_len(self) -> None:
        ll: LinkedList[int] = LinkedList()
        self.assertEqual(len(ll), 0)
        ll.insert_head(1)
        ll.insert_tail(2)
        ll.insert_tail(3)
        self.assertEqual(len(ll), 3)
        self.assertEqual(list(ll), [1, 2, 3])

    def test_index_get_set(self) -> None:
        ll = LinkedList([10, 20, 30])
        self.assertEqual(ll[0], 10)
        self.assertEqual(ll[2], 30)
        ll[1] = 99
        self.assertEqual(list(ll), [10, 99, 30])
        with self.assertRaises(IndexError):
            _ = ll[3]
        with self.assertRaises(IndexError):
            ll[-1] = 1

    def test_delete_head_tail(self) -> None:
        ll = LinkedList([1, 2, 3])
        self.assertEqual(ll.delete_head(), 1)
        self.assertEqual(ll.delete_tail(), 3)
        self.assertEqual(list(ll), [2])
        self.assertEqual(ll.delete_tail(), 2)
        with self.assertRaises(IndexError):
            ll.delete_head()

    def test_find_contains(self) -> None:
        ll = LinkedList(["a", "b", "c", "b"])
        self.assertEqual(ll.find("a"), 0)
        self.assertEqual(ll.find("b"), 1)
        self.assertEqual(ll.find("z"), -1)
        self.assertTrue(ll.contains("c"))
        self.assertFalse(ll.contains("x"))

    def test_reverse(self) -> None:
        ll = LinkedList([1, 2, 3, 4])
        ll.reverse()
        self.assertEqual(list(ll), [4, 3, 2, 1])
        self.assertEqual(len(ll), 4)
        # reverse back
        ll.reverse()
        self.assertEqual(list(ll), [1, 2, 3, 4])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
