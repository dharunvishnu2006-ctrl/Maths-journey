class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# UNDERSTAND
# Input: head of a singly linked list, integer n (1 <= n <= length of list).
# Output: head of the linked list with the n-th node from the end removed.
# Edge cases:
#   - Single-node list ([1], n = 1) -> returns None ([]).
#   - Removing the head itself ([1, 2], n = 2) -> returns second node ([2]).
#   - Removing the tail node ([1, 2], n = 1) -> returns [1].
# One-pass approach: use a dummy node before head and two pointers (fast, slow) separated by n + 1 steps so slow stops right before the target node.

# PLAN
# 1. Create a dummy node pointing to head; set fast = dummy and slow = dummy.
# 2. Advance fast by n + 1 steps so the gap between fast and slow is n + 1 nodes.
# 3. Advance both fast and slow one step at a time until fast reaches None.
# 4. Bypass the target node by setting slow.next = slow.next.next.
# 5. Return dummy.next as the new head.


def remove_nth_from_end(head, n):
    # 1. Create a dummy node pointing to head; set fast = dummy and slow = dummy.
    dummy = ListNode(0, head)
    fast = dummy
    slow = dummy

    # 2. Advance fast by n + 1 steps so the gap between fast and slow is n + 1 nodes.
    for _ in range(n + 1):
        fast = fast.next

    # 3. Advance both fast and slow one step at a time until fast reaches None.
    while fast is not None:
        fast = fast.next
        slow = slow.next

    # 4. Bypass the target node by setting slow.next = slow.next.next.
    slow.next = slow.next.next

    # 5. Return dummy.next as the new head.
    return dummy.next


def build_list(values):
    dummy = ListNode(0)
    curr = dummy
    for v in values:
        curr.next = ListNode(v)
        curr = curr.next
    return dummy.next


def to_list(head):
    result = []
    curr = head
    while curr is not None:
        result.append(curr.val)
        curr = curr.next
    return result


if __name__ == "__main__":
    # Edge cases first
    # 1. Single node, n = 1 -> []
    res1 = to_list(remove_nth_from_end(build_list([1]), 1))
    assert res1 == []
    print("Edge 1 ([1], n=1):", res1)

    # 2. Remove head node ([1, 2], n = 2 -> [2])
    res2 = to_list(remove_nth_from_end(build_list([1, 2]), 2))
    assert res2 == [2]
    print("Edge 2 ([1, 2], n=2):", res2)

    # 3. Remove tail node ([1, 2], n = 1 -> [1])
    res3 = to_list(remove_nth_from_end(build_list([1, 2]), 1))
    assert res3 == [1]
    print("Edge 3 ([1, 2], n=1):", res3)

    # Standard case
    # 4. Middle node ([1, 2, 3, 4, 5], n = 2 -> [1, 2, 3, 5])
    res4 = to_list(remove_nth_from_end(build_list([1, 2, 3, 4, 5]), 2))
    assert res4 == [1, 2, 3, 5]
    print("Standard ([1, 2, 3, 4, 5], n=2):", res4)