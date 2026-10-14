class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def build_list(arr):
    dummy = curr = ListNode(0)
    for x in arr:
        curr.next = ListNode(x)
        curr = curr.next
    return dummy.next

def to_list(head):
    res, curr = [], head
    while curr:
        res.append(curr.val)
        curr = curr.next
    return res

def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n + 1):
        fast = fast.next
    while fast:
        fast, slow = fast.next, slow.next
    slow.next = slow.next.next
    return dummy.next

cases = [([1, 2, 3, 4, 5], 2, [1, 2, 3, 5]), ([1], 1, []), ([1, 2], 1, [1]), ([1, 2], 2, [2])]
for arr, n, exp in cases:
    res = to_list(removeNthFromEnd(build_list(arr), n))
    print(f"{arr}, n={n} -> {res} | {'PASS' if res == exp else 'FAIL'}")