class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# UNDERSTAND
# - Input: root of a binary tree (TreeNode or None).
# - Output: bool — True if valid BST, False otherwise.
# - BST invariant: for every node, all descendants in left subtree must be strictly < node.val,
#   and all descendants in right subtree must be strictly > node.val.
# - Trap: checking only immediate children (left.val < node.val < right.val) fails on [5, 4, 6, None, None, 3, 7],
#   where 3 < 6 locally, but 3 violates ancestor 5's lower bound (> 5).
# - Edge cases: empty tree (None -> True), single node (-> True), duplicate values (e.g. [2, 2, 2] -> False).

# PLAN
# 1. Define a recursive helper validate(node, low, high) tracking exclusive bounds (low, high) initialized to (-inf, +inf).
# 2. Base case: if node is None, return True.
# 3. Check if low < node.val < high; if not, return False immediately.
# 4. Recursively validate left subtree with updated upper bound: validate(node.left, low, node.val).
# 5. Recursively validate right subtree with updated lower bound: validate(node.right, node.val, high).
# 6. Return True only if both left and right recursive calls return True.
# 7. Write isolated test functions (edge cases first) building trees manually with TreeNode(...) and run them.


def is_valid_bst(root: TreeNode | None) -> bool:
    # 1. Define a recursive helper validate(node, low, high) tracking exclusive bounds (low, high).
    def validate(node: TreeNode | None, low: float, high: float) -> bool:
        # 2. Base case: if node is None, return True.
        if node is None:
            return True

        # 3. Check if low < node.val < high; if not, return False immediately.
        if not (low < node.val < high):
            return False

        # 4 & 5. Recursively validate left (upper bound = node.val) and right (lower bound = node.val).
        # 6. Return True only if both subtrees are valid.
        return validate(node.left, low, node.val) and validate(
            node.right, node.val, high
        )

    return validate(root, float("-inf"), float("inf"))


# 7. Isolated test functions (edge cases first)
def test_empty_tree():
    assert is_valid_bst(None) is True


def test_single_node():
    root = TreeNode(42)
    assert is_valid_bst(root) is True


def test_duplicate_values():
    # [2, 2, 2] -> strict inequality required, so False
    root = TreeNode(2, TreeNode(2), TreeNode(2))
    assert is_valid_bst(root) is False


def test_negative_values():
    # [-10, -20, -5] -> True
    root = TreeNode(-10, TreeNode(-20), TreeNode(-5))
    assert is_valid_bst(root) is True


def test_example_1_valid():
    # [2, 1, 3] -> True
    root = TreeNode(2, TreeNode(1), TreeNode(3))
    assert is_valid_bst(root) is True


def test_example_2_invalid():
    # [5, 1, 4, None, None, 3, 6] -> False
    root = TreeNode(5, TreeNode(1), TreeNode(4, TreeNode(3), TreeNode(6)))
    assert is_valid_bst(root) is False


def test_example_3_subtree_trap():
    # [5, 4, 6, None, None, 3, 7] -> False (3 is in 5's right subtree)
    root = TreeNode(5, TreeNode(4), TreeNode(6, TreeNode(3), TreeNode(7)))
    assert is_valid_bst(root) is False


if __name__ == "__main__":
    test_empty_tree()
    test_single_node()
    test_duplicate_values()
    test_negative_values()
    test_example_1_valid()
    test_example_2_invalid()
    test_example_3_subtree_trap()
    print("All 7 tests passed!")