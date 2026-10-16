from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def levelOrder(root: TreeNode):
    # 1. If root is None, return empty list []
    if not root:
        return []

    # 2. Initialize result list and a queue with [root]
    result = []
    queue = deque([root])

    # 3. While queue is not empty:
    while queue:
        # a. Get count of nodes in current level: level_size = len(queue)
        level_size = len(queue)
        # b. Initialize empty list 'current_level'
        current_level = []

        # c. Loop level_size times:
        for _ in range(level_size):
            # - Pop front node from queue
            node = queue.popleft()
            # - Append node.val to current_level
            current_level.append(node.val)
            # - If node.left exists, push to queue
            if node.left:
                queue.append(node.left)
            # - If node.right exists, push to queue
            if node.right:
                queue.append(node.right)

        # d. Append current_level to result
        result.append(current_level)

    # 4. Return result
    return result

# --- Test Cases ---
# Test 1: Example tree [3, 9, 20, null, null, 15, 7]
root1 = TreeNode(3)
root1.left = TreeNode(9)
root1.right = TreeNode(20, TreeNode(15), TreeNode(7))

# Test 2: Empty tree
root2 = None

# Test 3: One-sided left tree 1 -> 2 -> 3
root3 = TreeNode(1, TreeNode(2, TreeNode(3)))

print("Test 1 (Example):  ", levelOrder(root1))
print("Test 2 (Empty):    ", levelOrder(root2))
print("Test 3 (One-sided):", levelOrder(root3))