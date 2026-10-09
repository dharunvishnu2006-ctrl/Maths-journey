def length_of_longest_substring(s: str) -> int:
    # 1. Init: last_seen = {}, left = 0, max_len = 0
    last_seen, left, max_len = {}, 0, 0
    # 2. Iterate right, char across s
    for right, char in enumerate(s):
        # 3. If char in last_seen and last_seen[char] >= left: jump left
        if char in last_seen and last_seen[char] >= left:
            left = last_seen[char] + 1
        # 4. last_seen[char] = right
        last_seen[char] = right
        # 5. max_len = max(max_len, right - left + 1)
        max_len = max(max_len, right - left + 1)
    # 6. Return max_len
    return max_len

tests = ["", "a", "bbbbb", "abcdef", " ", "abcabcbb", "b", "pwwkew"]
print([length_of_longest_substring(t) for t in tests])