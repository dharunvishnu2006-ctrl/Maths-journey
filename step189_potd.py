from collections import defaultdict

# Pattern: Hash Map with Canonical Key
# 1. Map canonical signature -> list of words.
# 2. For each word, sort its letters to make the key (e.g. "eat" -> "aet").
#    Use "".join(...) to turn the sorted list into an immutable string key.
# 3. Append word to map[key].
# 4. Return list of map values.


def groupAnagrams(strs):
  groups = defaultdict(list)
  for s in strs:
    key = "".join(sorted(s))
    groups[key].append(s)
  return list(groups.values())


# Tests: edge cases first, then main example
tests = [[""], ["a"], ["ab", "ab"], ["eat", "tea", "tan", "ate", "nat", "bat"]]

for t in tests:
  print(groupAnagrams(t))