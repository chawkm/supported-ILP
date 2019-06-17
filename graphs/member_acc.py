from itertools import permutations, chain

list_length = 3
nums = [i for i in range (list_length)]

lists = [tuple(x) for x in chain(*[permutations(nums, i) for i in range(list_length + 1)])]

errors = 0.0
for a in lists:
    for b in nums:
        if b in a:
            # member
            if not((len(a) > 0 and b == a[0]) or not b in a):
                errors += 1.0
        else:
            # not member
            if (len(a) > 0 and b == a[0]) or not b in a:
                errors += 1.0


print(errors)
print(len(lists), lists)
print(errors / (len(lists) * len(nums)))