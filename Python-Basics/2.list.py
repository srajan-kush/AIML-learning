# items = ['1',"name",{1,2,3}]
# items

# items[0]
# items[:1]
# items[2:]
# items.append("water")
# items[3]
# items.insert(2,"srajan")
# items

# food = ["pasta","banana"]

# items = items + food
# items

# food + "soda"

# len(items)
# "pasta" in items
# "soda" in items

# Python 3.13.6 (tags/v3.13.6:4e66535, Aug  6 2025, 14:36:00) [MSC v.1944 64 bit (AMD64)] on win32
# Type "help", "copyright", "credits" or "license" for more information.
# Ctrl click to launch VS Code Native REPL
# >>> items = ['1',"name",{1,2,3}]
# >>> items
# ['1', 'name', {1, 2, 3}]
# >>> items[0]
# '1'
# >>> items[1]
# 'name'
# >>> items[1:]
# ['name', {1, 2, 3}]
# >>> items[:1]
# ['1']
# >>> items[2:]
# [{1, 2, 3}]
# >>> items.append("water")
# >>> items
# ['1', 'name', {1, 2, 3}, 'water']
# >>> items[3]
# 'water'
# >>> items.insert(2,"srajan")
# >>> items
# ['1', 'name', 'srajan', {1, 2, 3}, 'water']
# >>> items + food
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#     import platform
#             ^^^^
# NameError: name 'food' is not defined
# >>> food = ["pasta","banana"]
# >>> items + food
# ['1', 'name', 'srajan', {1, 2, 3}, 'water', 'pasta', 'banana']
# >>> items = items + food
# >>> items
# ['1', 'name', 'srajan', {1, 2, 3}, 'water', 'pasta', 'banana']
# >>> food + "soda"
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#     import platform
#     ^^^^^^^^^^^^^
# TypeError: can only concatenate list (not "str") to list
# >>> len(items)
# 7
# >>> "pasta" in items
# True
# >>> "soda" in items
# False
# >>>









numbers = [1,2,3,4,5,6,7]
even = []
for i in numbers:
    if i % 2 == 0:
        even.append(i)

even

even = [i for i in numbers if i%2 == 0]
even

squr = [i * i for i in numbers]
squr


s = set([1,2,3,4,4,3,2,2,3,4,4])
s

even = {i for i in numbers if i%2==0}
even


cities=["mumbai","new york","peris"]
countries=["india","usa","france"]
z = zip(cities,countries)

for a in z:
    print(a)

d = {city:country for city, country in zip(cities,countries)}
d








