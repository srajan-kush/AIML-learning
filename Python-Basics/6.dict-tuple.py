
d = {"tom":685848848, "rob":45436546,"joe":424235}
d
d["joe"]

d["sam"] = 534567457
d
del d["joe"]

for key in d:
    print("key:",key,"value:",d[key])


for k,v in d.items():
    print("key:",k,"value:",v)


"tom" in d

"sameer" in d

d.clear()
d


point = (5,9)
point[0]
point[1]

point[0] = 4 #not possible


# Python 3.13.6 (tags/v3.13.6:4e66535, Aug  6 2025, 14:36:00) [MSC v.1944 64 bit (AMD64)] on win32
# Type "help", "copyright", "credits" or "license" for more information.
# Ctrl click to launch VS Code Native REPL
# >>> d
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#     import platform
#     ^
# NameError: name 'd' is not defined. Did you mean: 'id'?
# >>> d = {"tom":685848848, "rob":45436546,"joe":424235}
# >>> d
# {'tom': 685848848, 'rob': 45436546, 'joe': 424235}
# >>> d
# {'tom': 685848848, 'rob': 45436546, 'joe': 424235}
# >>> d
# {'tom': 685848848, 'rob': 45436546, 'joe': 424235}
# >>> 
# >>> d
# {'tom': 685848848, 'rob': 45436546, 'joe': 424235}
# >>> d["joe"]
# 424235
# >>> d["sam"] = 534567457
# >>> d
# {'tom': 685848848, 'rob': 45436546, 'joe': 424235, 'sam': 534567457}
# >>> del d["joe"]
# >>> print("key:",key,"value:",d[key])
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#     import platform
#                  ^^
# NameError: name 'key' is not defined
# >>> for key in d:
# ...     print("key:",key,"value:",d[key])
# ...
# key: tom value: 685848848
# key: rob value: 45436546
# key: sam value: 534567457
# >>> for k,v in d.items():
# ...     print("key:",k,"value:",v)
# ...
# key: tom value: 685848848
# key: rob value: 45436546
# key: sam value: 534567457
# >>> "tom" in d
# True
# >>> "sameer" in d
# False
# >>> d.clear()
# >>> d
# {}
# >>> point = (5,9)
# >>> point[0]
# 5
# >>> point[1]
# 9
# >>> point[0] = 4
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#     import platform
#     ^^^^^^^^
# TypeError: 'tuple' object does not support item assignment
# >>>