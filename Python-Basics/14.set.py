
# set is unordeed collection of i=unique elements

numbers = {1,2,3,4,2,3,1,4,3,2,1,4}
type(numbers)
numbers

a=set()
a.add(1)
a

b = {}
b
type(b)
# b.add(1)
b = {'something'}
type(b)
b

# Python 3.13.6 (tags/v3.13.6:4e66535, Aug  6 2025, 14:36:00) [MSC v.1944 64 bit (AMD64)] on win32
# Type "help", "copyright", "credits" or "license" for more information.
# Ctrl click to launch VS Code Native REPL
# >>> numbers = {1,2,3,4,2,3,1,4,3,2,1,4}
# >>> type(numbers)
# <class 'set'>
# >>> numbers
# {1, 2, 3, 4}
# >>> a=set()
# >>> a.add(2)
# >>> a.add(3)
# >>> a.add(4)
# >>> a.add(1)
# >>> a.add(1)
# >>> a.add(1)
# >>> a
# {1, 2, 3, 4}
# >>> 
# >>> b
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#     import platform
#     ^
# NameError: name 'b' is not defined
# >>>
# >>> b = {}
# >>> b
# {}
# >>>
# >>> type(b)
# <class 'dict'>
# >>> b.add(1)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#     import platform
#     ^^^^^
# AttributeError: 'dict' object has no attribute 'add'
# >>> type(b)
# <class 'dict'>
# >>> b
# {}
# >>>
# >>> b.add(1)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#     import platform
#     ^^^^^
# AttributeError: 'dict' object has no attribute 'add'
# >>> b = {'something'}
# >>> type(b)
# <class 'set'>
# >>> b
# {'something'}
# >>>
# >>> b
# {'something'}
# >>>
# >>>







