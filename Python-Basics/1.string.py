text = "ice cream"
print(text)

# text[0] = 'g'
# >>> text[0] = 'g'
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#     import platform
#     ^^^^^^^
# TypeError: 'str' object does not support item assignment

text[0:3]
text[0:]
text[4:]

text[:3]

text = "hello"
text

address = "i purple" \
""

address = "i purplr \nnew this"
address

text + address

num = 25

text + num

num = str(num)


text + num

# Python 3.13.6 (tags/v3.13.6:4e66535, Aug  6 2025, 14:36:00) [MSC v.1944 64 bit (AMD64)] on win32
# Type "help", "copyright", "credits" or "license" for more information.
# Ctrl click to launch VS Code Native REPL
# >>> text = "ice cream"
# >>> print(text)
# ice cream
# >>> text[0] = 'g'
# ice cream
# ice cream
# >>> text[0] = 'g'
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#     import platform
#     ^^^^^^^
# TypeError: 'str' object does not support item assignment
# >>> text[0:3]
# 'ice'
# >>> text[0:]
# 'ice cream'
# >>> text[4:]
# 'cream'
# >>> text[:3]
# 'ice'
# >>> text = "hello"
# >>> text
# 'hello'
# >>> address = "i purple" \
# ... ""
# >>> 
# >>> address = 'i purplr \this'
# >>> address
# 'i purplr \this'
# >>> address = "i purplr \nnew this"
# >>> address
# 'i purplr \nnew this'
# >>> text + address
# 'helloi purplr \nnew this'
# >>> text + num
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#     import platform
#            ^^^
# NameError: name 'num' is not defined. Did you mean: 'sum'?
# >>> str(num)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#     import platform
#         ^^^
# NameError: name 'num' is not defined. Did you mean: 'sum'?
# >>> num = 25
# >>> str(num)
# '25'
# >>> text + num
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#     import platform
#     ^^^^^^^^^^
# TypeError: can only concatenate str (not "int") to str
# >>> text = "hello"
# >>> text + num
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#     import platform
#     ^^^^^^^^^^
# TypeError: can only concatenate str (not "int") to str
# >>> text = "hello"
# >>> num = 25
# >>> text + num
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#     import platform
#     ^^^^^^^^^^
# TypeError: can only concatenate str (not "int") to str
# >>> text = "hello"
# >>> num = 25
# >>> str(num)
# '25'
# >>> text + num
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#     import platform
#     ^^^^^^^^^^
# TypeError: can only concatenate str (not "int") to str
# >>> num = str(num)
# >>> text + num
# 'hello25'