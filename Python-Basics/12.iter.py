
# a = ["hey","bro","you'r","awesome"]
# for i in a:
#     print(i)

# dir(a)

# itr = iter(a)

# next(itr)
# next(itr)
# next(itr)
# next(itr)
# next(itr)

# dir(itr)

# itr = reversed(a)
# next(itr)

# next(itr)

# Python 3.13.6 (tags/v3.13.6:4e66535, Aug  6 2025, 14:36:00) [MSC v.1944 64 bit (AMD64)] on win32       
# Type "help", "copyright", "credits" or "license" for more information.
# Ctrl click to launch VS Code Native REPL
# >>> a = ["hey","bro","you'r","awesome"]
# >>> for i in a:
# ...     print(i)
# ...
# hey
# bro
# you'r
# awesome
# >>> dir(a)
# ['__add__', '__class__', '__class_getitem__', '__contains__', '__delattr__', '__delitem__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getstate__', '__gt__', '__hash__', '__iadd__', '__imul__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'append', 'clear', 'copy', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort']
# >>> itr = iter(a)
# >>> next(itr)
# 'hey'
# >>> next(itr)
# 'bro'
# >>> next(itr)
# "you'r"
# >>> next(itr)
# 'awesome'
# >>> next(itr)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#     import platform
#     ^^^^^^^^^
# StopIteration
# >>> dir(itr)
# ['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__length_hint__', '__lt__', '__ne__', '__new__', '__next__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__']
# >>> itr = reversed(a)
# >>> next(itr)
# 'awesome'
# >>> next(itr)
# "you'r"
# >>>


class RemoteControl():
    def __init__(self):
        self.channels = ["HBO","cnn","abc","espn"]
        self.index = -1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.index += 1
        if self.index  == len(self.channels):
            raise StopIteration
        
        return self.channels[self.index]
    


r = RemoteControl()
itr = iter(r)

print(next(itr))
print(next(itr))
print(next(itr))
print(next(itr))







