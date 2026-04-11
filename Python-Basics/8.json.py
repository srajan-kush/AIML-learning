book = {}
book['tom'] = {
    'name':'tom',
    'address':'TKG',
    'phone':2432545
}

book['bob'] = {
    'name':'bob',
    'address':'BOB',
    'phone':134234
}

import json
s = json.dumps(book)
# print(s)

with open("Python-Basics/book.txt","w") as f:
    f.write(s)


f = open("Python-Basics/book.txt","r")
s = f.read()
s

import json
book = json.loads(s)
book
type(book)

book['bob']
book['bob']['phone']

for person in book:
    print(book[person])


# >>>
# >>> import json
# >>> book = json.loads(s)
# >>> book
# {'tom': {'name': 'tom', 'address': 'TKG', 'phone': 2432545}, 'bob': {'name': 'bob', 'address': 'BOB', 'phone': 134234}}
# >>> type(book)
# <class 'dict'>
# >>> book['bob']
# {'name': 'bob', 'address': 'BOB', 'phone': 134234}
# >>> book['bob']['phone']
# 134234
# >>> for person in book:
# ...     print(book[person])
# ...
# {'name': 'tom', 'address': 'TKG', 'phone': 2432545}
# {'name': 'bob', 'address': 'BOB', 'phone': 134234}
# >>>



