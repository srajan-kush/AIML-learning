# 1/0
# 'abc' + 2

# x = input("Enter number1: ")
# y = input("Enter number2: ")

# try:
#     z = x / int(y)
# except ZeroDivisionError as e:
#     print('ZeroDivisionError')
#     z = None
# except TypeError as e:
#     # print('Exception Type ',type(e).__name__)
#     print('TypeError')
#     z = None


# print("Division is: ",z)


# try:
#     raise MemoryError('memory error')
# except MemoryError as e:
#     print(e)



class Accident(Exception):
    def __init__(self,msg):
        self.msg = msg


    def print_exception(self):
        print("User Defined exception: ",self.msg)

    def handle(self):
        print("accident occured. take detour")


try:
    raise Accident('crash between two cars')
except Accident as e:
    e.handle()

