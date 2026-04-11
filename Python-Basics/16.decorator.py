
import time

def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__ + " " +  str((end - start)*1000) + " mili sec")
        return result
    
    return wrapper

@time_it
def square(numbers):
    result = []
    for number in numbers:
        result.append(number*number)
    return result

@time_it
def cube(numbers):
    result = []
    for number in numbers:
        result.append(number*number*number)
    return result


array = range(1,1000000)
out_square = square(array)
out_cube = cube(array)













































# import time


# def square(numbers):
#     start = time.time()
#     result = []
#     for number in numbers:
#         result.append(number*number)

#     end = time.time()

#     print("calculate square took "+ str((end - start)* 1000) + " mili sec")
#     return result


# def cube(numbers):
#     start = time.time()
#     result = []
#     for number in numbers:
#         result.append(number*number*number)
#     end = time.time()

#     print("calculate cube took "+ str((end - start)* 1000) + " mili sec")
#     return result

# array = range(1,1000000)
# out_square = square(array)
# out_cube = cube(array)
