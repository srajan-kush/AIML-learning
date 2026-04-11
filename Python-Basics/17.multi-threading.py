
import time
import threading

def square(numbers):
    print("Calculating squares of numbes: ")
    for number in numbers:
        time.sleep(0.2)
        print("Square: ",number*number)


def cube(numbers):
    print("Calculating Cubes of numbes: ")
    for number in numbers:
        time.sleep(0.2)
        print("Cube: ",number*number*number)


t = time.time()
array = [2,3,8,9]
# out_square = square(array)
# out_cube = cube(array)

t1 = threading.Thread(target=square, args=(array,))
t2 = threading.Thread(target=cube, args=(array,))

t1.start()
t2.start()

t1.join()
t2.join()


print("done in : ", time.time() - t)
print("Hah... I am done with all my work now!")




