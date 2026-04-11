

# import time
# import multiprocessing

# def square(numbers):
#     for n in numbers:
#         time.sleep(5)
#         print('square '+ str(n*n))
    
# def cube(numbers):
#     for n in numbers:
#         time.sleep(5)
#         print('cube '+ str(n*n*n))

# if __name__ == "__main__":
#     arr = [2,3,9,8]
#     p1 = multiprocessing.Process(target = square, args = (arr,))
#     p2 = multiprocessing.Process(target = cube, args = (arr,))

#     p1.start()
#     p2.start()

#     p1.join()
#     p2.join()

#     print("Done!")





# # queue is used to share data betweeen two processses
# import time
# import multiprocessing



# def square(numbers,q):
#     global result
#     for n in numbers:
#         q.put(n*n)


# if __name__ == "__main__":
#     arr = [2,3,9,8]
#     q = multiprocessing.Queue()
#     p1 = multiprocessing.Process(target = square, args = (arr,q))

#     p1.start()
#     p1.join()

#     while q.empty() is False:
#         print(q.get())

#     print("Done!")



# Multiprocessing lock


# import time
# import multiprocessing

# def deposite(balance,lock):
#     for i in range(100):
#         time.sleep(0.01)
#         lock.acquire()
#         balance.value = balance.value + 1
#         lock.release()

# def withdraw(balance,lock):
#     for i in range(100):
#         time.sleep(0.01)
#         lock.acquire()
#         balance.value = balance.value - 1
#         lock.release()


# if __name__ == '__main__':
#     balance = multiprocessing.Value('i',200)
#     lock = multiprocessing.Lock()
#     d = multiprocessing.Process(target=deposite, args = (balance,lock))
#     w = multiprocessing.Process(target=withdraw, args=(balance,lock))
    
#     d.start()
#     w.start()

#     d.join()
#     w.join()

#     print(balance.value)



# Multi processing pool

from multiprocessing import Pool

def f(n):
    return n*n

if __name__ == "__main__":
    array = [1,2,3,4,5]

    p = Pool()
    result = p.map(f,array)
    
    print(result)


