import threading
import time

count = 0
def addCount1():

    for i in range(50):
        lock1.acquire()
        global count
        count += 1
        lock1.release()

    print("thread1的count:", count)

def addCount2():

    for i in range(50):
        lock1.acquire()
        global count
        count += 1
        lock1.release()
    print("thread2的count:", count)


if __name__ == "__main__":

    lock1 = threading.Lock()
    thread1 = threading.Thread(target=addCount1)
    thread2 = threading.Thread(target=addCount2)

    thread1.start()
    thread2.start()

    while len(threading.enumerate()) != 1:
        time.sleep(1)

    print("最终count:", count)