import threading
import os
import time

def sing(name,num):
    for i  in range(num):
        print(name+"唱歌。。。")
        time.sleep(0.5)


def dance(name,num):
    for i  in range(num):
        print(name+"跳舞。。。")
        time.sleep(0.5)


if __name__ == "__main__":

    thread_sing = threading.Thread(target=sing,args=("小明",3))
    thread_dance = threading.Thread(target=dance,args=("小美",3))


    thread_sing.daemon = True
    thread_dance.daemon = True

    thread_sing.start()
    thread_dance.start()
    exit()
    thread_list = threading.enumerate()
    print(thread_list)

