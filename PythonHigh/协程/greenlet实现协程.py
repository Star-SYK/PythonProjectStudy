import time

from greenlet import greenlet

def work1():
    while True:
        print("----正在执行任务work1----")
        time.sleep(0.5)
        # g2.switch()


def work2():
    while True:
        print("----正在执行任务work2----")
        time.sleep(0.5)
        # g1.switch()



if __name__ == "__main__":
    g1 = greenlet(work1)
    g2 = greenlet(work2)


    g1.switch()