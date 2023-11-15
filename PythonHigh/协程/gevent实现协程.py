
from gevent import monkey
monkey.patch_all()

import time
import gevent



def work1():
    while True:
        print("----正在执行任务work1----",gevent.getcurrent())
        time.sleep(0.5)
        # gevent.sleep(0.5)



def work2():
    while True:
        print("----正在执行任务work2----",gevent.getcurrent())
        time.sleep(0.5)
        # gevent.sleep(0.5)



if __name__ == "__main__":
    g1 = gevent.spawn(work1)
    g2 = gevent.spawn(work2)

    g1.join()
    g2.join()



