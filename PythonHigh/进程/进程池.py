import multiprocessing
import time


def copy_file():
    print("正在拷贝文件.....", multiprocessing.current_process())
    time.sleep(0.5)


if __name__ == '__main__':

    pool = multiprocessing.Pool(3)

    for i in range(6):
        # 同步方式执行
        # pool.apply(copy_file)

        # 异步方式执行
        pool.apply_async(copy_file)

    # 线程池不在接收新的任务
    pool.close()
    # 让主线程等待子进程执行结束后再退出
    pool.join()