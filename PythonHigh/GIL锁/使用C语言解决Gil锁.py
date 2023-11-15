import ctypes
import threading

# testlib = ctypes.cdll.LoadLibrary("./libtest.so")
testlib = ctypes.WinDLL("./library.dll")

t1 = threading.Thread(target=testlib.Loop)
t1.start()

while True:
    pass