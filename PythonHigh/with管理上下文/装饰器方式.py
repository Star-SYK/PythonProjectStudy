from contextlib import contextmanager

@contextmanager
def myOpen(fileName,mode):
    print("==进入上文==")
    file = open(fileName, mode, encoding="utf_8")
    yield file
    print("==进入上文==")
    file.close()

with myOpen("C:\\Users\\Administrator\\Desktop\\新建文本文档.txt","r") as file:
    file_text = file.read()
    print(file_text)
