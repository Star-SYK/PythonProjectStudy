

class MyOpen(object):
    def __init__(self,fileName,mode):
        self.fileName = fileName
        self.mode = mode

    def __enter__(self):
        print("==进入上文==")
        self.file = open(self.fileName,self.mode,encoding="utf_8")
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("==进入上文==")
        self.file.close()

if __name__ == "__main__":

    with MyOpen("C:\\Users\\Administrator\\Desktop\\新建文本文档.txt","r") as file:
        file_text = file.read()
        print(file_text)