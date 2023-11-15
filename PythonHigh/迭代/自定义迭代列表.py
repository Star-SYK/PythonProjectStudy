class MyList(object):
    def __init__(self):
        self.items = []


    def __iter__(self):
        myList_Iterator = MyListIterator(self.items)
        return myList_Iterator

    def addItem(self,data):
        self.items.append(data)
        print(self.items)


class MyListIterator:
    def __init__(self,list):
        self.list = list
        #记录迭代器迭代的位置
        self.current_index = 0


    def __next__(self):
        # 1.判断下标是否越界
       if self.current_index < len(self.list):
           data = self.list[self.current_index]
           self.current_index += 1
           return data
       else:
           raise StopIteration


    def __iter__(self):
        return self




if __name__ == "__main__":
    myList = MyList()

    myList.addItem("张三")
    myList.addItem("李四")
    myList.addItem("王五")

    for value in myList:
        print("name:",value)