from collections.abc import Iterable


ret = isinstance([1,2,3],Iterable)

print(ret)
print("-"*20)


ret = isinstance(10,Iterable)

print(ret)
print("-"*20)

class MyClass(object):

    def __iter__(self):
        pass


myClass = MyClass()

ret = isinstance(myClass,Iterable)

print(ret)