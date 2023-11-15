

data_list = [x*2 for x in range(10)]
print(data_list)

# 1.列表推到器初始化生成器
data_list2 = (x*2 for x in range(10))

value = next(data_list2)
print(value)



# 2.yield初始化生成器

def test():
    yield 10

n = test()
print(next(n))



# 生成器实现斐波拉数列

fipolacci_list = ()

# 普通方法实现
def Fipolacci(index):
    a,b = 1,1
    for i in range(index):
        a,b = b,a+b
    return a
# 生成器实现
def Fipolacci2(index):
    a,b = 1,1

    current_index = 0
    while current_index < index:
        data = a
        a, b = b, a + b
        current_index += 1
        stop_flag =  yield data
        if stop_flag == 1:
            return "斐波拉契生成器停止"


if __name__ == "__main__":
    for i in range(10):
        print(Fipolacci(i))

    generator = Fipolacci2(1)
    print(next(generator))

    try:
        generator.send(1)
        print(next(generator))
    except Exception as e:
        print(e)