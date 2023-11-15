# def authentication(func):
#     def function_in(username,password):
#         print("----正在进行身份验证，请稍后。。。。---")
#         func(username,password)
#     return function_in
#
#
# @authentication
# def login(username,password):
#     print("=====正在登录====")
#     print("username:",username)
#     print("password:",password)


def authentication(func):
    def function_in(*args,**kwargs):
        print("----正在进行身份验证，请稍后。。。。---")
        return func(*args,**kwargs)
    return function_in


@authentication
def login(*args,**kwargs):
    print("=====正在登录====")
    print("args:",args)
    print("kwargs:",kwargs)

    return "success"

if __name__ == '__main__':
    result = login("张三",password = "123456")
    print(result)
    pass