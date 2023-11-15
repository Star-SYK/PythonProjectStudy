
def authentication(func):

    def function_in():
        print("----正在进行身份验证，请稍后。。。。---")
        func()
    return function_in


@authentication
def login():
    print("=====正在登录====")


if __name__ == '__main__':
    login()
    pass