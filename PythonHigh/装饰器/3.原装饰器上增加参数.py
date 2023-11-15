def test(path):
    print("路由地址为：",path)
    def authentication(func):
        def function_in():
            print("----正在进行身份验证，请稍后。。。。---")
            func()
        return function_in
    return authentication

@test("login.py")
def login():
    print("=====正在登录====")


if __name__ == '__main__':
    login()
    pass