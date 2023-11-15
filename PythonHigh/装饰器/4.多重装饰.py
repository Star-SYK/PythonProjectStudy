

def makBlod(func):
    def function_in():
        return "<b>"+func()+"</b>"
    return function_in

def makItalic(func):
    def function_in():
        return "<i>"+func()+"</i>"
    return function_in

@makItalic
@makBlod
def text():
    return "这是一段测试文字"

if __name__ == '__main__':
    result = text()
    print(result)