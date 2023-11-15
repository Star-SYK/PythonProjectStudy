
class Foo(object):

    def __init__(self):
        self.current_price = 10

    def price(self):
        return self.current_price

    def set_price(self,value):
        self.current_price = value

    def delete_price(self):
        self.current_price = 0

    bar = property(price,set_price,delete_price)

fo = Foo()

print(fo.bar)

fo.bar = 20
print(fo.bar)

del fo.bar
print(fo.bar)

