
class Page(object):

    def __init__(self,current_page,page_size):

        self.current_page = current_page;
        self.page_size = page_size;

    @property
    def start(self):

        return  (self.current_page - 1)*10 + 1

    @property
    def end(self):
        return  self.current_page * self.page_size


page = Page(1,10)

print(page.start)
print(page.end)