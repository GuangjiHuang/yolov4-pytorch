class A(object):
    b_name = 'A-C'
    def __init__(self):
        #self.b_name = 'A'
        pass

    def print(self):
        print("A's method")

class B(A):

    #b_name = 'B-c'
    def __init__(self):
        #self.b_name = 'B'
        super().__init__()

    @classmethod
    def print(cls):
        print("B's class method")

class Mystr(str):
    def __init__(self, string):
        self.string = string
        self.__p_val = "private"
        self._Mystr__p_val = "hello"

    def my_func(self):
        print("my_func")

my_str = Mystr("hello")

print(my_str._Mystr__p_val)
