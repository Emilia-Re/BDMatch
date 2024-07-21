def add_method_decorator(cls):
    """装饰器，用于给类添加一个新的方法"""

    def new_method(self):
        print("This is a new method added by the decorator.")

    cls.new_method = new_method
    return cls


@add_method_decorator
class MyClass:
    def __init__(self, value):
        self.value = value

    def original_method(self):
        print(f"Original method. Value: {self.value}")


# 使用装饰器装饰后的类
obj = MyClass(10)
obj.original_method()  # 输出: Original method. Value: 10
obj.new_method()  # 输出: This is a new method added by the decorator.
