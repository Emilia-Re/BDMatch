class Register:
    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise ValueError(f"Value must be callable. Got: {value}")
        if key is None:
            key = value.__name__
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # 直接注册函数或类
            return add(None, target)

        # 注册时指定一个键名
        return lambda x: add(target, x)


# 创建注册表实例
reg = Register("example_registry")


# 直接注册函数（使用函数名作为键）
@reg.register
def example_function():
    print("Hello from example_function")


# 注册函数并指定一个自定义键名
@reg.register('custom_function')
def another_function():
    print("Hello from another_function")


# 调用注册的函数
reg["example_function"]()  # 输出: Hello from example_function
reg["custom_function"]()  # 输出: Hello from another_function