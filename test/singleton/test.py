class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls)
            # cls._instance =cls.__new__(cls)
        return cls._instance

    def __init__(self):
        self.value = None

# 测试
singleton1 = Singleton()
singleton2 = Singleton()

singleton1.value = 10
print(singleton2.value)  # 输出: 10

print(singleton1 is singleton2)  # 输出: True，说明两个对象是同一个实例