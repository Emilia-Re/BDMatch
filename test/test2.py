import time


def timer(threshold):
    def decoractor(func):
        def wrapper(*args,**kwargs):
            start=time.time()
            result=func(*args,**kwargs)
            end=time.time()
            print(f"Time elapsed:{end-start}")
            if end-start>threshold:
                print("time used exceeded the threshold!")
            return result
        return wrapper
    return decoractor
@timer(0.2)
def square(x):
    time.sleep(0.1)
    return x**2

# new_func=decoractor(square)

print(square(2))

