import time


class Timer:
    def __init__(self, name: str = None) -> None:
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc_info):
        self.t1 = time.time()
        elapsed_time = self.t1 - self.t0
        if self.name is None:
            print(f"Execution completed in {elapsed_time:.3f}s")
        else:
            print(f"{self.name} completed in {elapsed_time:.3f}s")
