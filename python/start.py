from math import sin
from typing import Callable, Generator, Iterable, Iterator, NamedTuple

TestFn = Callable[[float, float], float]
TestCases = Iterable[tuple[float, float]]
Env = Iterator[tuple[float, float]]
ZERO = 0.

class TheError(NamedTuple):
    i: int
    y: float
    fx: float
    margin: float

    @property
    def diff(self):
        return self.y - self.fx

    @property
    def err(self):
        return abs(self.diff)

    def message_cli(self):
        sign = "+" if self.diff > 0 else "-"
        return f"Error: {self.err:.2f} ({sign})"

def banner(msg: str) -> str:
    return f" ------ {msg} ------"

class Level(NamedTuple):
    i: int
    env: Env
    margin: float = 0 

    @staticmethod
    def case_report(i: int, x: float, y: float, fx: float):
        return f"{i:2d} | Testing on {x:2.2f} -> {y:2.2f} | >> {fx:2.2f} "

    def error_report(self, theError: TheError, sum_err: float):
        margin_msg = f"{sum_err:.2f}<={self.margin:.2f}(margin)" if self.margin != 0 else f"{sum_err}"
        return f"{theError.message_cli()} | Cumulative: {margin_msg}"
    
    def game_end_message(self, sum_err: float):
        return f"Failed: Cumulative: {sum_err:.2f} is larger than margin {self.margin:.2f}"

    @classmethod
    def from_cases(cls, i: int, cases: TestCases, margin: float) -> Level:
        return cls(i, iter(cases), margin)

    def test_(self, f: TestFn):
        print()
        print(banner(f"Level {self.i}"))
        sum_err_ = diff_ = ZERO
        i = 0
        while True:
            try:
                x, y = next(self.env)
                fx = f(x, diff_)
                print(self.case_report(i, x, y, fx), end="")
                theError = TheError(i, y, fx, self.margin)
                diff_ = theError.diff
                sum_err_ += theError.err
                print(f"| {self.error_report(theError, sum_err_)}")
                if abs(sum_err_) <= self.margin:
                    i += 1
                    continue
                else:
                    print(self.game_end_message(sum_err_))
                    exit()
            except StopIteration:
                break
        print(banner("All Pass"))

lvl_i_ = 0
def SL_(cases: TestCases, margin=0):
    global lvl_i_
    lvl_i_ += 1
    return Level.from_cases(lvl_i_, cases, margin)

## Level 1
lvl = SL_([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
test_fn = lambda x, _: x
lvl.test_(test_fn)

## Level 2
lvl = SL_([(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)])
test_fn = lambda x, _: 1
lvl.test_(test_fn)

## Level 3
lvl = SL_([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
test_fn = lambda x, _: x + 1
lvl.test_(test_fn)

## Level 4
lvl = SL_([(0, 1), (1, 3), (2, 5), (3, 7), (4, 9)])
test_fn = lambda x, _: 2 * x + 1
lvl.test_(test_fn)

## Level 4.x
lvl = SL_([(x, 2*x + 1) for x in range(20)], 50)
store_: float = 0
def f4(x: float, err: float):
    global store_
    store_ += 1
    return x + store_

lvl.test_(f4)

## Level 5
lvl = SL_([(x, 2.1*x + 0.8) for x in range(5)])
test_fn = lambda x, _: 2.1 * x + 0.8 
lvl.test_(test_fn)

## Level 6
lvl_ = SL_([(0, 1.2), (1, 3.1), (2, 5.51), (3, 7.2), (4, 8.9)], 2)
test_fn = lambda x, _: 2 * x + 1.1 
lvl_.test_(test_fn)

## Level 7
lvl = SL_([(x, sin(0.2 * x)) for x in range(12)], 2.2)
test_fn = lambda x, _: 0.125 * x 
lvl.test_(test_fn)

## Level 7
lvl = SL_([(x, sin(0.2 * x)) for x in range(20)], 5)
store_ = 0
def f7(x: float, err: float):
    global store_
    store_ += err - 0.8
    return x + store_ 

lvl.test_(f7)