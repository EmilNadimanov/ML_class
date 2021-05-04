from functools import reduce
from typing import Union, List, overload


Numeric = Union[int, float]


def _typecheck(variable, *allowed_classes):
    if isinstance(variable, allowed_classes):
        pass
    else:
        raise NotImplementedError(
            f"operation not supported for type {variable.__class__}")


class Scalar:
    def __init__(self: 'Scalar', val: float):
        _typecheck(val, float, int)
        self.val = float(val)

    @overload
    def __mul__(self: 'Scalar', other: 'Scalar') -> 'Scalar':
        pass

    @overload
    def __mul__(self: 'Scalar', other: 'Vector') -> 'Vector':
        pass

    def __mul__(self, other):
        _typecheck(other, Scalar, Vector)
        if isinstance(other, Vector):
            entries = list(map(lambda x: x.val * self.val, other.entries))
            return Vector(*entries)
        elif isinstance(other, Scalar):
            return Scalar(other.val * self.val)

    def __add__(self: 'Scalar', other: 'Scalar') -> 'Scalar':
        _typecheck(other, Scalar)
        return Scalar(self.val + other.val)

    def __sub__(self: 'Scalar', other: 'Scalar') -> 'Scalar':
        _typecheck(other, Scalar)
        return Scalar(self.val - other.val)

    def __truediv__(self: 'Scalar', other: 'Scalar') -> 'Scalar':
        _typecheck(other, Scalar)
        return Scalar(self.val / other.val)

    def __rtruediv__(self: 'Scalar', other: 'Vector') -> 'Vector':
        _typecheck(other, Vector)
        entries = list(map(lambda x: x.val / self.val, other))
        return Vector(*entries)

    def __repr__(self: 'Scalar') -> str:
        return "Scalar(%r)" % self.val

    def sign(self: 'Scalar') -> int:
        if self.val == 0:
            return 0
        elif self.val > 1:
            return 1
        else:
            return -1

    def __float__(self: 'Scalar') -> float:
        return self.val


class Vector:
    def __init__(self: 'Vector', *entries: Numeric):
        self.entries: List['Scalar'] = [Scalar(e) for e in entries]

    @staticmethod
    def zero(size: int) -> 'Vector':
        return Vector(*[0 for _ in range(int(size))])

    def __add__(self: 'Vector', other: 'Vector') -> 'Vector':
        _typecheck(other, Vector)
        if len(self) == len(other):
            vec = Vector.zero(len(self))
            for idx in range(len(vec)):
                vec[idx] = self[idx] + other[idx]
        else:
            raise Exception(
                'Summation of vectors of unequal size is not allowed')

        return vec

    @overload
    def __mul__(self: 'Vector', other: int) -> 'Vector':
        pass

    @overload
    def __mul__(self: 'Vector', other: 'Vector') -> Scalar:
        pass

    def __mul__(self, other):
        _typecheck(other, Vector, int, Scalar)
        if isinstance(other, int):
            entries = list(map(lambda x: x.val * other, self.entries))
            return Vector(*entries)
        elif isinstance(other, Scalar):
            entries = list(map(lambda x: x.val * other.val, self.entries))
            return Vector(*entries)
        elif isinstance(other, Vector):
            if len(self) == len(other):
                vec = Vector.zero(len(self))
                for idx in range(len(vec)):
                    vec[idx] = self[idx] * other[idx]
                return reduce(lambda x, y: x + y, vec)
            else:
                raise Exception(
                    'Summation of vectors of unequal size is not allowed')

    def __sub__(self: 'Vector', other: 'Vector') -> 'Vector':
        _typecheck(other, Vector)
        return self + other * (-1)

    def magnitude(self: 'Vector') -> 'Scalar':
        return Scalar(len(self.entries))

    def unit(self: 'Vector') -> 'Vector':
        return self / self.magnitude()

    def __len__(self: 'Vector') -> int:
        return len(self.entries)

    def __repr__(self: 'Vector') -> str:
        return "Vector%s" % repr(self.entries)

    def __iter__(self: 'Vector'):
        return iter(self.entries)

    def __getitem__(self, key: int):
        return self.entries[key]

    def __setitem__(self, key: int, third):
        self.entries[key] = third
