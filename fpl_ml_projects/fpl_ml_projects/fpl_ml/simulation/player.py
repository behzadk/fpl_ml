class Player:
    def __init__(self, name: str, position: int, value: float, club: str):
        self.name = name
        self.position = position
        self.value = value
        self.club = club

    def copy(self):
        new_cls = type(self)(self.name, self.position, self.value, self.club)
        new_cls.__dict__ = self.__dict__.copy()

        return new_cls

    def __str__(self):
        return f"{self.name}, {self.position}, {self.value}, {self.club}"

    def __eq__(self, other):
        return self.name == other.name