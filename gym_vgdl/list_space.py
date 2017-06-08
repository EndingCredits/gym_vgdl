from gym import Space

class list_space(Space):
    def __init__(self, basespace):
        self.basespace = basespace
        self.shape = [None] + list(self.basespace.shape)

    #def sample(self):
    #    return [space.sample() for space in self.spaces]

    def shape(self):
        return tuple(self.shape)
