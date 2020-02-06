from typing import List


def map_y_array_to_int(y_letter):
    '''
    >>> map_y_array_to_int(['A', 'B', 'C'])
    [0, 1, 2]
    '''

    def map_to_int(letter):
        if letter == 'A':
            return 0
        elif letter == 'B':
            return 1
        elif letter == 'C':
            return 2

    return [map_to_int(y) for y in y_letter]


def proba_to_letters(probabilities):
    '''
    >>> proba_to_letters([[0.4, 0.3, 0.3], [0.1,0.5,0.4], [0.2,0.1,0.7]])
    ['A', 'B', 'C']
    '''

    # def map_to_ints(xs):
    #     return  max(range(len(xs)), key=xs.__getitem__)
    def map_to_letter(xs):
        letters = ['A', 'B', 'C']
        return letters[max(range(len(xs)), key=xs.__getitem__)]

    return list(map(map_to_letter, probabilities))


def letters_to_ints(xs: List[str]):
    '''
    >>> letters_to_ints(['A', 'B', 'C'])
    [0, 1, 2]
    '''
    letters = {'A': 0, 'B': 1, 'C': 2}
    return [letters[x] for x in xs]


def floats_to_ints(xs: List[float]):
    '''
    >>> floats_to_ints([0.2, 0.58, 1.74])
    [0, 1, 2]
    '''
    return [round(x) for x in xs]


def proba_to_ints(probabilities):
    '''
    >>> proba_to_ints([[0.4, 0.3, 0.3], [0.1,0.5,0.4], [0.2,0.1,0.7]])
    [0, 1, 2]
    '''
    return letters_to_ints(proba_to_letters(probabilities))
