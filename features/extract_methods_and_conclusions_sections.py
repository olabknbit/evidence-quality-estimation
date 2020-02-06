def extract_method_from_abstract(abstract: str) -> str:
    import nltk

    sent_text = nltk.sent_tokenize(abstract)
    n = len(sent_text)

    if n == 0:
        return ''
    elif n == 1 or n == 2 or n == 3:
        return sent_text[0]
    elif n == 4 or n == 5:
        return ' '.join(sent_text[1:3])
    elif n == 6 or n == 7 or n == 8:
        return ' '.join(sent_text[1:4])
    else:
        return ' '.join(sent_text[2:5])


def extract_conclusion_from_abstract(abstract: str) -> str:
    import nltk

    sent_text = nltk.sent_tokenize(abstract)
    n = len(sent_text)

    if n < 2:
        return ''
    elif n < 9:
        return sent_text[-1]
    else:
        return ' '.join(sent_text[-2:])
