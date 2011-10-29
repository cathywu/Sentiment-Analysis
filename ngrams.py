import collections
import data
def words(s):
    words = []
    current = ""
    not_mode = False
    not_words = set(["not", "isn't", "doesn't"])
    for i in s:
        if i.isalnum():
            current += i
        elif i.isspace():
            if not current:
                continue
            if not_mode:
                current += "_NOT"
            words.append(current)
            if current in not_words:
                not_mode = True
            current = ""
        else:
            words.append(i)
            not_mode = False
            if not current:
                continue
            if not_mode:
                current += "_NOT"
            words.append(current)

            current = ""

    return words
            
def ngrams(n, s):
    lwr = s.lower()
    ws = words(s)
    current = collections.deque(ws[:n])
    grams = data.DefDict(1)
    for pos in range(n, len(ws)):
        grams[" ".join(current)] += 1
        current.popleft()
        current.append(ws[pos])
    return grams

def ngrams_to_matrix(grams, classes):
    keysets = [set(k) for k in grams]
    allgramset = set()
    for k in keysets:
        allgramset = allgramset.union(k)
    allgrams = list(allgramset)
    vecs = data.DefDict(())
    for g, c in zip(grams, classes):
        vec = []
        for a in allgrams:
            vec.append(g[a])
        vecs[tuple(vec)] += c
    return data.Data(vecs)
    

if __name__ == "__main__":
    print ngrams(3, "Now is the time for all good men to not come to the aid of their party! Now is the time for all bad women to leave the aid of their country? This, being war, is bad")
