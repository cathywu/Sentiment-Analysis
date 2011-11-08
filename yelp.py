import json
from data import DefDict
import movie
import classifier
import ngrams
class Yelp:
    def __init__(self, loc):
        f = open(loc)
        jsons = [json.loads(i) for i in f.readlines()]
        jsons = [i for i in jsons if i['type'] == 'review']
        self.stars = dict([(i, []) for i in range(1, 6)])
        for j in jsons:
            if j['type'] == 'review':
                self.stars[j['stars']].append(j['text'])
                
    def rate(self, classifier, n, dictionary):
        outputs = []
        for group in self.stars:
            classes = DefDict(0)
            print "Rating group", group
            for rev in self.stars[group]:
                vec = ngrams.ngram_vector(n, rev, dictionary)
                classes[classifier.classify(vec)] += 1                            
            outputs.append(classes)
        return outputs
    
if __name__ == "__main__":
    m = movie.MovieReviews(classifier.BayesPresenceClassifier, 2)
    print "Reading Yelp data"
    y = Yelp("yelp/json")
    print "Rating"
    print y.rate(m.classifier, 2, m.dictionary)
