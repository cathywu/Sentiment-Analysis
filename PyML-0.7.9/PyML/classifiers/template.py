from baseClassifiers import Classifier

class Template(Classifier) :

    """A template for a pyml classifier"""

    attributes = {}

    def __init__(self, arg = None, **args) :

        Classifier.__init__(self, **args)

    def train(self, data, **args) :

        Classifier.train(self, data, **args)

        # this should be the last command in the train function
        # if you redefine the "test" function you can follow the code in
        # assess.test to save the testingTime
        self.log.trainingTime = self.getTrainingTime()


    def decisionFunc(self, data, i) :

        return margin

    def classify(self, data, i) :

        return (margin, classification)

    
