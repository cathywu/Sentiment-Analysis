
class PyMLobject (object) :

    def __init__(self, arg=None, **args) :
        """
        Takes care of keyword arguments that are defined in the attributes
        class attribute
        """
        if not hasattr(self, 'attributes') : return
        if self.__class__ == arg.__class__ :
            for attribute in self.attributes :
                setattr(self, attribute, getattr(arg, attribute))
        else :
            for attribute in self.attributes :
                if attribute in args :
                    setattr(self, attribute, args[attribute])
                else :
                    setattr(self, attribute, self.attributes[attribute])
    
