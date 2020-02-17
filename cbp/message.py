import numpy as np 

class Message(object):
    def __init__(self,sender,val):
        self.sender = sender
        # FIXME divide by zero or nan
        val = np.nan_to_num(val)
        partion = np.sum(val.flatten())
        if np.isclose(partion, [0]):
            print("complaining message zero")
            self.val = np.zeros_like(val)
        else:
            self.val = val/np.sum(val.flatten())
