from __future__ import annotations

import math

class PlattScoreCalibrator:
    """Tiny logistic score calibrator trained on held-out relevance labels."""
    def __init__(self,learning_rate:float=.05,epochs:int=300,l2:float=1e-3)->None:self.learning_rate=learning_rate;self.epochs=epochs;self.l2=l2;self.a=1.0;self.b=0.0
    @staticmethod
    def _sigmoid(x:float)->float:
        if x>=0:z=math.exp(-x);return 1/(1+z)
        z=math.exp(x);return z/(1+z)
    def fit(self,rows:list[tuple[float,int|bool|float]])->None:
        if not rows:return
        for _ in range(self.epochs):
            for score,label in rows:
                y=max(0.0,min(1.0,float(label)));pred=self._sigmoid(self.a*float(score)+self.b);err=y-pred
                self.a+=self.learning_rate*(err*float(score)-self.l2*self.a);self.b+=self.learning_rate*err
    def predict(self,score:float)->float:return self._sigmoid(self.a*float(score)+self.b)
    def __call__(self,score:float)->float:return self.predict(score)
