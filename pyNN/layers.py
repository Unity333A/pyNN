import numpy as np
from typing import Dict
from pyNN.tensor import Tensor

class Layer:
    def __init__(self)->None:
        self.params:Dict[str,Tensor]={}
    def forward(self, inputs:Tensor)->Tensor:
        """
        Generate the tensor output using
        linear combination of inputs and layer weights
        """
        raise NotImplementedError
    
    def backward(self, grad:Tensor)->Tensor:
        """
        Backpropagate the error gradient through the
        layers adjusting the layer parameters
        """
        raise NotImplementedError
    
    def Linear(Layer):
        """
        It computes output=inputs @ w+ b
        """
        def __init__(self,input_size:int,output_size:int)->None:
            #Inputs will be (batch_size,input_size)
            #Outputs will be (batch_size,output_size)
            super().__init__()
            self.params["w"]=np.random.randn(input_size,output_size)
            self.params["b"]=np.random.randn(output_size)

    

