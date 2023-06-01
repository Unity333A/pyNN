import numpy as np
from typing import Dict
from pyNN.tensor import Tensor

class Layer:
    def __init__(self)->None:
        self.params:Dict[str,Tensor]={}
        self.grads: Dict[str,Tensor]={}
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
    
    def forward(self,inputs:Tensor)->Tensor:
        """
        outputs=inputs @ w + b
        """
        self.inputs=inputs #Storing the inputs for farther usage during backpropagation
        return inputs @ self.params["w"] + self.params["b"]
     
    def backward(self, grad:Tensor)->Tensor:
        """
        If y=f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) * b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad,axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"]
    









