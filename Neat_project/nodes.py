import random
from enum import Enum

activation_name = ["sigmoid", "tanh"]
aggregation_function = ["sum", "max"]

class NodeType(Enum):
    INPUT = "Input"
    HIDDEN = "Hidden"
    OUTPUT = "Output"

class Node:
    def __init__(self, node_id, node_type):
        self.node_id = node_id
        self.node_type = node_type
        self.bias = random.uniform(-1, 1)
        self.response = 1.0  # Assuming a default response value
        
        if node_type == NodeType.HIDDEN or node_type == NodeType.OUTPUT:
            self.activation = random.choice(activation_name)
            self.aggregation = random.choice(aggregation_function)
            if self.activation not in activation_name:
                raise ValueError(f"Invalid activation function: {self.activation}")
            if self.aggregation not in aggregation_function:
                raise ValueError(f"Invalid aggregation function: {self.aggregation}")
        else:
            self.activation = None
            self.aggregation = None

    def __str__(self):
        return f"Node {self.node_id}\nType: {self.node_type}\nBias: {self.bias}\nActivation: {self.activation}\nAggregation: {self.aggregation}"
