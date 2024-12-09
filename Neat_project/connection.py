import random

class Connection:
    def __init__(self, innovation_number, in_node_id, out_node_id):
        self.innovation_number = innovation_number
        self.in_node_id = in_node_id
        self.out_node_id = out_node_id
        self.weight = random.uniform(-1.0, 1.0)  # Initialize weight
        self.enabled = True

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def is_valid(self, in_node, out_node):
        return self.in_node_id == in_node and self.out_node_id == out_node

    def __str__(self):
        return (f"IN: {self.in_node_id}\nOUT: {self.out_node_id}\n"
                f"WEIGHT: {self.weight:.2f}\nENABLED: {self.enabled}\nINNOV: {self.innovation_number}")
