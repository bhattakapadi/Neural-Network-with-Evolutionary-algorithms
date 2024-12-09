import math
import neat
from neat.genome import DefaultGenomeConfig, DefaultGenome
from neat.graphs import feed_forward_layers
from nodes import Node, NodeType  # Ensure NodeType is imported correctly


class FeedForwardNetwork:
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)

    def activate(self, inputs):
    
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                node_inputs.append(float(self.values[i]) * float(w))
            
            s = agg_func(node_inputs)

            activation_input = bias + response * s

            self.values[node] = act_func(activation_input)
        return [self.values[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config):
        new_connections = []

        for conn in genome.connections:
            if conn.enabled:
                new_connections.append((str(conn.in_node_id), str(conn.out_node_id)))

        input_keys = [str(k) for k in config.genome_config.input_keys]
        output_keys = [str(k) for k in config.genome_config.output_keys]

        layers = feed_forward_layers(input_keys, output_keys, new_connections)
        node_evals = []


        for layer in layers:
            for node in layer:
                inputs = []
                for conn_key in new_connections:
                    inode, onode = str(conn_key[0]), str(conn_key[1])
                    if onode == node:
                        inputs.append((inode, conn_key[1]))  # Ensure inputs are correctly formed

                # Get the node object from the genome
                ng = genome.getNodeFromNumbers(int(node))
                # Get activation and aggregation functions from the node itself
                aggregation_function = config.genome_config.aggregation_function_defs[ng.aggregation]
                activation_function = config.genome_config.activation_defs[ng.activation]

                # Append node evaluation tuple
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        return FeedForwardNetwork(input_keys, output_keys, node_evals)

