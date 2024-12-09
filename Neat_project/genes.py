import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
from nodes import Node, NodeType
from connection import Connection



activation_name = ["sigmoid", "tanh"]
aggregation_function = ["sum", "max"]

class Genome:
    def __init__(self, genome_id, n_inputs, n_outputs):
        self.genome_id = genome_id
        self.nodes = []
        self.connections = []
        self.fitness = 0
        self.innovation_counter = 0
        self.show_weights = False
        self.species_id=None
        self.adjusted_fitness=0

        # Initialize input nodes with negative IDs
        for i in range(1, n_inputs + 1):
            node_type = NodeType.INPUT
            node = Node(-i, node_type)
            self.nodes.append(node)

        # Initialize output nodes with non-negative IDs starting from 0
        for i in range(n_outputs):
            node_type = NodeType.OUTPUT
            node = Node(i, node_type)
            self.nodes.append(node)

        # Create connections from each input to each output node
        input_nodes = [node for node in self.nodes if node.node_type == NodeType.INPUT]
        output_nodes = [node for node in self.nodes if node.node_type == NodeType.OUTPUT]

        for input_node in input_nodes:
            for output_node in output_nodes:
                self.innovation_counter += 1
                self.connections.append(Connection(self.innovation_counter, input_node.node_id, output_node.node_id))
                
    def get_innovation_number(self):
        self.innovation_counter += 1
        return self.innovation_counter
    
    def mutate_weight_and_bias(self):
        for node in self.nodes:
            mutation_type = random.choice(['replace', 'none'])
            if mutation_type == 'replace':
                if random.random() < 0.5:
                    node.bias += random.uniform(-0.5, 0.5)
                else:
                    node.bias -= random.uniform(-0.5, 0.5)
        for conn in self.connections:
            mutation_type = random.choice(['replace', 'none'])
            if mutation_type == 'replace':
                if random.random() < 0.5:
                    conn.weight += random.uniform(-0.5, 0.5)
                else:
                    conn.weight -= random.uniform(-0.5, 0.5)
                    
    def getConnectionFromNumbers(self,in_node, out_node):
        for conn in self.connections:
            if conn.in_node_id == in_node and conn.out_node_id == out_node:
                return conn
            
    def getNodeFromNumbers(self,inode_num):
        for conn in self.nodes:
            if conn.node_id == inode_num:
                return conn
    
    def mutate_add_node(self):
        enabled_connections = [conn for conn in self.connections if conn.enabled]
        if not enabled_connections:
            return

        chosen_conn = random.choice(enabled_connections)
        chosen_conn.disable()

        new_node_id = len(self.nodes) + 1
        new_node = Node(new_node_id, NodeType.HIDDEN)
        self.nodes.append(new_node)

        conn1 = Connection(self.get_innovation_number(), chosen_conn.in_node_id, new_node_id)
        conn1.weight = chosen_conn.weight
        self.connections.append(conn1)

        conn2 = Connection(self.get_innovation_number(), new_node_id, chosen_conn.out_node_id)
        conn2.weight = 1.0
        self.connections.append(conn2)

    def mutate_add_connection(self):
        possible_pairs = [
            (in_node.node_id, out_node.node_id)
            for in_node in self.nodes if in_node.node_type != NodeType.OUTPUT
            for out_node in self.nodes if out_node.node_id != in_node.node_id and out_node.node_type != NodeType.INPUT
        ]

        existing_connections = {(conn.in_node_id, conn.out_node_id) for conn in self.connections}
        possible_pairs = [pair for pair in possible_pairs if pair not in existing_connections]

        if not possible_pairs:
            return

        in_node_id, out_node_id = random.choice(possible_pairs)

        if self.creates_cycle(in_node_id, out_node_id):
            return

        self.innovation_counter += 1
        new_connection = Connection(self.innovation_counter, in_node_id, out_node_id)
        self.connections.append(new_connection)



    def creates_cycle(self, start_node, end_node):
        G = nx.DiGraph()
        for conn in self.connections:
            if conn.enabled:
                G.add_edge(conn.in_node_id, conn.out_node_id)

        # Add the potential new edge temporarily to check for cycles
        G.add_edge(start_node, end_node)
        try:
            cycle = nx.find_cycle(G, orientation="original")
            return True
        except nx.NetworkXNoCycle:
            return False
        
    def mutate_change_activation_aggregation(self):
        for node in self.nodes:
            if node.node_type == NodeType.HIDDEN:
                if random.random() < 0.1:  # 10% chance to change activation function
                    node.activation = random.choice(activation_name)
                if random.random() < 0.1:  # 10% chance to change aggregation function
                    node.aggregation = random.choice(aggregation_function)

    def mutate(self):
        if random.random() < 0.3:
            self.mutate_weight_and_bias()
        elif random.random() < 0.2:
            self.mutate_add_node()
        elif random.random() < 0.1:
            self.mutate_add_connection()
        self.mutate_change_activation_aggregation()

            
    
    def crossover(self, parent1, parent2, reenable_connection_gene_rate=0.25):
        # Clear existing nodes and connections in the child genome
        self.nodes = []
        self.connections = []
        self.innovation_counter = 0

        # Determine the fitter parent based on fitness
        if parent1.fitness >= parent2.fitness:
            best_parent = parent1
            other_parent = parent2
        else:
            best_parent = parent2
            other_parent = parent1

        # Crossover connections
        # Randomly add matching genes from both parents
        for c_gene in best_parent.connections:
            matching_gene = other_parent.getConnectionFromNumbers(c_gene.in_node_id, c_gene.out_node_id)

            if matching_gene is not None:
                # Randomly choose where to inherit gene from
                if random.random() < 0.5:
                    child_gene = deepcopy(c_gene)
                else:
                    child_gene = deepcopy(matching_gene)
            else:
                # No matching gene - inherit disjoint and excess genes from best parent
                child_gene = deepcopy(c_gene)

            # Apply rate of disabled gene being re-enabled
            if not child_gene.enabled:
                is_reenabled = random.random() <= reenable_connection_gene_rate
                enabled_in_best_parent = best_parent.getConnectionFromNumbers(child_gene.in_node_id, child_gene.out_node_id).enabled

                if is_reenabled or enabled_in_best_parent:
                    child_gene.enabled = True

            self.connections.append(child_gene)

        # Crossover Nodes
        # Randomly add matching genes from both parents
        all_nodes = {node.node_id: node for node in (best_parent.nodes + other_parent.nodes)}

        for node in best_parent.nodes:
            matching_node = other_parent.getNodeFromNumbers(node.node_id)
            if matching_node is not None:
                # Randomly choose where to inherit gene from
                if random.random() < 0.5:
                    child_node = deepcopy(node)
                else:
                    child_node = deepcopy(matching_node)
            else:
                # No matching gene - inherit disjoint and excess genes from best parent
                child_node = deepcopy(node)

            self.nodes.append(child_node)

        # Ensure all input nodes are included
        input_nodes = [node for node in all_nodes.values() if node.node_type == NodeType.INPUT]
        for node in input_nodes:
            if node.node_id not in [n.node_id for n in self.nodes]:
                self.nodes.append(deepcopy(node))
    
        
        
    def distance(genome1, genome2, c1=1, c2=1, c3=0.4):
        genes1 = genome1.connections
        genes2 = genome2.connections

        max_innov1 = max([gene.innovation_number for gene in genes1])
        max_innov2 = max([gene.innovation_number for gene in genes2])
        max_innov = max(max_innov1, max_innov2)

        matching_genes = 0
        weight_diff_sum = 0.0
        excess_disjoint = 0

        for gene1 in genes1:
            found_matching = False
            for gene2 in genes2:
                if gene1.innovation_number == gene2.innovation_number:
                    matching_genes += 1
                    weight_diff_sum += abs(gene1.weight - gene2.weight)
                    found_matching = True
                    break

            if not found_matching:
                if gene1.innovation_number <= max_innov2:
                    excess_disjoint += 1

        for gene2 in genes2:
            found_matching = False
            for gene1 in genes1:
                if gene2.innovation_number == gene1.innovation_number:
                    found_matching = True
                    break

            if not found_matching and gene2.innovation_number <= max_innov1:
                excess_disjoint += 1

        average_weight_diff = weight_diff_sum / matching_genes if matching_genes > 0 else 0.0

        N = max(len(genes1), len(genes2))
        distance = (c1 * excess_disjoint / N) + (c2 * excess_disjoint / N) + (c3 * average_weight_diff)

        return distance


    def visualize(self):
        G = nx.DiGraph()

        for node in self.nodes:
            G.add_node(node.node_id, label=f'{node.node_id}')

        for conn in self.connections:
            if self.show_weights:
                if conn.enabled:
                    G.add_edge(conn.in_node_id, conn.out_node_id, label=round(conn.weight, 2))
            else:
                if conn.enabled:
                    G.add_edge(conn.in_node_id, conn.out_node_id)

        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='lightblue', node_size=2000, font_size=10, connectionstyle='arc3,rad=0.1')

        if self.show_weights:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        else:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, alpha=0.0)

        plt.show()
        
    def visualize2(self):
        G = nx.DiGraph()

        for node in self.nodes:
            G.add_node(node.node_id, label=f'{node.node_id}')

        for conn in self.connections:
            if conn.enabled:
                if self.show_weights:
                    G.add_edge(conn.in_node_id, conn.out_node_id, label=round(conn.weight, 2))
                else:
                    G.add_edge(conn.in_node_id, conn.out_node_id)

        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'label')

        # Draw enabled connections
        nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='lightblue', node_size=2000, font_size=10, connectionstyle='arc3,rad=0.1')

        # Draw disabled connections as dotted lines
        for conn in self.connections:
            if not conn.enabled:
                nx.draw_networkx_edges(G, pos, edgelist=[(conn.in_node_id, conn.out_node_id)], style='dotted', edge_color='red', alpha=0.5)

        if self.show_weights:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        else:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, alpha=0.0)

        plt.show()
