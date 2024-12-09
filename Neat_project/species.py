from nodes import Node, NodeType
from connection import Connection
from genes import Genome
import random

class Species:
    def __init__(self, representative_genome, species_id):
        self.representative_genome = representative_genome
        self.members = [representative_genome]
        self.species_id = species_id
        self.best_fitness = 0
        self.generations_since_improvement = 0

    def add_member(self, genome):
        self.members.append(genome)

    def clear_members(self):
        self.members = [self.representative_genome]

    def update_stagnation(self):
        current_best_fitness = max(genome.fitness for genome in self.members)
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.generations_since_improvement = 0
        else:
            self.generations_since_improvement += 1

def prune_species(species_list, stagnation_threshold):
    pruned_species = [species for species in species_list if species.generations_since_improvement < stagnation_threshold]

    # Ensure at least one species remains
    if len(pruned_species) == 0 and len(species_list) > 0:
        pruned_species.append(max(species_list, key=lambda s: s.best_fitness))
    
    return pruned_species

def speciate(population, compatibility_threshold, c1, c2, c3):
    species_list = []
    next_species_id = 0

    for genome in population:
        assigned_to_species = False

        for species in species_list:
            distances = Genome.distance(genome, species.representative_genome, c1, c2, c3)
            if distances < compatibility_threshold:
                species.add_member(genome)
                genome.species_id = species.species_id
                assigned_to_species = True
                break

        if not assigned_to_species:
            new_species = Species(genome, next_species_id)
            species_list.append(new_species)
            genome.species_id = next_species_id
            next_species_id += 1

    return species_list

def calculate_adjusted_fitness(species_list):
    for species in species_list:
        for genome in species.members:
            genome.adjusted_fitness = genome.fitness / len(species.members)

def allocate_offspring(species_list, total_offspring):
    total_adjusted_fitness = sum(genome.adjusted_fitness for species in species_list for genome in species.members)
    #print("total_adjusted_fitness: ",total_adjusted_fitness)
    
    if total_adjusted_fitness == 0:
        offspring_counts = [total_offspring // len(species_list)] * len(species_list)
        #print("offspring_counts: ",offspring_counts)
        #print("total_offspring % len(species_list) ",total_offspring % len(species_list))
        for i in range(total_offspring % len(species_list)):
            offspring_counts[i] += 1
    else:
        offspring_counts = []
        for species in species_list:
            species_fitness = sum(genome.adjusted_fitness for genome in species.members)
            offspring_count = int((species_fitness / total_adjusted_fitness) * total_offspring)
            offspring_counts.append(offspring_count)
    
    return offspring_counts
            
def select_parents(species, offspring_count):
    parents = []
    fitnesses = [genome.fitness for genome in species.members]
    for _ in range(offspring_count):
        selected = random.choices(species.members, weights=fitnesses, k=2)
        parents.append(selected)

    return parents
