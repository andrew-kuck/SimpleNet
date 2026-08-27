from net.net import Node, Net
from data.data_helper import gen_deterministic_data
from copy import deepcopy
from random import randrange
import math

def create_net():
    net = Net()
    nodes = [Node() for node in range(7)]
    for i in range(3):
        nodes[i].destination_nodes = [nodes[3], nodes[4], nodes[5]]
        nodes[i].destination_weights = [1, 1, 1]
        nodes[i].relu = False
    for i in range(3, 6):
        nodes[i].destination_nodes.append(nodes[6])
        nodes[i].destination_weights.append(0)
    net.nodes = nodes
    net.input_nodes = nodes[:3]
    nodes[6].relu = False
    net.output_nodes = [nodes[6]]
    return net


def load_data(net):
    X, y = gen_deterministic_data(1)
    net.load_input(X[0])
    net.target = y[0]


def perturb_net(net):
    while True:
        target_1_node_index = randrange(len(net.nodes))
        target_1_node = net.nodes[target_1_node_index]
        if len(target_1_node.destination_weights) == 0:
            continue
        target_1_weight_index = randrange(len(target_1_node.destination_weights))
        break
    while True:
        target_2_node_index = randrange(len(net.nodes))
        target_2_node = net.nodes[target_2_node_index]
        if len(target_2_node.destination_weights) == 0:
            continue
        target_2_weight_index = randrange(len(target_2_node.destination_weights))
        if (target_1_node_index, target_1_weight_index) == (target_2_node_index, target_2_weight_index):
            continue
        break
    perturbed_nets = []
    for perturbation in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)):
        perturbed_net = deepcopy(net)
        perturbed_net.nodes[target_1_node_index].destination_weights[target_1_weight_index] += perturbation[0]
        perturbed_net.nodes[target_2_node_index].destination_weights[target_2_weight_index] += perturbation[1]
        perturbed_nets.append(perturbed_net)
    return perturbed_nets


def train(net, rounds):
    for _ in range(rounds):
        perturbed_nets = perturb_net(net)
        best_error = math.inf
        best_perturbed_net = None
        for perturbed_net in perturbed_nets:
            X, y = gen_deterministic_data(10)
            error = 0
            for i in range(len(X)):
                error += perturbed_net.test(X[i], y[i])
            if error < best_error:
                best_error = error
                best_perturbed_net = perturbed_net
        net = best_perturbed_net
    return net, best_error


if __name__ == '__main__':
    net = create_net()
    net, best_error = train(net, 10000)
    print(best_error)
    print(net)


