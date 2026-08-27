from data.data_helper import gen_deterministic_data

class Node:

    def __init__(self):
        self.destination_nodes = []
        self.destination_weights = []
        self.inputs = []
        self.relu = True

    def forward(self):
        input = sum(self.inputs)
        if self.relu:
            input = max(0, input)
        if len(self.destination_nodes) == 0:
            self.inputs = [input]
        elif len(self.destination_nodes) > 0:
            for i, destination_node in enumerate(self.destination_nodes):
                output = input * self.destination_weights[i]
                destination_node.inputs.append(output)
            self.inputs = []


class Net:

    def __init__(self):
        self.nodes = []
        self.input_nodes = []
        self.output_nodes = []
        self.queue = []
        self.queue_set = set()
        self.target = None

    def load_input(self, x):
        if len(x) != len(self.input_nodes):
            raise
        for i, x_i in enumerate(x):
            input_node = self.input_nodes[i]
            input_node.inputs.append(x_i)
            self.queue.append(input_node)
            self.queue_set.add(input_node)

    def get_output(self):
        output = []
        for output_node in self.output_nodes:
            node_output = output_node.inputs
            output_node.inputs = []
            if len(node_output) != 1:
                raise
            output.append(node_output[0])
        if len(output) == 1:
            output = output[0]
        return output

    def forward(self):
        if len(self.queue) == 0:
            return False
        else:
            node = self.queue.pop(0)
            self.queue_set.remove(node)
            for destination_node in node.destination_nodes:
                if destination_node not in self.queue_set:
                    self.queue.append(destination_node)
                    self.queue_set.add(destination_node)
            node.forward()
            return True

    def propagate(self):
        while self.forward():
            pass

    def test(self, x, y):
        self.load_input(x)
        self.propagate()
        output = self.get_output()
        error = abs(output - y)
        return error

    def __str__(self):
        print_string = []
        for i, node in enumerate(self.nodes):
            print_string.append(f'Node {i}: {node.inputs} | {node.destination_weights}')
        return '\n'.join(print_string)