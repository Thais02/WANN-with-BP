from collections import deque
import re
import pickle
import os
import time

from node import Node
from activation_funcs import *
from utils import *


class Net:
    def __init__(self, nodes=[], config=None):
        self.nodes = nodes

        if self.nodes:
            self.input_nodes = []
            self.output_nodes = []
            for node in self.nodes:
                node.init_weights()
                node.net = self
                if not node.pre_nodes and \
                (node.key in config.genome_config.input_keys if config else node.key <= -1):  # input node
                    self.input_nodes.append(node)
                elif not node.post_nodes and \
                    (node.key in config.genome_config.output_keys if config else node.key >= 0):  # output node
                    self.output_nodes.append(node)
                    node.activation = None
                elif not node.pre_nodes or not node.post_nodes:  # orphan
                    self.nodes.remove(node)

    def _infer(self, x):
        if len(x) != len(self.input_nodes):
            raise ValueError(f'wrong input shape: expecting {len(self.input_nodes)} but got {len(x)}')
        q = deque()  # FIFO
        qs = set()
        for node, input in zip(self.input_nodes, x):
            node.input = input
            nxt_set = set(node.forward())
            diff = nxt_set.difference(qs)
            q.extend(diff)
            [qs.add(x) for x in diff]
        while len(q) != 0:
            node = q.popleft()
            nxt_set = set(node.forward())
            diff = nxt_set.difference(qs)
            q.extend(diff)
            [qs.add(x) for x in diff]

        softmax_outputs = softmax.calc([node.output for node in self.output_nodes])

        for i, node in enumerate(self.output_nodes):
            node.output = softmax_outputs[i]  # done for proper error calculation during backpropagation

        return softmax_outputs

    def activate(self, inputs, batch=False):
        """Forward pass"""
        if batch:
            return [self._infer(input) for input in inputs]
        else:
            return self._infer(inputs)

    def predict(self, inputs, batch=False):
        """Forward pass. Alias to .activate()"""
        return self.activate(inputs, batch)

    def __call__(self, inputs, batch=False):
        """Forward pass"""
        return self.activate(inputs, batch)

    def train(self, x_train, y_train, epochs=1, verbose=True, save=False, x_test=None, y_test=None):
        """Train the network. Generator function yielding loss each epoch

        :param x_train: training input data
        :param y_train: training input labels
        :param epochs: number of epochs to train for, defaults to 1
        :type epochs: int, optional
        :param verbose: Whether to print the start of each epoch, defaults to True
        :type verbose: bool, optional
        :param save: Whether to save a checkpoint of this model every 5 epochs, defaults to False
        :type save: bool, optional
        :param x_test: test input data if test loss is desired, defaults to None
        :param y_test: test input labels if test loss is desired, defaults to None
        :yield: Yields the train loss and the test loss or None
        :rtype: (float, float or None)
        """
        max_target = max(y_train)
        assert max_target+1 == len(self.output_nodes)
        if save:
            save_dir = f'/models/{int(time.time())}'
            os.makedirs(save_dir)
        for epoch in range(epochs):
            if verbose:
                print(f'training... (epoch {epoch})')
            outputs = []
            for x, y in zip(x_train, y_train):
                output = self.activate(x)
                outputs.append(output)
                y_lst = np.zeros(max_target+1)
                y_lst[y] = 1
                q = deque()  # FIFO
                qs = set()
                for i, node in enumerate(self.output_nodes):
                    node.target = y_lst[i]
                    nxt_set = set(node.backward())
                    diff = nxt_set.difference(qs)
                    q.extend(diff)
                    [qs.add(x) for x in diff]
                while len(q) != 0:
                    node = q.popleft()
                    nxt_set = set(node.backward())
                    diff = nxt_set.difference(qs)
                    q.extend(diff)
                    [qs.add(x) for x in diff]
                for node in self.nodes:
                    node.update_weights()
            if save and epoch % 5 == 0:
                save_path = f'{save_dir}/net_{epoch}.pkl'
                with open(save_path, 'wb') as file:
                    print(os.path.join(os.getcwd(), save_path))
                    pickle.dump(self, file)
            if x_test is not None and y_test is not None:
                yield loss(outputs, y_train), loss(self.activate(x_test, batch=True), y_test)
            else:
                yield loss(outputs, y_train), None

    def set_learning_rate(self, learning_rate):
        for node in self.nodes:
            node.learning_rate = learning_rate

    @classmethod
    def from_genome(cls, genome, config):
        nodes = {}

        for key in config.genome_config.input_keys:
            nodes[key] = Node(key=key)

        for key, node_neat in genome.nodes.items():  # only includes hidden- and output-nodes
            nodes[key] = Node(  # TODO add other Node options (bias, etc.)
                key=key,
                activation=activation_funcs.get(node_neat.activation, None)
            )

        for con in genome.connections.values():
            if con.enabled:
                pre = con.key[0]
                post = con.key[1]
                nodes[pre].add_post(nodes[post], con.weight)

        return cls(list(nodes.values()), config)

    @classmethod
    def from_file(cls, path, input_size):
        nodes = {}
        for i in range(input_size):
            key = -(i+1)
            nodes[key] = Node(key=key)
        with open(path, 'r') as model_file:
            lines = model_file.readlines()
            flag = "node"
            for line in lines:
                if "Nodes" in line:
                    flag = "node"
                    continue
                elif "Connections" in line:
                    flag = "connection"
                    continue

                if flag == "node":
                    key = int(re.findall("key=(.*?), b", line)[0])
                    bias = float(re.findall("bias=(.*?), r", line)[0])
                    weight = float(re.findall('response=(.*?), ac', line)[0])
                    activation = re.findall("activation=(.*?), ag", line)[0]
                    nodes[key] = Node(
                        key=key,
                        activation=activation_funcs.get(activation, None)
                    )

                if flag == "connection":
                    enabled = bool(re.findall("enabled=(.*)\)", line)[0])
                    if enabled:
                        weight = float(re.findall('weight=(.*?),', line)[0])
                        pre = (int(re.findall("key=\((.*?), ", line)[0]))
                        post = (int(re.findall("key=.*, (.*?)\), w", line)[0]))
                        nodes[pre].add_post(nodes[post], weight)

        return cls(list(nodes.values()))