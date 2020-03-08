
from collections import Counter

class HuffmanTreeNode(object):
    def __init__(self, probability, value=None, left=None, right=None):
        self.probability = probability
        self.value = value
        self.left = left
        self.right = right

    def generate_codes(self, previous_code=''):
        """Generates a huffman code.

        The dictionary keys are the encoded values, while the
        dictionary values are the Huffman codes themselves.
        This makes compression easier, but could be done better.
        """
        # Can a node have one child? Or always either two or zero?
        if self.left is None and self.right is None:
            yield (self.value, previous_code)
        else:
            yield from self.left.generate_codes(previous_code + '0')
            yield from self.right.generate_codes(previous_code + '1')

    def __repr__(self):
        return '{}(value={})'.format(self.__class__.__name__, self.value)


def huffman_encode(data):
    nodes = []
    for value, count in Counter(data).items():
        probability = count / len(data)
        nodes.append(HuffmanTreeNode(probability, value))

    while len(nodes) > 1:
        nodes.sort(reverse=True, key=lambda node: node.probability)
        child_left = nodes.pop()
        child_right = nodes.pop()
        internal_node = HuffmanTreeNode(
            probability=(child_left.probability + child_right.probability),
            left=child_left,
            right=child_right,
        )
        nodes.append(internal_node)

    tree = nodes[0]
    huffman_code = dict(tree.generate_codes())
    huffman_encoded_data = [huffman_code[value] for value in data]
    return huffman_code, huffman_encoded_data
