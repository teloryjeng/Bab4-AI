from numpy import exp, array, random, dot

# Tabel kebenaran OR: Output 0 jika [0,0], selain itu bernilai 1
inputs = array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = array([[0, 1, 1, 1]]).T

random.seed(1)
weights = 2 * random.random((2, 1)) - 1

def neuron(inputs, weights):
    return 1 / (1 + exp(-(dot(inputs, weights))))

for iteration in range(50000):
    output = neuron(inputs, weights)
    weights += dot(inputs.T, (outputs - output) * output * (1 - output))

x = array([0, 1])
print(neuron(x, weights))