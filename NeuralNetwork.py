import random
import numpy
import numpy.matlib

class NeuralNetwork:

#--------------------------------------------------------- Constructor ---------------------------------------------------------

    def __init__(self, input, hidden_list, output):
        #Memorize Size of Neural Network
        self.input = input
        self.output = output
        self.hidden = []
        for hidden in hidden_list:
            self.hidden.append(hidden)


        #Generate weights
        #Input weights
        self.input_weights = numpy.matlib.rand(self.hidden[0], self.input)
        self.input_weights = ((self.input_weights * 2) - 1) * 2

        #Hidden weights
        self.hidden_weights = []
        for i in range(len(self.hidden) - 1):
            hidden_weight = numpy.matlib.rand(self.hidden[i+1], self.hidden[i])
            hidden_weight = ((hidden_weight * 2) - 1) * 2

            self.hidden_weights.append(hidden_weight)

        #Output weights
        self.output_weights = numpy.matlib.rand(self.output, self.hidden[-1])
        self.output_weights = ((self.output_weights * 2) - 1) * 2


        #Generate bias
        #Hidden bias
        self.hidden_bias = []
        for i in range(len(self.hidden)):
            hidden_bia = numpy.matlib.rand(self.hidden[i], 1)
            hidden_bia = ((hidden_bia * 2) - 1) * 2

            self.hidden_bias.append(hidden_bia)

        #Output bias
        self.output_bias = numpy.matlib.rand(self.output, 1)
        self.output_bias = ((self.output_bias * 2) - 1) * 2

#--------------------------------------------------------- Neural Network ---------------------------------------------------------

    #Activation Function
    def activation(self, x):
        return 1/(1 + numpy.exp(-x))
        #return x


    #Feed Forward and return result
    def feedForward(self, inputs):

        #Input
        inputs_matrix = numpy.matrix(inputs).transpose()

        inputs_result = numpy.dot(self.input_weights, inputs_matrix)
        inputs_result = numpy.add(inputs_result, self.hidden_bias[0])

        #inputs_result = self.activation(inputs_result)
        inputs_result = numpy.vectorize(self.activation)(inputs_result)

        #Hidden
        hidden_result = inputs_result
        for i in range(1, len(self.hidden)):
            hidden_result = numpy.dot(self.hidden_weights[i-1], hidden_result)
            hidden_result = numpy.add(hidden_result, self.hidden_bias[i])

            #hidden_result = self.activation(hidden_result)
            hidden_result = numpy.vectorize(self.activation)(hidden_result)


        #Output
        output_result = numpy.dot(self.output_weights, hidden_result)
        output_result = numpy.add(output_result, self.output_bias)

        #output_result = self.activation(output_result)
        output_result = numpy.vectorize(self.activation)(output_result)

        return output_result

#--------------------------------------------------------- Evolution ---------------------------------------------------------
    
    def copy(self):
        #Create a new Neural Network of same size
        new_NN = NeuralNetwork(self.input, self.hidden.copy(), self.output)

        #Copy weights
        new_NN.input_weights = self.input_weights.copy()
        for i in range(len(self.hidden_weights)):
            new_NN.hidden_weights[i] = self.hidden_weights[i].copy()
        new_NN.output_weights = self.output_weights.copy()

        #Copy bias
        for i in range(len(self.hidden_weights)):
            new_NN.hidden_bias[i] = self.hidden_bias[i].copy()
        new_NN.output_bias = self.output_bias.copy()

        #Return new Neural Network
        return new_NN


    #Mutate the Neural Network
    def mutate(self, mutation_rate, mutation_scale):
        #Mutate weights
        self.input_weights = numpy.vectorize(mutate_value)(self.input_weights, mutation_rate, mutation_scale)
        for i in range(len(self.hidden_weights)):
            self.hidden_weights[i] = numpy.vectorize(mutate_value)(self.hidden_weights[i], mutation_rate, mutation_scale)
        self.output_weights = numpy.vectorize(mutate_value)(self.output_weights, mutation_rate, mutation_scale)

        #Mutate bias
        for i in range(len(self.hidden_weights)):
            self.hidden_bias[i] = numpy.vectorize(mutate_value)(self.hidden_bias[i], mutation_rate, mutation_scale)
        self.output_bias = numpy.vectorize(mutate_value)(self.output_bias, mutation_rate, mutation_scale)


    #Reproduce the Neural Network
    def reproduce(self, mutation_rate = 0.1, mutation_scale = 0.1):
        #Copy this Neural Network
        child = self.copy()

        #Mutate new Neural Network
        child.mutate(mutation_rate, mutation_scale)

        #Return new Neural Network
        return child
    

    #Crossover the Neural Network with another Neural Network
    def crossover(self, other, selection_rate = 0.7, mutation_rate = 0.1, mutation_scale = 0.1):
        #Create a new Neural Network of same size
        child = NeuralNetwork(self.input, self.hidden, self.output)

        #Copy weights from one of both parents randomly
        child.input_weights = numpy.vectorize(random_selection)(self.input_weights, other.input_weights, selection_rate)
        for i in range(len(self.hidden_weights)):
            child.hidden_weights[i] = numpy.vectorize(random_selection)(self.hidden_weights[i], other.hidden_weights[i], selection_rate)
        child.output_weights = numpy.vectorize(random_selection)(self.output_weights, other.output_weights, selection_rate)

        #Copy bias from one of both parents randomly
        for i in range(len(self.hidden_weights)):
            child.hidden_bias[i] = numpy.vectorize(random_selection)(self.hidden_bias[i], other.hidden_bias[i], selection_rate)
        child.output_bias = numpy.vectorize(random_selection)(self.output_bias, other.output_bias, selection_rate)

        #Mutate new Neural Network
        child.mutate(mutation_rate, mutation_scale)

        #Return new Neural Network
        return child

#--------------------------------------------------------- Util ---------------------------------------------------------

#Randomly decide and mutate given value
def mutate_value(value, mutation_rate, mutation_scale):
    #Randomly decide
    if random.random() < mutation_rate:
        #Randomly mutate
        mutation =  numpy.random.normal(0, mutation_scale)
        return value + mutation
    else:
        #Return original value
        return value


#Randomly select wich of the two given value will be returned
def random_selection(value, other_value, selection_rate):
    #Randomly decide
    if random.random() < selection_rate:
        #Return original value
        return value
    else:
        #Return other value
        return other_value