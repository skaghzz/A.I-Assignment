from util import *
import sys
import matplotlib.pyplot as plt
plt.ion()

def InitNN(num_inputs, num_hiddens, num_outputs):
    """Initializes NN parameters."""
    W1 = 0.01 * np.random.randn(num_inputs, num_hiddens)
    W2 = 0.01 * np.random.randn(num_hiddens, num_outputs)
    b1 = np.zeros((num_hiddens, 1))
    b2 = np.zeros((num_outputs, 1))
    return W1, W2, b1, b2

def TrainNN(num_hiddens, eps, momentum, num_epochs):
    """Trains a single hidden layer NN.
    Inputs:
        num_hiddens: NUmber of hidden units.
        eps: Learning rate.
        momentum: Momentum.
        num_epochs: Number of epochs to run training for.

    Returns:
        W1: First layer weights.
        W2: Second layer weights.
        b1: Hidden layer bias.
        b2: Output layer bias.
        train_error: Training error at at epoch.
        valid_error: Validation error at at epoch.
    """

    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
    W1, W2, b1, b2 = InitNN(inputs_train.shape[0], num_hiddens, target_train.shape[0])
    dW1 = np.zeros(W1.shape)
    dW2 = np.zeros(W2.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    train_error = []
    valid_error = []
    train_correct = []
    valid_correct = []
    num_train_cases = inputs_train.shape[1]
    for epoch in xrange(num_epochs):
        # Forward prop
        h_input = np.dot(W1.T, inputs_train) + b1  # Input to hidden layer.
        h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
        logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
        prediction = 1 / (1 + np.exp(-logit))  # Output prediction.

        # Compute cross entropy
        train_CE = -np.mean(target_train * np.log(prediction) + (1 - target_train) * np.log(1 - prediction))

        # Count correct classification
        encodedPrediction = onehotEncoding(prediction)
        # 1,1 / 0,0 => CE : 0
        # 0,1 / 1,0 => CE : infinite
        cntTrainCorrect = 0
        cntTrainCorrectTotal = encodedPrediction.shape[1]
        for i in range(cntTrainCorrectTotal):
            if (encodedPrediction[0][i] == target_train[0][i]):
                cntTrainCorrect = cntTrainCorrect + 1
        trainCorrectnessPercent = (cntTrainCorrect / float(cntTrainCorrectTotal)) * 100

        # Compute deriv
        dEbydlogit = prediction - target_train

        # Backprop
        dEbydh_output = np.dot(W2, dEbydlogit)
        dEbydh_input = dEbydh_output * h_output * (1 - h_output)

        # Gradients for weights and biases.
        dEbydW2 = np.dot(h_output, dEbydlogit.T)
        dEbydb2 = np.sum(dEbydlogit, axis=1).reshape(-1, 1)
        dEbydW1 = np.dot(inputs_train, dEbydh_input.T)
        dEbydb1 = np.sum(dEbydh_input, axis=1).reshape(-1, 1)

        #%%%% Update the weights at the end of the epoch %%%%%%
        dW1 = momentum * dW1 - (eps / num_train_cases) * dEbydW1
        dW2 = momentum * dW2 - (eps / num_train_cases) * dEbydW2
        db1 = momentum * db1 - (eps / num_train_cases) * dEbydb1
        db2 = momentum * db2 - (eps / num_train_cases) * dEbydb2

        W1 = W1 + dW1
        W2 = W2 + dW2
        b1 = b1 + db1
        b2 = b2 + db2

        valid_CE = Evaluate(inputs_valid, target_valid, W1, W2, b1, b2)
        cntValidCorrect, cntValidTotal = evaluateCorrect(inputs_valid, target_valid, W1, W2, b1, b2)
        validCorrectnessPercent = (cntValidCorrect / float(cntValidTotal)) * 100

        train_error.append(train_CE)
        valid_error.append(valid_CE)
        train_correct.append(trainCorrectnessPercent)
        valid_correct.append(validCorrectnessPercent)
        sys.stdout.write('\rStep %d Train CE %.5f, Validation CE %.5f' % (epoch, train_CE, valid_CE))
        sys.stdout.flush()
        if (epoch % 100 == 0):
            sys.stdout.write('\n')

    sys.stdout.write('\n')
    final_train_error = Evaluate(inputs_train, target_train, W1, W2, b1, b2)
    final_valid_error = Evaluate(inputs_valid, target_valid, W1, W2, b1, b2)
    final_test_error = Evaluate(inputs_test, target_test, W1, W2, b1, b2)
    print 'Error: Train %.5f, Validation %.5f, Test %.5f' % (final_train_error, final_valid_error, final_test_error)
    return W1, W2, b1, b2, train_error, valid_error,train_correct, valid_correct

def Evaluate(inputs, target, W1, W2, b1, b2):
    """Evaluates the model on inputs and target."""
    h_input = np.dot(W1.T, inputs) + b1  # Input to hidden layer.
    h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
    logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
    prediction = 1 / (1 + np.exp(-logit))  # Output prediction.
    CE = -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))
    return CE

def onehotEncoding(prediction):
    endcodedPrediction = np.zeros(prediction.shape)
    for i in range(prediction.shape[1]):
        if(prediction[0][i] >= 0.5):
            endcodedPrediction[0][i] = 1
        else:
            endcodedPrediction[0][i] = 0
    return endcodedPrediction

def evaluateCorrect(inputs, target, W1, W2, b1, b2):
    h_input = np.dot(W1.T, inputs) + b1  # Input to hidden layer.
    h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
    logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
    prediction = 1 / (1 + np.exp(-logit))  # Output prediction.\
    encodedPrediction = onehotEncoding(prediction)
    # 1,1 / 0,0 => CE : 0
    # 0,1 / 1,0 => CE : infinite
    cntCorrect = 0
    cntCorrectTotal = encodedPrediction.shape[1]
    for i in range(cntCorrectTotal):
        if(encodedPrediction[0][i] == target[0][i]):
            cntCorrect = cntCorrect + 1
    return cntCorrect, cntCorrectTotal

def DisplayErrorPlot(train_error, valid_error):
    fig = plt.figure(1)
    plt.clf()
    plt.plot(range(len(train_error)), train_error, 'b', label='Train')
    plt.plot(range(len(valid_error)), valid_error, 'g', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Cross entropy')
    plt.legend()
    plt.draw()
    fig.savefig("CrossEntropy.svg", dpi=fig.dpi)
    #raw_input('Press Enter to exit.')

def DisplayClassificationErrorPlot(train_correct,valid_correct):
    fig = plt.figure(2)
    plt.clf()
    plt.plot(range(len(train_correct)), np.subtract([100.0,],train_correct), 'b', label='Train')
    plt.plot(range(len(valid_correct)), np.subtract([100.0,],valid_correct), 'g', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Classification error %')
    plt.legend()
    plt.draw()
    fig.savefig("classificationError.svg", dpi=fig.dpi)

def DisplayClassificationCorrectPlot(train_correct,valid_correct):
    fig = plt.figure(2)
    plt.clf()
    plt.plot(range(len(train_correct)), train_correct, 'b', label='Train')
    plt.plot(range(len(valid_correct)), valid_correct, 'g', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Classification Correct %')
    plt.legend()
    plt.draw()
    fig.savefig("Correctness.svg", dpi=fig.dpi)

def SaveModel(modelfile, W1, W2, b1, b2, train_error, valid_error):
    """Saves the model to a numpy file."""
    model = {'W1': W1, 'W2' : W2, 'b1' : b1, 'b2' : b2,
             'train_error' : train_error, 'valid_error' : valid_error}
    print 'Writing model to %s' % modelfile
    np.savez(modelfile, **model)

def LoadModel(modelfile):
    """Loads model from numpy file."""
    model = np.load(modelfile)
    return model['W1'], model['W2'], model['b1'], model['b2'], model['train_error'], model['valid_error']

def main():
    num_hiddens = 100
    eps = 0.02
    momentum = 0.5
    num_epochs = 1500
    W1, W2, b1, b2, train_error, valid_error, train_correct, valid_correct = TrainNN(num_hiddens, eps, momentum, num_epochs)
    DisplayErrorPlot(train_error, valid_error)
    DisplayClassificationErrorPlot(train_correct, valid_correct)
    DisplayClassificationCorrectPlot(train_correct, valid_correct)
    # If you wish to save the model for future use :
    # outputfile = 'model.npz'
    # SaveModel(outputfile, W1, W2, b1, b2, train_error, valid_error)

if __name__ == '__main__':
    main()
