__author__ = 'JinHoon'

"""
    This file contains a RNN benchmark for a standard RNN with tap -1 without minibatches.
    It uses a softmax output layer. It can be used to compare THEANO with another
    RNN code snippet.

    This version can have different sequence lengths. You can access them via a special
    index table.

    data format:

        input  ...  matrix:[N][frame]
        output ...  vector:[target1|...|targetN]

        access a input sequence N via the indexTable idx
           input[idx['input'][N]:idx['input'][N+1]]

        access a target (output) N via the indexTable idx
            target[idx['target'][N]:idx['target'][N+1]]

    NOTE:
        - Please take care that you only compare equal networks with equal datasets.
        - this version works with taps [-N,...,-1]
        - if you want to use tap [-1] please init with [-1,0], due to a shape mismatch
        - if you want to use tap [-2,-1] please init with [-2,-1]
"""

import sys
import os
import time
import numpy
import theano
import theano.tensor as T
import glob

#---------------------------------------------------------------------------------
class RNN(object):

    #---------------------------------------------------------------------------------
    def __init__(self, rng, output_taps, n_in, n_hidden, n_out, mode, profile, dtype=theano.config.floatX):
        """
            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights

            :type order: int32
            :param order: order of the RNN (used for higher order RNNs)

            :type n_in: int32
            :param n_in: number of input neurons

            :type n_hidden: int32
            :param n_hidden: number of hidden units

            :type dtype: theano.config.floatX
            :param dtype: theano 32/64bit mode
        """

        #--------------------------------------------------------------------
        # PROBLEM section (needs review)
        #--------------------------------------------------------------------
        """
         here the the recurrent delay elements (taps) will be created
         You can change the line below to add special delays like [-4,-1]

         Problem: single delays like tap = [-1] lead to a T.matrix() generation
                  in step() line:117. A theano shape error will be generated.
                  Hence we have to create at least a tuple [-1,0] for order.1 RNNs.
                  Tap[0] will not be used in step (skipped)

         QuickFIX: tap = [-1,0] -> tap[0] won't be used
         Description: we have to create at least a tuple [-1,0] used in
                      outputs_info line:91
                      otherwise a T.matrix will be generated in step(),
                      leading to an theano shape error.
         Discuss: @theano-dev-group
        """

        # length of output taps
        self.len_output_taps = len(output_taps)
        # input over the time (1st dim is the time)
        self.x  = T.dmatrix('x')
        # target over the time (1st dim is the time)
        self.y  = T.ivector('y')
        # recurrent activations over the time (1st dim is the time)
        self.H = T.dmatrix()
        # learning rate
        self.lr = T.fscalar()

        # input to hidden layer weights
        W_in = theano.shared(numpy.asarray(rng.uniform(
                low  = - numpy.sqrt(6./(n_in+n_hidden)),
                high = numpy.sqrt(6./(n_in+n_hidden)),
                size = (n_in, n_hidden)),
                dtype = dtype), name='W_in')
        b_in = theano.shared(numpy.zeros((n_hidden,), dtype=dtype), name='b_in')

        # recurrent bias
        b_h = theano.shared(numpy.zeros((n_hidden,)).astype(dtype), name='b_h')
        # recurrent activations
        self.h = theano.shared(numpy.zeros((n_hidden, n_hidden), dtype=dtype), name='h')
        # recurrent weights as real values
        W = [theano.shared(numpy.asarray(rng.uniform(low  = - numpy.sqrt(6./(n_hidden+n_hidden)),\
                                    high = numpy.sqrt(6./(n_hidden+n_hidden)),\
                                    size = (n_hidden, n_hidden)), dtype = dtype),\
                                    name='W_r'+str(u)) for u in range(self.len_output_taps)]

        # hidden to output layer weights
        W_out = theano.shared(numpy.asarray(rng.uniform(
                low  = - numpy.sqrt(6./(n_hidden+n_out)),
                high = numpy.sqrt(6./(n_hidden+n_out)),
                size = (n_hidden, n_out)),
                dtype = dtype), name = 'W_out')
        b_out = theano.shared(numpy.zeros((n_out,), dtype=dtype), name='b_out')

        # stack the network parameters
        self.params = []
        self.params.extend(W)
        self.params.extend([b_h])
        self.params.extend([W_in, b_in])

        # this is the recursive BBTP loop
        # `self.H` are the recurrent activations, 'y_act_list' is the output for the
        # entire sequence over the time
        h, updates = theano.scan(fn = self.step, \
                             sequences = dict(input = self.x, taps = [0]), \
                             outputs_info = dict(initial = self.H, taps = output_taps), \
                             non_sequences = self.params,
                             mode=mode,
                             profile=profile)

        # softmax output signal
        self.y_act = T.nnet.softmax(T.dot(h, W_out) + b_out)

        # error between output and target
        self.cost = -T.mean(T.log(self.y_act)[T.arange(self.y.shape[0]), self.y])

        # add the output network params
        self.params.extend([W_out, b_out])

    def step(self, u_t, *args):
            """
                step function to calculate BPTT

                type u_t: T.matrix()
                param u_t: input sequence of the network

                type * args: python parameter list
                param * args: this is needed to implement a more general model of the step function
                             see theano@users: http: // groups.google.com / group / theano - users / \
                             browse_thread / thread / 2fa44792c9cdd0d5

            """

            # get the recurrent activations
            r_act_vals = [args[u] for u in range(self.len_output_taps)]

            # get the recurrent weights
            r_weights = [args[u] for u in range(self.len_output_taps, (self.len_output_taps) * 2)]

            # get the input/output weights
            b_h = args[self.len_output_taps * 2]
            W_in = args[self.len_output_taps * 2 + 1]
            b_in = args[self.len_output_taps * 2 + 2]

            # sum up the recurrent activations
            act = theano.dot(r_act_vals[0], r_weights[0]) + b_h
            for u in range(1, self.len_output_taps):
                act += T.dot(r_act_vals[u], r_weights[u]) + b_h

            # compute the new recurrent activation
            h_t = T.tanh(T.dot(u_t, W_in) + b_in + act)

            return h_t

    #---------------------------------------------------------------------------------
    def build_finetune_functions(self, train_set_x, train_set_y, learning_rate, mode, profile):
        """
            type train_set_x: T.matrix()
            param train_set_x: training input sequences of the network

            type train_set_y: T.ivector()
            param train_set_y: training output sequences of the network

            type learning_rate: float32
            param learning_rate: learning_rate of the training algorithm

            type mode: str
            param mode: theano function compile mode

        """

        #-----------------------------------------
        # THEANO variables for the data access
        #-----------------------------------------
        # we specify direct indizes here, since this allows to have different seq. lengths
        i_idx_0 = T.iscalar('input_start') # index to input start
        i_idx_1 = T.iscalar('input_stop') # index to input stop
        t_idx_0 = T.iscalar('target_start') # index to target start
        t_idx_1 = T.iscalar('target_stop') # index to target stop

        print('Compiling')

        #-----------------------------------------
        # THEANO train function
        #-----------------------------------------
        gparams = []
        for param in self.params:
            gparam = T.grad(self.cost, param)
            gparams.append(gparam)

        # specify how to update the parameters of the model as a dictionary
        updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(self.params, gparams)
        ]

        # define the train function
        t_inputs = [i_idx_0] + [i_idx_1] + [t_idx_0] + [t_idx_1]
        train_fn = theano.function(inputs = t_inputs,
                outputs = self.cost,
                updates = updates,
                givens={self.x:train_set_x[i_idx_0:i_idx_1],
                        self.y:train_set_y[t_idx_0:t_idx_1],
                        self.H:T.cast(self.h,'float64'),
                        self.lr:T.cast(learning_rate,'float64')},
                mode = mode,
                profile = profile)

        return train_fn


#---------------------------------------------------------------------------------
class Engine(object):

    def __init__(self,
                learning_rate=0.01,
                n_epochs=20,
                output_taps=[-1, 0]):

        #-----------------------------------------
        # BENCHMARK SETUP
        #-----------------------------------------
        """
            Please note that if you are comparing different RNNs (C/C++/TORCH/...)
            the data & network topology and parameters should be at least the same, to
            be fair.

            f.e. samples, batchsize, learning_rate, epochs, #neurons, ... (other obvious stuff)
        """

        #-----------------------------------------
        # THEANO SETUP
        #-----------------------------------------
        # setup mode
        mode = theano.Mode(linker='cvm')
        # setup profile
        profile = 0

        #-----------------------------------------
        # MODEL SETUP
        #-----------------------------------------
        dataset = glob.glob('../Data/smallHybrid/*')
        N = 500  # number of samples
        n_in = 9460 # number of inumpyut units
        n_hidden = 1000# number of hidden units
        n_out = 129 # number of output units
        n_symbol = 43
        length = 5 #sequence length
        batch_size = 500 # batch_size
        print('network: n_in:', n_in, 'n_hidden:', n_hidden, 'n_out:', n_out, 'output:softmax')
        print('data: samples:', N, 'batch_size:', batch_size, 'sequence length', length)

        #load Data
        def preprocessData(path):
            files = []
            for file in path:
                files.append(file)
            files.sort()

            label = numpy.zeros((N, n_symbol))
            feature = numpy.zeros((N, n_in))
            index = 0
            col_index = 0

            for file in files:
                binary = numpy.fromfile(file, dtype='float64')
                numRow=binary[0]
                numCol=binary[1]

                print("Num Row", numRow)
                print("Num Col", numCol)
                binary=numpy.delete(binary,[0,1])
                binary=binary.reshape((numRow,numCol))
                binary = binary[:N]

                label[:,index] = binary[:,0]
                feature[:, col_index:col_index+numCol-1] = binary[:, 1:]

                col_index += numCol-1
                index += 1

            feature = feature[:,:col_index]
            label = label + 1
            label = label[:,0]

            print("label", label.shape)
            print("feature", feature.shape)

            # label = label.squeeze()
            # feature = feature.squeeze()

            print("label", label.shape)
            print("feature", feature.shape)
            label = label.astype('int32')
            feature = feature.astype('float64')

            return feature, label

        def trainTestSplit(feature, label):
            n_train = int(0.6 * N)
            n_valid = int(0.2 * N)

            x_train = feature[:n_train]
            y_train = label[:n_train]

            x_valid = feature[n_train: n_train + n_valid]
            y_valid = label[n_train: n_train + n_valid]

            x_test = feature[n_train + n_valid:]
            y_test = label[n_train + n_valid:]

            train_set = (x_train, y_train)
            valid_set = (x_valid, y_valid)
            test_set = (x_test, y_test)

            return train_set, valid_set, test_set
            #return x_train, y_train, x_valid, y_valid, x_test, y_test

        def shared_dataset(data_xy, borrow=True):
            """ Function that loads the dataset into shared variables
            The reason we store our dataset in shared variables is to allow
            Theano to copy it into the GPU memory (when code is run on GPU).
            Since copying data into the GPU is slow, copying a minibatch everytime
            is needed (the default behaviour if the data is not in a shared
            variable) would lead to a large decrease in performance.
            """
            data_x, data_y = data_xy

            shared_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)


            # When storing data on the GPU it has to be stored as floats
            # therefore we will store the labels as ``floatX`` as well
            # (``shared_y`` does exactly that). But during our computations
            # we need them as ints (we use labels as index, and if they are
            # floats it doesn't make sense) therefore instead of returning
            # ``shared_y`` we will have to cast it to int. This little hack
            # lets ous get around this issue
            return shared_x, T.cast(shared_y, 'int32')

        # create data vectors
        # TODO: fix reshape, ValueError: ('setting an array element with a sequence.', 'Bad inumpyut argument to theano)
        #data_x = theano.shared(numpy.random.uniform(size=(N, minibatch, n_in)).astype(theano.config.floatX))
        #data_y = theano.shared(numpy.random.uniform(size=(N*minibatch)).astype(theano.config.floatX))
        # data_x = numpy.random.uniform(size=(N, minibatch, n_in)).astype(theano.config.floatX)
        # data_y = numpy.random.uniform(size=(N*minibatch)).astype('int32')

        feature, label = preprocessData(dataset)
        train_set, valid_set, test_set = trainTestSplit(feature, label)
        x_test, y_test = shared_dataset(test_set)
        x_valid, y_valid = shared_dataset(valid_set)
        x_train, y_train = shared_dataset(train_set)
        #x_train, y_train, x_valid, y_valid, x_test, y_test = trainTestSplit(feature, label)

        inputIndex = [ u*length*batch_size for u in range(N+1)]
        targetIndex = [ u*length*batch_size for u in range(N+1)]
        seq = { 'inputs':inputIndex, 'targets':targetIndex }
        #-----------------------------------------
        # RNN SETUP
        #-----------------------------------------
        # initialize random generator
        rng = numpy.random.RandomState(1234)
        # construct the CTC_RNN class
        classifier = RNN(rng=rng, output_taps=output_taps, n_in=n_in, n_hidden=n_hidden, n_out=n_out, mode=mode, profile=profile)
        # fetch the training function
        train_fn = classifier.build_finetune_functions(x_train, y_train, learning_rate, mode, profile)

        #-----------------------------------------
        # BENCHMARK START
        #-----------------------------------------
        # start the benchmark
        start_time = time.clock()
        print('Running', n_epochs)
        for _ in range(n_epochs) :
            for j in range(0,N,batch_size) :
                cost = train_fn(seq['inputs'][j], seq['inputs'][j+batch_size], seq['targets'][j], seq['targets'][j+batch_size])
                print("cost", cost)
        print >> sys.stderr, ('     training epoch time (%.5fm)' % ((time.clock()-start_time)/60.))
#---------------------------------------------------------------------------------
if __name__ == '__main__':

    Engine(n_epochs=2)