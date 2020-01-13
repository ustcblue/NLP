import numpy as np
import theano as theano
import theano.tensor as T
#from utils import *
import operator

class LSTMTheano:
    
    def __init__(self, pinyin_dim=100, character_dim=100, hidden_dim=100, bptt_truncate=-1):
        # Assign instance variables
        self.pinyin_dim = pinyin_dim
        self.character_dim = character_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        E = np.random.uniform(-np.sqrt(1./pinyin_dim), np.sqrt(1./pinyin_dim), (hidden_dim,pinyin_dim))
        
        Uf = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (8, hidden_dim, hidden_dim))
        Wf = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (8, hidden_dim, hidden_dim))
        Vf = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (character_dim, hidden_dim))

        Ub = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (8, hidden_dim, hidden_dim))
        Wb = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (8, hidden_dim, hidden_dim))
        Vb = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (character_dim, hidden_dim))
        
        # Theano: Created shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))

        self.Ub = theano.shared(name='Ub', value=Ub.astype(theano.config.floatX))
        self.Wb = theano.shared(name='Wb', value=Wb.astype(theano.config.floatX))      
        
        self.Uf = theano.shared(name='Uf', value=Uf.astype(theano.config.floatX))
        self.Wf = theano.shared(name='Wf', value=Wf.astype(theano.config.floatX))      
        
        self.Vf = theano.shared(name='Vf', value=Vf.astype(theano.config.floatX))
        self.Vb = theano.shared(name='Vb', value=Vb.astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    
    def __theano_build__(self):
        E, Ub, Vb, Vf, Wb, Uf, Wf = self.E, self.Ub, self.Vb, self.Vf, self.Wb, self.Uf, self.Wf
        x = T.ivector('x')
        y = T.ivector('y')
        
        def forward_prop_step(x_t, c_t_prev, c_t_prev_2, s_t_prev, s_t_prev_2, E, U, W):
            x_e = E[:,x_t]
            i = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t_prev))
            f = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t_prev))
            o = T.nnet.hard_sigmoid(U[2].dot(x_e) + W[2].dot(s_t_prev))

            g = T.tanh(U[3].dot(x_e) + W[3].dot(s_t_prev))

            c_t = c_t_prev * f + g * i
            s_t = T.tanh(c_t) * o
            
            i_2 = T.nnet.hard_sigmoid(U[4].dot(s_t) + W[0].dot(s_t_prev_2))
            f_2 = T.nnet.hard_sigmoid(U[5].dot(s_t) + W[1].dot(s_t_prev_2))
            o_2 = T.nnet.hard_sigmoid(U[6].dot(s_t) + W[2].dot(s_t_prev_2))

            g_2 = T.tanh(U[7].dot(s_t) + W[7].dot(s_t_prev_2))

            c_t_2 = c_t_prev_2 * f_2 + g_2 * i_2
            s_t_2 = T.tanh(c_t_2) * o_2

            return [c_t, c_t_2, s_t, s_t_2]

        [cf,cf_2,sf,sf_2], up = theano.scan(
            forward_prop_step,
            sequences=[ dict(input=x, taps = [0]) ],
            outputs_info=[dict(initial=T.zeros(self.hidden_dim), taps = [-1]), dict(initial=T.zeros(self.hidden_dim), taps = [-1]), dict(initial=T.zeros(self.hidden_dim), taps = [-1]), dict(initial=T.zeros(self.hidden_dim), taps = [-1])],
            non_sequences=[E, Uf, Wf],
            truncate_gradient=self.bptt_truncate,
            strict=True)
        
        def backward_prop_step(x_t, c_t_next, c_t_next_2, s_t_next, s_t_next_2, E, U, W):
            x_e = E[:,x_t]
            i = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t_next))
            f = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t_next))
            o = T.nnet.hard_sigmoid(U[2].dot(x_e) + W[2].dot(s_t_next))

            g = T.tanh(U[3].dot(x_e) + W[3].dot(s_t_next))

            c_t = c_t_next * f + g * i
            s_t = T.tanh(c_t) * o
            
            i_2 = T.nnet.hard_sigmoid(U[4].dot(s_t) + W[0].dot(s_t_next_2))
            f_2 = T.nnet.hard_sigmoid(U[5].dot(s_t) + W[1].dot(s_t_next_2))
            o_2 = T.nnet.hard_sigmoid(U[6].dot(s_t) + W[2].dot(s_t_next_2))

            g_2 = T.tanh(U[7].dot(s_t) + W[7].dot(s_t_next_2))

            c_t_2 = c_t_next_2 * f_2 + g_2 * i_2
            s_t_2 = T.tanh(c_t_2) * o_2

            return [c_t, c_t_2, s_t, s_t_2]

        [cb,cb_2,sb,sb_2], up = theano.scan(
            forward_prop_step,
            sequences=[ dict(input=x, taps = [0]) ],
            outputs_info=[dict(initial=T.zeros(self.hidden_dim), taps = [-1]), dict(initial=T.zeros(self.hidden_dim), taps = [-1]), dict(initial=T.zeros(self.hidden_dim), taps = [-1]), dict(initial=T.zeros(self.hidden_dim), taps = [-1])],
            non_sequences=[E, Ub, Wb],
            truncate_gradient=self.bptt_truncate,
            go_backwards=True,
            strict=True)
 
        o_c = (Vf.dot(sf_2.transpose()) + Vb.dot(sb_2.transpose())).transpose()

        o = T.nnet.softmax(o_c)
        
        prediction = T.argmax(o, axis=1)
        
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        # Gradients
        dUf = T.grad(o_error, Uf)
        dVf = T.grad(o_error, Vf)
        dWf = T.grad(o_error, Wf)
        dE = T.grad(o_error, E)
        dUb = T.grad(o_error, Ub)
        dWb = T.grad(o_error, Wb)
        dVb = T.grad(o_error, Vb)

        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], [prediction,o])
        self.ce_error = theano.function([x, y], o_error)
        #self.bptt = theano.function([x, y], [dUf, dV, dWf, dE])
        
        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x,y,learning_rate], [], 
                      updates=[(self.Ub, self.Ub - learning_rate * dUb),
                              (self.Vb, self.Vb - learning_rate * dVb),
                              (self.Wb, self.Wb - learning_rate * dWb),
                              (self.E, self.E - learning_rate * dE),
                              (self.Uf, self.Uf - learning_rate * dUf),
                              (self.Wf, self.Wf - learning_rate * dWf),
                              (self.Vf, self.Vf - learning_rate * dVf)])

    def calculate_total_loss(self, X, Y):
        '''
        idx = 0
        loss = []

        for x,y in zip(X,Y):
            l = self.ce_error(x,y)
            print "%d\t%f" % (idx, l)
            loss.append(l)
            idx += 1

        return np.sum(loss)
        '''
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)   

def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" % gradminus
                print "Estimated_gradient: %f" % estimated_gradient
                print "Backpropagation gradient: %f" % backprop_gradient
                print "Relative Error: %f" % relative_error
                return 
            it.iternext()
        print "Gradient check for parameter %s passed." % (pname)
