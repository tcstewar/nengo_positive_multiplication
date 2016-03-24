import nengo
import numpy as np
import ctn_benchmark

class MultiplicativeNetwork(nengo.Network):
    def __init__(self, exponents, epsilon=0.001, n_neurons=200, result_dist=nengo.dists.Uniform(0,1)):
        super(MultiplicativeNetwork, self).__init__()
        self.exponents = np.array(exponents, dtype=float)
        self.w = self.exponents / np.sum(np.abs(self.exponents))
        self.epsilon = epsilon
        
        with self:
            # when connecting to this input, you must apply self.encode_func to transform the data
            self.input = nengo.Node(None, size_in=len(exponents), label='input')
            
            # this is the summed values (hidden layer 2 in the original formulation)
            self.total = nengo.Ensemble(n_neurons=n_neurons, dimensions=1)
            
            # this is the decoded value (output node in the original formulation)
            self.output = nengo.Node(None, size_in=1)
            
            # scale the inputs by w and sum them into the total population
            nengo.Connection(self.input, self.total, transform=[self.w], synapse=None)
            # apply the bias
            self.bias = nengo.Node([np.sum(self.w)-1], label='bias')
            nengo.Connection(self.bias, self.total, synapse=None)
            
            # decode back out to the original 0-1 range
            nengo.Connection(self.total, self.output, function=self.decode_func, synapse=None)
        
        # when computing the output decoder, optimize for data in the given range
        # (note that result_dist is the expected distribution of output values)
        self.total.eval_points = self.encode_func_scaled(result_dist.sample(1000, d=1))
        
        
    # the mapping from total back to the original 0-1 range
    def decode_func(self, x):
        return self.epsilon ** (self.ialpha(x)*np.sum(np.abs(self.exponents)))
    
    # the transformation for a single input
    def encode_func(self, x):
        x = np.maximum(x, self.epsilon)
        return self.alpha(np.log(x)/np.log(self.epsilon))
    
    # the inverse of the decode_func
    def encode_func_scaled(self, x):
        x = np.maximum(x, self.epsilon)
        return self.alpha(np.log(x)/np.log(self.epsilon)/np.sum(np.abs(self.exponents)))
        
    def alpha(self, x):
        a = self.epsilon
        b = 1
        return (x - (a+b)/2)*2/(b-a)
    def ialpha(self, x):
        a = self.epsilon
        b = 1
        return (x/2*(b-a))+(a+b)/2        


class PosMult(ctn_benchmark.Benchmark):
    def params(self):
        self.default('number of input values', D=3)
        self.default('number of neurons', n_neurons=200)
        self.default('number of input neurons', n_neurons_input=0)
        self.default('number of output neurons', n_neurons_output=0)
        self.default('epsilon', epsilon=0.001)
        self.default('number of samples', n_samples=40)
        self.default('time per sample', T_sample=0.3)
        self.default('synapse', synapse=0.03)
        self.default('probe synapse', probe_synapse=0.03)

    def model(self, p):
        samples = np.random.uniform(0,1, size=(p.n_samples, p.D))
        self.exponents = np.ones(p.D)

        model = nengo.Network()
        with model:
            stim = nengo.Node(lambda t: samples[int(t / p.T_sample) % p.n_samples])
            self.p_stim = nengo.Probe(stim, synapse=None)
            
            mult = MultiplicativeNetwork(self.exponents,
                                         epsilon=p.epsilon,
                                         n_neurons=p.n_neurons)
            for c in mult.connections:
                if c.post_obj is mult.output:
                    self.mult_output_conn = c

            if p.n_neurons_input == 0:
                nengo.Connection(stim, mult.input, function=mult.encode_func,
                                 synapse=None)
            else:
                inputs = nengo.networks.EnsembleArray(n_neurons=p.n_neurons_input,
                        n_ensembles=p.D)
                inputs.add_output('encode', mult.encode_func)
                nengo.Connection(stim, inputs.input, synapse=None)
                nengo.Connection(inputs.encode, mult.input, synapse=p.synapse)

            if p.n_neurons_output == 0:
                self.p_output = nengo.Probe(mult.output, synapse=p.probe_synapse)
            else:
                output = nengo.Ensemble(n_neurons=p.n_neurons_output,
                                        dimensions=1)
                nengo.Connection(mult.output, output, synapse=p.synapse)
                self.p_output = nengo.Probe(output, synapse=p.probe_synapse)

        return model

    def evaluate(self, p, sim, plt):
        T = p.n_samples * p.T_sample
        sim.run(T)
        self.record_speed(T)

        rmse_decoder = np.mean(sim.data[self.mult_output_conn].solver_info['rmses'])

        ideal = np.ones_like(sim.trange())
        for i in range(p.D):
            ideal *= sim.data[self.p_stim][:,i]**self.exponents[i]
        if p.n_neurons_input > 0:
            ideal = nengo.synapses.filt(ideal,nengo.synapses.Lowpass(p.synapse), dt=sim.dt)    
        if p.n_neurons_output > 0:
            ideal = nengo.synapses.filt(ideal,nengo.synapses.Lowpass(p.synapse), dt=sim.dt)    
        ideal = nengo.synapses.filt(ideal,nengo.synapses.Lowpass(p.probe_synapse), dt=sim.dt)    
            
        if plt is not None:
            plt.plot(sim.trange(), sim.data[self.p_output])
            plt.plot(sim.trange(), ideal)    

        rmse = np.sqrt(np.mean((sim.data[self.p_output][:,0]-ideal)**2))
        return dict(rmse=rmse,
                    rmse_decoder=rmse_decoder)

if __name__ == '__main__':
    PosMult().run()

