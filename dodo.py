import posmult
import ctn_benchmark

def task_vary_n_neurons():
    def run():
        for seed in range(9, 100):
            for D in [2,3,4]:
                for n_neurons in [50, 100, 200, 500, 1000]:
                    posmult.PosMult().run(seed=seed, D=D, n_neurons=n_neurons,
                            data_dir='vary_n_neurons')
    return dict(actions=[run], verbosity=2)

def task_plot_n_neurons():
    def plot():
        import pylab

        all_data = []
        for D in [2,3,4]:
            data = ctn_benchmark.Data('vary_n_neurons')

            for d in data.data[:]:
                if d['_D'] != D:
                    data.data.remove(d)
            all_data.append(data)

        plt = ctn_benchmark.Plot(all_data)
        plt.lines('_n_neurons', ['rmse'],plt=pylab)
        pylab.legend(['D=%d' % d for d in [2,3,4]])
        pylab.xscale('log')
        pylab.show()
    return dict(actions=[plot])



