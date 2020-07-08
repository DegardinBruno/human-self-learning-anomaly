from scipy import stats
import numpy as np, os
import matplotlib.pyplot as plt

std_deviation = [ [1, -1], [1, 2], [-1, -2], [2, 3], [-2, -3], [3, 10], [-3, -10]]
fill_color    = [ ['#0b559f', '#2b7bba', '#2b7bba', '#539ecd', '#539ecd', '#89bedc', '#89bedc'] , ['#B10102', '#c03334', '#c03334', '#c84d4d', '#c84d4d', '#d06667', '#d06667'] ]


def predict_prob(kernel_0, kernel_1, score, prob0, prob1):
    prediction = (prob1 * kernel_1(score))/(prob0*kernel_0(score)+prob1*kernel_1(score))
    return [1-prediction[0], prediction[0]]


def gaussian_kde(y_train, X_train, destiny, aton_iteration, X_test=None):


    scores0 = [x for i,x in enumerate(X_train) if y_train[i]==0]
    scores1 = [x for i,x in enumerate(X_train) if y_train[i]==1]

    prob0 = 0.5  # Posteriors
    prob1 = 0.5  # Posteriors

    kernel0 = stats.gaussian_kde(scores0)                 # Obtain the kernels for negative instances
    kernel1 = stats.gaussian_kde(scores1)                 # Obtain the kernels for positive instances


    kernel0.set_bandwidth(bw_method=kernel0.factor / 3.)  # Set bandwidth to adjust more or less the kernels, Scott's rule bandwidth factor employed
    kernel1.set_bandwidth(bw_method=kernel1.factor / 3.)

    predictions_plot = np.array([predict_prob(kernel0, kernel1, score, prob0, prob1) for score in np.arange(0,1,0.01)])                # Visualization of the probabilities
    predictions = np.array([predict_prob(kernel0, kernel1, score, prob0, prob1) for score in X_test]) if X_test is not None else None  # Estimate the probabilities

    # Plot probabilities
    plt.figure(1)
    plt.clf()
    plt.plot(np.arange(0, 1, 0.01), predictions_plot[:,0], color=fill_color[0][0], linewidth=2.5, alpha=0.7, label='Normal Max: ' + str(round(max(predictions_plot[:,0]), 3)))
    plt.plot(np.arange(0, 1, 0.01), predictions_plot[:,1], color=fill_color[1][0], linewidth=2.5, alpha=0.7, label='Fight Max: ' + str(round(max(predictions_plot[:,1]), 3)))
    plt.grid()
    plt.title('Bayesian Probability', fontsize=10)
    plt.xlabel('Scores')
    plt.ylabel('Probabilities')
    plt.tick_params(axis='both', labelsize=20)
    plt.ylim(-0.04,1.04)
    plt.savefig(os.path.join('results',destiny, 'VAL','bayesian_probability_'+str(aton_iteration)+'.pdf'))

    # Plot Kernels
    plt.figure(2)
    plt.clf()
    plt.plot(kernel0(np.arange(0,1,0.01)),color='blue')
    plt.plot(kernel1(np.arange(0,1,0.01)),color='red')
    plt.title('Gaussian Kernel Density Estimation', fontsize=10)
    plt.xlabel('Scores')
    plt.ylabel('Distribution')
    plt.savefig(os.path.join('results', destiny, 'VAL', 'gaussKernel_'+str(aton_iteration)+'.pdf'))

    return np.array([0,1]), predictions if X_test is not None else None, (max(predictions[:,0]) // 0.001 / 1000) if X_test is not None else None, (max(predictions[:,1]) // 0.001 / 1000) if X_test is not None else None


def histogram(y_train, X_train, destiny, aton_iteration):
    scores0 = [x for i,x in enumerate(X_train) if y_train[i]==0]
    scores1 = [x for i,x in enumerate(X_train) if y_train[i]==1]

    fig, ax = plt.subplots()

    freqs_0, edges_0 = np.histogram(scores0, 100)
    freqs_0 = np.divide(freqs_0, np.sum(freqs_0))
    bins_0 = (edges_0[:-1] + edges_0[1:]) / 2
    ax.bar(bins_0, freqs_0, width=bins_0[1] - bins_0[0], align='center', alpha=0.5, color='blue', label = str(len(scores0)) + ' Scores for 0\'s')


    freqs_1, edges_1 = np.histogram(scores1, 100)
    freqs_1 = np.divide(freqs_1, np.sum(freqs_1))
    bins_1 = (edges_1[:-1] + edges_1[1:]) / 2
    ax.bar(bins_1, freqs_1, width=bins_1[1] - bins_1[0], align='center', alpha=0.5, color='red', label = str(len(scores1)) + ' Scores for 1\'s')
    plt.ylim( (pow(10,-4),pow(10,0)) )
    ax.tick_params(axis='both', labelsize=20)
    ax.set_yscale('log')
    #ax.legend()

    plt.savefig(os.path.join('results', destiny, 'hist_'+str(aton_iteration)+'.pdf'))


