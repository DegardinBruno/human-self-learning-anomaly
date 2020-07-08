import matplotlib.pyplot as plt
import numpy as np, os
from scipy.io import loadmat


def plot_AUC(source, aton_iteration):  # Plot ROC
    auc = loadmat(os.path.join('results', source, str(aton_iteration), 'eval_AUC_'+str(aton_iteration)+'.mat'))

    plt.figure(100)
    plt.clf()
    plt.plot(np.concatenate(auc['X']), np.concatenate(auc['Y']), color='blue', linewidth=3.5)
    plt.legend(['AUC: ' + str(round(np.concatenate(auc['AUC'])[0], 4) * 100) + '%'])
    plt.savefig(os.path.join('results', source, 'AUC_'+str(aton_iteration)+'.pdf'))


def stats_batch(batch_loss, aton_iteration):  # Plot training

    plt.clf()

    plt.text(0.5,1,'WS/SS Epoch ' + str(aton_iteration))

    plt.title('Train Loss')
    plt.plot(batch_loss, color='#b83232', label='Loss: ' + str(round(batch_loss[-1],4)))
    plt.legend(loc='upper right')


    plt.pause(0.005)


