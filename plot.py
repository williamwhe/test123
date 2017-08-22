import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def iterVsAttack( ):

    plt.figure(0, facecolor = "None")
    ax = plt.gca()
    iters = [] # iteration 
    atrs = [] # attack succes rate
    with open('out6/result.txt' , 'r') as fid:
    	rows = fid.read().strip().split('\n')
    	for row in rows:
    		tmps = row.split(',')
    		iteration = tmps[0].split(':')[1]
    		atr = tmps[1].split(': ')[1]
    		iters.append(iteration)
    		atrs.append(atr)

    plt.plot(iters , atrs, label = "\lambda 0.8" , linestyle='-', linewidth = 4)
       
    # plt.xlim([0.0, 20])
    # plt.ylim([0.0, 1.05])
    # plt.plot([0, len(base)], [0.5, 0.5], 'k--')
    plt.xlabel('Iter', fontsize = 15)
    plt.ylabel('Attack Rate', fontsize = 15)
    plt.title('')
    plt.grid(True)
    # print xtricks[idx]
    # ax.set_xticklabels(xtricks[idx], fontsize = 10)
    plt.legend(loc = "lower right", prop = {'size' : 12}, borderpad = 1,fancybox=True, framealpha = 0.5)
    plt.savefig('figures/iterVsATR.png')
    plt.show()

iterVsAttack()