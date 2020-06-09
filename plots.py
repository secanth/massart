import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
import seaborn as sns
from matplotlib.font_manager import FontProperties

colors = ['red','blue','green','purple']
algos = ['rf','lreg','our','rcn']

# plots two data sets side by side, used for synthetic data plot
def compact_plot(x,ys,yerrs,titles,legend_locs,title):
	fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
	for j in range(2):
		ax = axs[j]
		for i, algo in enumerate(algos):
			ax.errorbar(x, ys[j][i,:], yerrs[j][i,:], label = algo, ms=3, marker='.',color=colors[i],linewidth=0.8)
			ax.set_title(titles[j])
			ax.set_xlabel('noise rate $\eta$')
			ax.set_ylabel('accuracy')
			ax.set(aspect=0.7)
			ax.legend(loc=legend_locs[j],prop={'size': 6})
	fig.tight_layout(pad=0.0)
	plt.savefig('figures/%s.pdf' % title,bbox_inches='tight', transparent="True", pad_inches=0)

# generates synthetic plots
def gaussian_plot():
	x = np.arange(0.05, 0.5, 0.05)
	with open('gauss/rcn.json', 'r') as f:
		rcn_data = np.array(json.load(f))
		rcn_means = np.average(rcn_data,axis=1)
		rcn_stds = np.std(rcn_data,axis=1)
	with open('gauss/massart.json', 'r') as f:
		massart_data = np.array(json.load(f))
		massart_means = np.average(massart_data,axis=1)
		massart_stds = np.std(massart_data,axis=1)

	ys = np.array([rcn_means,massart_means])
	yerrs = np.array([rcn_stds,massart_stds])
	compact_plot(x,ys,yerrs,['RCN','Massart'],['lower left','lower left'],'synthetic')

# generates real data plots
def real_plot(censor):
	xs = np.linspace(0,0.4,5)
	ss = ['race_Black','gender_Female','native-country_United-States']
	actual_ss = ['>50K aframer','>50K female','>50K immigrant']
	not_ss = ['>50K, not aframer','>50K, not female','>50K, not immigrant']
	fig, axs = plt.subplots(nrows=3,ncols=3,sharey=True,sharex=True)
	if censor:
		folder = 'real/censor/'
	else:
		folder = 'real/noncensor/'
	for r,s in enumerate(ss):
		with open(folder + 'overall%s.json' % s,'r') as f:
			data1 = np.array(json.load(f))
			means1 = np.average(data1,axis=1)
			stds1 = np.std(data1,axis=1)

		with open(folder + 'target%s.json' % s, 'r') as f:
			data2 = np.array(json.load(f))
			means2 = np.average(data2,axis=1)
			stds2 = np.std(data2,axis=1)

		with open(folder + 'complement%s.json' % s, 'r') as f:
			data3 = np.array(json.load(f))
			means3 = np.average(data3,axis=1)
			stds3 = np.std(data2,axis=1)

		means = [means1,means2,means3]
		stds = [stds1,stds2,stds3]

		titles = ['adv %s: overall' % r, 'adv %s: %s' % (r,actual_ss[r]),'adv %s: %s' % (r,not_ss[r])]
		for i,algo in enumerate(algos):
			for j in range(3):
				axs[r][j].errorbar(xs,means[j][i,:],yerr=stds[j][i,:],label=algo,ms=3, marker='.',color=colors[i],linewidth=0.8)
				axs[r][j].set_title(titles[j],size=9)
		axs[2][1].set_xlabel('noise rate $\eta$',size=8)
		axs[1][0].set_ylabel('accuracy',size=8)
		axs[2][0].legend(loc='lower left', prop={'size':7})
	for i in range(3):
		for j in range(3):
			for tick in axs[i][j].xaxis.get_major_ticks():
				tick.label.set_fontsize(7) 
			for tick in axs[i][j].yaxis.get_major_ticks():
				tick.label.set_fontsize(7) 
	plt.tight_layout(pad=1.0)
	if censor:
		filename = 'censormain.pdf'
	else:
		filename = "noncensormain.pdf"
	plt.savefig('figures/%s' % filename,bbox_inches='tight', transparent="True", pad_inches=0)


gaussian_plot()
real_plot(True)
real_plot(False)
