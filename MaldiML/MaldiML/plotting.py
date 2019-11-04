# -*- coding: utf-8 -*-
# @Author: weisc
# @Date:   2019-09-16 09:25:11
# @Last Modified by:   weisc
# @Last Modified time: 2019-10-01 10:55:54

import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns

import os, xlrd, matplotlib, csv, re, six

from MaldiAI_pub1.metrics       import *
from MaldiAI_pub1.experiments   import *
from MaldiAI_pub1.utils         import count_elements, anglicize_labels, shorten_labels, calc_pos_class_ratio, PosRatio_pval
# from MaldiAI_pub1.utils import col_map as antibiotic2col_map
from MaldiAI_pub1.utils import col_map_anglicize, col_map
from string                     import ascii_letters
from collections                import Counter, namedtuple
from collections                import defaultdict as ddict
from itertools                  import compress
from operator                   import add

import matplotlib.pyplot        as plt
import matplotlib.colors        as mcolors
import matplotlib.patches       as mpatches
from matplotlib                 import gridspec, colors
from matplotlib.ticker          import NullFormatter
from matplotlib.colors          import Normalize

from sklearn.metrics            import roc_auc_score, confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.svm                import SVC
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from scipy.spatial.distance     import cdist
from scipy.cluster              import hierarchy
from scipy.spatial              import distance_matrix
from scipy.cluster.hierarchy    import dendrogram
from sklearn.decomposition      import IncrementalPCA
from sklearn.manifold           import TSNE, MDS, SpectralEmbedding, Isomap
from sklearn.cluster            import AgglomerativeClustering

from matplotlib.colors import LinearSegmentedColormap

plot_dir = '/links/groups/borgwardt/Projects/maldi_tof_diagnostics/plots/'

np.random.seed(123)
col_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
col_listUC = col_dict.keys()
col_list = [str(col_listUC[i]) for i in range(len(col_listUC))]
col_list_50 = sns.color_palette() \
                +sns.color_palette('bright') \
                +sns.color_palette('dark') \
                +sns.color_palette('pastel') \
                +sns.color_palette('colorblind')
np.random.shuffle(col_list)

#------------------
# helper functions
#------------------


def change_width_horizontal_bars(ax, new_value) :

    for patch in ax.patches :
        current_width = patch.get_height()
        diff = current_width - new_value

        # we change the bar width
        patch.set_height(new_value)

        # we recenter the bar
        patch.set_y(patch.get_y() + diff * .5)


#collist_ab = sns.color_palette('Paired', n_colors=12)+sns.color_palette("Set2") # higher than 12 will cause color repeats!

# antibiotic2col_map_anglicize = {}
# for key in antibiotic2col_map.keys():
#     antibiotic2col_map_anglicize[anglicize_labels(key)] = antibiotic2col_map[key]


#------------------
# s1 
#------------------


def plot_combined_vmeauc(list_ytrue, list_yprob, list_pheno, filename, legend=True, legend_fontsize=10, ylim_max=1.0, xlim_max=1.0):
    sns.set_style("whitegrid")
    plt.close('all')
    plt.figure(figsize=(12,12))

    sns.set_style("whitegrid")
    for i in range(len(list_ytrue)):
        vme, me_inv, thresholds = vme_auc_curve(list_ytrue[i], list_yprob[i])
        me = 1-me_inv
        plt.step(vme, me, color=col_map_anglicize[anglicize_labels(list_pheno[i])], alpha=1.0, where='post', linewidth=2)

    if legend:
        patches = [plt.plot([],[], marker=(4, 0, 45), ms=10, ls='', color=col_map_anglicize[anglicize_labels(list_pheno[i])], label="{:s}".format(anglicize_labels(list_pheno[i])))[0] for i in range(len(list_pheno))]     
        plt.legend(handles=patches, loc='upper right', prop={'size': legend_fontsize})

    plt.xlabel('Very Major Error', fontsize=15)
    plt.ylabel('Major Error', fontsize=15)
    plt.ylim([0.0, ylim_max])
    plt.xlim([0.0, xlim_max])

    plt.fill_between([0,0.015],0, [0.03,0.03], step='post', alpha=0.2, color='r')
    plt.plot(0.015,0.03, color='r', marker='o')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()

    fname1 = plot_dir+'{}_combined.png'.format(filename)
    fname2 = plot_dir+'{}_combined.pdf'.format(filename)
    plt.savefig(fname1)
    plt.savefig(fname2)  



def plot_vmeauc(list_Experiments, list_test_datasets, filename, legend=True, legend_fontsize=10, ylim_max=1.0, xlim_max=1.0, print_AB=None):

    list_ytrue = []
    list_yprob = []
    list_pheno = []

    # if print antibiotics not defined, take all
    if print_AB==None:
        print_AB = [e.phenotype for e in list_Experiments]
    

    for i, exp in enumerate(list_Experiments):
        data = list_test_datasets[i]

        if exp.phenotype in print_AB:
            list_ytrue.append(data.y)
            list_yprob.append(exp.estimator.predict_proba(data.X)[:, 1])
            
            # phenotype can be in 'antibiotic' or 'species-antibiotic' format
            if len(exp.phenotype.split('-'))>1:
                list_pheno.append(anglicize_labels(exp.phenotype.split('-')[1]))
            else:
                list_pheno.append(anglicize_labels(exp.phenotype))

        assert len(list_ytrue) == len(list_yprob) == len(list_pheno)
    plot_combined_vmeauc(list_ytrue, list_yprob, list_pheno, filename, legend, legend_fontsize=legend_fontsize, ylim_max=ylim_max, xlim_max=xlim_max)

 

# formerly plot_AMRpred_listROC_2bars
def plot_AMRprediction(list_Datasets, list_Experiments, list_rocauc_back, list_rocauc_front, 
    order_metrics=None, list_stds=None, list_pvals=None,
    color_by_samplesize=False, color_front='firebrick',  color_back=sns.color_palette()[0], 
    label_back='spectra', label_front='species', 
    filename='no_name_plot'):

    sns.palplot(sns.color_palette("deep", 10))
    assert len(list_Datasets) == len(list_Experiments) == len(list_rocauc_back) == len(list_rocauc_front)
    if list_pvals: assert len(list_Datasets) == len(list_pvals)
    if list_stds: assert len(list_Datasets) == len(list_stds)

    if order_metrics is None:
        order_metrics=list_rocauc_back

    # extract and convert labels
    list_labels = [unicode(x.tasks[0], errors='replace') for x in list_Datasets]
    list_labels = [x.decode('utf-8') for x in list_labels]
    list_labels = np.array([anglicize_labels(ll) for ll in list_labels])
    
    # extract metrics
    ROCAUC_comb = np.array(list_rocauc_back)
    ROCAUC_front_comb = np.array(list_rocauc_front)

    class_ratios = np.array([x.pos_class_ratio[0] for x in list_Datasets])

    num_samples = np.array([x.sample_size[0] for x in list_Datasets])
    clf_comb = np.array([x.clf_type for x in list_Experiments])
    
    # list standard deviations
    if list_stds:
        stds =  np.array(list_stds)
    else:
        stds =  np.array([x.stds for x in list_Experiments])

    # list pvalues
    if list_pvals:
        pvals =  np.array(list_pvals)

    # data preparation
    metric_for_sorting = np.array(order_metrics)
    auc_ix          = np.argsort(-metric_for_sorting)
    list_labels     = list_labels[auc_ix]
    class_ratios    = class_ratios[auc_ix]
    clf_comb        = clf_comb[auc_ix]
    num_samples     = num_samples[auc_ix]
    stds            = stds[auc_ix]
    if list_pvals:  pvals = pvals[auc_ix]
    ROCAUC_front_comb   = ROCAUC_front_comb[auc_ix]
    ROCAUC_comb         = ROCAUC_comb[auc_ix]

    if color_by_samplesize:
        color_idx = ['']*len(num_samples)
        for i in range(len(color_idx)):
            if num_samples[i] < 500:
                color_idx[i] = sns.color_palette()[4]
            if num_samples[i] > 500 and num_samples[i] < 5000:
                color_idx[i] = sns.color_palette()[1]
            if num_samples[i] > 5000 and num_samples[i] < 15000:
                color_idx[i] = sns.color_palette()[2]
            if num_samples[i] > 15000:
                color_idx[i] = sns.color_palette()[0]
    else:
        color_idx = [color_back]*len(num_samples)

    pretty_class_ratio = [str(round(class_ratios[i],2))+'0' if len(str(round(class_ratios[i],2)))==3 else str(round(class_ratios[i],2)) for i in range(len(list_labels))] 

    list_labels_ratios = [list_labels[i]+' ['+pretty_class_ratio[i]+']' for i in range(len(list_labels))] 

    d = {'ROC-AUC':ROCAUC_comb,'ROC-AUC_front':ROCAUC_front_comb,
        'std':stds, 'antibiotic+ratios':list_labels_ratios, 'antibiotic':list_labels, 
        'ratios': class_ratios, 'number samples': num_samples}
    df = pd.DataFrame(d)

    
    class_ratios = class_ratios[np.newaxis, :]
    num_samples = num_samples[np.newaxis, :]


    # -------------------
    # plot AUC  

    plt.close('all')
    sns.set(font_scale=1.9)
    sns.set_context(rc={'lines.markeredgewidth': 0.1})
    fig = plt.figure(figsize=(22,15))
    gs = gridspec.GridSpec(1, 1) 


    # plot performance
    ax1 = plt.subplot(gs[0])
    # barplot in the back and front
    ax1 = sns.barplot(x="antibiotic+ratios", y="ROC-AUC", data=df, palette=color_idx)
    ax1 = sns.barplot(x="antibiotic+ratios", y="ROC-AUC_front", data=df, color=color_front)


    # plot errorbars
    x_errorbar = np.arange(0, len(ROCAUC_comb), 1)
    ax1.errorbar(x_errorbar, ROCAUC_comb, yerr=stds, fmt='o',color='black')

    # -------------------
    # plot p-values
    if list_pvals:
        y_values = ROCAUC_comb
        x_values = range(len(ROCAUC_comb))
        x_values_text = [x-0.02 for x in x_values]
        y_values_text = [r+stds[i]+0.055 for i, r in enumerate(y_values)]
        
        for i, xy in enumerate(zip(x_values, y_values)): 
            xytext = (x_values_text[i], y_values_text[i])                     
            ax1.annotate('{0:.2g}'.format(pvals[i]), xy=xy, xytext=xytext, color='black', fontsize=16, rotation=90)        


    # ------------------
    # prepare legend

    if color_by_samplesize:
        if np.min(num_samples)<500:
            col =   [sns.color_palette()[4], sns.color_palette()[1], sns.color_palette()[2],    sns.color_palette()[0], 'firebrick']
            texts = ['< 500',   '500 - 5000',   '5000 -15000','> 15000', label_front]
            mark =  [(4, 0, 45), (4, 0, 45),    (4, 0, 45),     (4, 0, 45),     (4, 0, 45)]
        if np.min(num_samples)>500 and np.min(num_samples)<5000:
            col =   [sns.color_palette()[1], sns.color_palette()[2],    sns.color_palette()[0], 'firebrick']
            texts = ['500 - 5000',  '5000 -15000','> 15000', label_front]
            mark =  [(4, 0, 45),    (4, 0, 45),     (4, 0, 45),     (4, 0, 45)]
        if np.min(num_samples)>5000:
            col =   [sns.color_palette()[2],    sns.color_palette()[0], 'firebrick']
            texts = ['5000 -15000','> 15000', label_front]
            mark =  [(4, 0, 45),    (4, 0, 45),     (4, 0, 45)]
    else:
        col =   [color_back, color_front]
        texts = [label_back, label_front]
        mark =  [(4, 0, 45), (4, 0, 45)]        
    patches = [plt.plot([],[], marker=mark[i], ms=25, ls='', color=col[i], label="{:s}".format(texts[i]))[0] for i in range(len(texts))]


    plt.ylim(0.5, 1.05)
    plt.xlim(0-0.5, len(ROCAUC_comb)-0.5)
    plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.02,1))
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    
    ax1.set_ylabel('AUROC')
    ax1.set_xlabel('')
    fig.tight_layout()

    fname1 = plot_dir + '{}_2bars.png'.format(filename)
    fname2 = plot_dir + '{}_2bars.pdf'.format(filename)
    fig.savefig(fname1)
    fig.savefig(fname2)


#------------------
# s2 
#------------------



#------------------
# s2 
#------------------

def plot_pointplot_from_metric_dicts(metric_dict, phenotype, plot_name, col=None):
    metrics1 = ['rocauc','prauc','prauc0']
    metrics2 = ['prauc_baseline','prauc0_baseline']
    dd1 = ddict(list)
    dd2 = ddict(list)

    phenotype = anglicize_labels(phenotype)

    l_species = metric_dict.keys()
    #print l_species

    for spec in l_species:
        for metr in metrics1:
            dd1['value'].append(getattr(metric_dict[spec], metr))
            dd1['species'].append(spec+' %S: '+str(round(getattr(metric_dict[spec], 'pos_class_ratio'),2)))
            dd1['sample_size'].append(int(getattr(metric_dict[spec], 'sample_size')))

            if metr=='prauc':
                dd1['metric'].append('PR-AUC S')
            elif metr=='prauc0':
                dd1['metric'].append('PR-AUC R/I')
            elif metr=='rocauc':
                dd1['metric'].append('ROC-AUC')
    

    for spec in l_species:
        for metr in metrics2:
            if metr=='prauc_baseline':
                dd2['value'].append(getattr(metric_dict[spec], 'pos_class_ratio'))
            elif metr=='prauc0_baseline':
                dd2['value'].append(1-getattr(metric_dict[spec], 'pos_class_ratio'))
            else:
                continue

            dd2['species'].append(spec+' %S: '+str(round(getattr(metric_dict[spec], 'pos_class_ratio'),2))+' test sample size: '+ str(getattr(metric_dict[spec], 'sample_size')))
            dd2['sample_size'].append(int(getattr(metric_dict[spec], 'sample_size')))

            if metr=='prauc_baseline':
                dd2['metric'].append('PR-AUC S baseline')
            elif metr=='prauc0_baseline':
                dd2['metric'].append('PR-AUC R/I baseline')

    df1 = pd.DataFrame(dd1)
    df1.sort_values(by='sample_size', ascending=False, inplace=True)
    df2 = pd.DataFrame(dd2)
    df2.sort_values(by='sample_size', ascending=False, inplace=True) 
    sns.set(style="whitegrid")

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(12,6))

    # Plot the total crashes
    sns.set_color_codes("muted")

    sns.pointplot(x="value", y="species", hue="metric",
              data=df1, dodge=0.0, join=False, palette="dark",
              markers=['d','d','d'],
              hue_order=['PR-AUC S','PR-AUC R/I','ROC-AUC'],
              scale=1.0, ci=None)

    sns.pointplot(x="value", y="species", hue="metric",
              data=df2, dodge=0.0, join=False, palette="dark",
              scale=2.0, yerr=4.0, ci=680,
              markers=['|','|'],
              hue_order=['PR-AUC S baseline','PR-AUC R/I baseline'])

    # Add a legend and informative axis label
    ax.set(ylabel=phenotype, 
           xlim=(0.0, 1.0), 
           xlabel='metric value')
    sns.despine(left=True, bottom=True)
    ax.vlines([0.5], *ax.get_ylim())
    ax.hlines(range(len(l_species)), *ax.get_xlim(), alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')


    plt.tight_layout()
    plt.savefig(plot_name+'_pointplot_{}.pdf'.format(phenotype))
    plt.savefig(plot_name+'_pointplot_{}.png'.format(phenotype))





def plot_AUCcurves_from_lists_VME(list_Experiments, list_test_datasets, phenotype, plot_name, mode='panel', col=None):

    list_ytrue = []
    list_yprob = []
    list_pheno = []
    list_class1_ratio = []
    
    for i, exp in enumerate(list_Experiments):
        print exp.phenotype
        data = list_test_datasets[i]
        list_ytrue.append(data.y)
        list_yprob.append(exp.estimator.predict_proba(data.X)[:, 1])
        list_class1_ratio.append(round(data.pos_class_ratio[0],3))
        if len(exp.phenotype.split('-'))>1:
            list_pheno.append(anglicize_labels(exp.phenotype.split('-')[1]))
        else:
            list_pheno.append(anglicize_labels(exp.phenotype))
        assert len(list_ytrue) == len(list_yprob) == len(list_pheno)
    n_antibiotics = len(list_ytrue)
    # 
    # plot
    #

    sns.set(style="whitegrid")
    collist = sns.color_palette()+sns.color_palette('bright')

    if mode=='panel':
        n_subpanels = int(len(list_ytrue))
    elif mode=='stacked':
        n_subpanels = 1
    
    fig, ax = plt.subplots(n_subpanels, 3, figsize=(30,n_subpanels*10))

    if mode=='stacked':
        ax = ax.reshape(1,3)


    # 2 panels
    for i in range(n_antibiotics):

        if mode=='panel':
            j = i
        elif mode=='stacked':
            j = 0
    
        col = collist[i]

        # ------------
        # panel1: ROC curve
        # ------------
        fpr, tpr, thresholds = roc_curve(list_ytrue[i], list_yprob[i], pos_label=1)
        rocauc = round(roc_auc_score(list_ytrue[i], list_yprob[i]), 3)
        pretty_rocauc = [str(roc)+'0' if len(str(roc))==3 else str(roc) for roc in [rocauc]] 

        
        lab = '{}\t'.format(list_pheno[i]).expandtabs()
        if len(lab)==24:
            lab = '{}\t\t'.format(list_pheno[i]).expandtabs()
        lab = lab+'AUROC: '+pretty_rocauc[0]

        ax[j,0].plot(fpr, tpr, color=col, label=lab, linewidth=3.0)
        ax[j,0].plot([0, 1], [0, 1], color='black', linestyle='--')
        if mode=='panel': ax[j,0].fill_between(fpr, tpr, step='post', alpha=0.2, color=col)

        # ------------
        # panel2: PRAUC-1 curve
        # ------------
        precision, recall, thresholds = precision_recall_curve(list_ytrue[i], list_yprob[i])
        prauc = round(average_precision_score(list_ytrue[i], list_yprob[i], average='weighted'), 3)
        pretty_prauc = [str(pr)+'0' if len(str(pr))==3 else str(pr) for pr in [prauc]] 

        lab = '{}\t'.format(list_pheno[i]).expandtabs()
        if len(lab)==24:
            lab = '{}\t\t'.format(list_pheno[i]).expandtabs()
        lab = lab+'AUPRC: '+pretty_prauc[0]+'   pcr: {}%'.format(str(100*list_class1_ratio[i]))

        ax[j,1].step(recall, precision, color=col, label=lab, alpha=1.0, where='post', linewidth=3.0)
        if mode=='panel': ax[j,1].fill_between(recall, precision, step='post', alpha=0.2, color=col)
        if mode=='panel': ax[j,1].axhline(y=float(list_class1_ratio[i]), color='black', linestyle='--')

        # ------------
        # panel3: VME curve
        # ------------
        vme, me_inv, thresholds = vme_auc_curve(list_ytrue[i], list_yprob[i])
        me = 1-me_inv
        vme_score = round(vme_auc_score(list_ytrue[i], list_yprob[i]),3)
        pretty_vme = [str(pr)+'0' if len(str(pr))==3 else str(pr) for pr in [vme_score]] 

        lab = '{}\t'.format(list_pheno[i]).expandtabs()
        if len(lab)==24:
            lab = '{}\t\t'.format(list_pheno[i]).expandtabs()
        lab = lab+'AUVME: '+pretty_vme[0]

        ax[j,2].step(vme, me, color=col, label=lab, alpha=1.0, where='post', linewidth=3.0)
        
        if mode=='panel': ax[j,2].fill_between(me, vme, step='post', alpha=0.2, color=col)

        # ------------
        # axes limits and labels
        # ------------
        ax[j,0].set_xlabel('False Positive Rate')
        ax[j,0].set_ylabel('True Positive Rate')
        ax[j,1].set_xlabel('Recall')
        ax[j,1].set_ylabel('Precision')
        ax[j,2].set_xlabel('Very major error')
        ax[j,2].set_ylabel('Major error')
        ax[j,0].legend(bbox_to_anchor=(0.99, 0.01), loc='lower right', prop={'family': 'DejaVu Sans Mono', 'size': 15})
        ax[j,1].legend(bbox_to_anchor=(0.01, 0.01), loc='lower left', prop={'family': 'DejaVu Sans Mono', 'size': 15})
        ax[j,2].legend(bbox_to_anchor=(0.99, 0.99), loc='upper right', prop={'family': 'DejaVu Sans Mono', 'size': 15})

        ax[j,0].set_xlim([-0.01,1.0])
        ax[j,0].set_ylim([0.0,1.01])
        ax[j,1].set_xlim([-0.01,1.0])
        ax[j,1].set_ylim([0.0,1.01])   
        ax[j,2].set_xlim([-0.01,1.0])
        ax[j,2].set_ylim([0.0,1.01]) 

    plt.tight_layout()
    plt.savefig(plot_dir+plot_name+'_AUCcurves_{}_{}.pdf'.format(phenotype, mode))
    plt.savefig(plot_dir+plot_name+'_AUCcurves_{}_{}.png'.format(phenotype, mode))


def plot_AUCcurves_1spec_allAB(list_Experiments, list_test_datasets, species, plot_name, mode='stacked', col=None):

    list_ytrue = []
    list_yprob = []
    list_pheno = []
    list_class1_ratio = []
    
    for i, exp in enumerate(list_Experiments):
        print exp.phenotype
        data = list_test_datasets[i]
        list_ytrue.append(data.y)
        list_yprob.append(exp.estimator.predict_proba(data.X)[:, 1])
        list_class1_ratio.append(round(data.pos_class_ratio[0],3))
        if len(exp.phenotype.split('-'))>1:
            list_pheno.append(anglicize_labels(exp.phenotype.split('-')[0]))
        else:
            list_pheno.append(anglicize_labels(exp.phenotype))
        assert len(list_ytrue) == len(list_yprob) == len(list_pheno)
    
    n_antibiotics = len(list_ytrue)

    # 
    # plot
    #

    sns.set(style="whitegrid")
    axes_fontsize = 18

    if mode=='panel':
        n_subpanels = int(len(list_ytrue))
    elif mode=='stacked':
        n_subpanels = 1
    
    fig, ax = plt.subplots(n_subpanels, 3, figsize=(30,n_subpanels*10))

    if mode=='stacked':
        ax = ax.reshape(1,3)


    # 2 panels
    for i in range(n_antibiotics):

        if mode=='panel':
            j = i
        elif mode=='stacked':
            j = 0
    
        col = col_map_anglicize[list_pheno[i]]
        list_pheno[i] = shorten_labels(list_pheno[i])

        # ------------
        # panel1: ROC curve
        # ------------
        fpr, tpr, thresholds = roc_curve(list_ytrue[i], list_yprob[i], pos_label=1)
        rocauc = round(roc_auc_score(list_ytrue[i], list_yprob[i]), 3)
        pretty_rocauc = [str(roc)+'0' if len(str(roc)) in [3,4] else str(roc) for roc in [rocauc]] 

        
        lab = '{}\t'.format(list_pheno[i]).expandtabs()
        if len(lab)<=15:
            lab = '{}\t\t'.format(list_pheno[i]).expandtabs() 
        lab = lab+'AUROC: '+pretty_rocauc[0]

        ax[j,0].plot(fpr, tpr, color=col, label=lab, linewidth=3.0)
        ax[j,0].plot([0, 1], [0, 1], color='black', linestyle='--')
        if mode=='panel': ax[j,0].fill_between(fpr, tpr, step='post', alpha=0.2, color=col)

        # ------------
        # panel2: PRAUC-1 curve
        # ------------
        precision, recall, thresholds = precision_recall_curve(list_ytrue[i], list_yprob[i])
        prauc = round(average_precision_score(list_ytrue[i], list_yprob[i], average='weighted'), 3)
        pretty_prauc = [str(pr)+'0' if len(str(pr)) in [3,4] else str(pr) for pr in [prauc]] 

        lab = '{}\t'.format(list_pheno[i]).expandtabs()
        if len(lab)<=15:
            lab = '{}\t\t'.format(list_pheno[i]).expandtabs()
        lab = lab+'AUPRC: '+pretty_prauc[0]+'   pcr: {}%'.format(str(100*list_class1_ratio[i]))

        ax[j,1].step(recall, precision, color=col, label=lab, alpha=1.0, where='post', linewidth=3.0)
        if mode=='panel': ax[j,1].fill_between(recall, precision, step='post', alpha=0.2, color=col)
        if mode=='panel': ax[j,1].axhline(y=float(list_class1_ratio[i]), color='black', linestyle='--')

        # ------------
        # panel3: VME curve
        # ------------
        vme, me_inv, thresholds = vme_auc_curve(list_ytrue[i], list_yprob[i])
        me = 1-me_inv
        vme_score = round(vme_auc_score(list_ytrue[i], list_yprob[i]),3)
        pretty_vme = [str(pr)+'0' if len(str(pr)) in [3,4] else str(pr) for pr in [vme_score]] 

        lab = '{}\t'.format(list_pheno[i]).expandtabs()
        if len(lab)<=15:
            lab = '{}\t\t'.format(list_pheno[i]).expandtabs() 
        lab = lab+'AUVME: '+pretty_vme[0]

        ax[j,2].step(vme, me, color=col, label=lab, alpha=1.0, where='post', linewidth=3.0)
        ax[j,2].fill_between([0,0.015],0, [0.03,0.03], step='post', alpha=0.2, color='r')
        ax[j,2].plot(0.015,0.03, color='r', marker='o')
        
        if mode=='panel': ax[j,2].fill_between(me, vme, step='post', alpha=0.2, color=col)

        # ------------
        # axes limits and labels
        # ------------
        ax[j,0].set_xlabel('False Positive Rate', fontsize=axes_fontsize)
        ax[j,0].set_ylabel('True Positive Rate', fontsize=axes_fontsize)
        ax[j,1].set_xlabel('Recall', fontsize=axes_fontsize)
        ax[j,1].set_ylabel('Precision', fontsize=axes_fontsize)
        ax[j,2].set_xlabel('Very major error', fontsize=axes_fontsize)
        ax[j,2].set_ylabel('Major error', fontsize=axes_fontsize)
        ax[j,0].legend(bbox_to_anchor=(0.99, 0.01), loc='lower right', prop={'family': 'DejaVu Sans Mono', 'size': 15})
        ax[j,1].legend(bbox_to_anchor=(0.01, 0.01), loc='lower left', prop={'family': 'DejaVu Sans Mono', 'size': 15})
        ax[j,2].legend(bbox_to_anchor=(0.99, 0.99), loc='upper right', prop={'family': 'DejaVu Sans Mono', 'size': 15})

        ax[j,0].set_xlim([-0.01,1.0])
        ax[j,0].set_ylim([0.0,1.01])
        ax[j,1].set_xlim([-0.01,1.0])
        ax[j,1].set_ylim([0.0,1.01])   
        ax[j,2].set_xlim([-0.01,1.0])
        ax[j,2].set_ylim([0.0,1.01]) 

    plt.tight_layout()
    plt.savefig(plot_name+'_1spec_{}_{}.pdf'.format(species, mode))
    plt.savefig(plot_name+'_1spec_{}_{}.png'.format(species, mode))





def plot_panel_pointplot_from_metric_dicts(l_metric_dict, l_phenotype, plot_name, col=None):
    assert len(l_metric_dict)<=6, 'Plotting script intended for max. 6 plots.'

    metrics1 = ['rocauc','prauc','prauc0']
    metrics2 = ['prauc_baseline','prauc0_baseline']
    
    # Initialize the matplotlib figure
    plt.close('all')
    f, [[ax1, ax2, ax3], [ax4, ax5, ax6]]  = plt.subplots(2, 3, figsize=(50,20))

    sns.set(style="whitegrid")
    sns.set_color_codes("muted")

    for i in range(len(l_metric_dict)):

        metric_dict = l_metric_dict[i]
        phenotype = l_phenotype[i]
        l_species = metric_dict.keys()
        
        dd1 = ddict(list)
        dd2 = ddict(list)

        for spec in l_species:
            for metr in metrics1:
                dd1['value'].append(getattr(metric_dict[spec], metr))
                dd1['species'].append(spec+' %S: '+ str(round(getattr(metric_dict[spec], 'pos_class_ratio'),2)) +' test sample size: '+ str(getattr(metric_dict[spec], 'sample_size')))

                if metr=='prauc':
                    dd1['metric'].append('PR-AUC S')
                elif metr=='prauc0':
                    dd1['metric'].append('PR-AUC R/I')
                elif metr=='rocauc':
                    dd1['metric'].append('ROC-AUC')

        for spec in l_species:
            for metr in metrics2:
                if metr=='prauc_baseline':
                    dd2['value'].append(getattr(metric_dict[spec], 'pos_class_ratio'))
                elif metr=='prauc0_baseline':
                    dd2['value'].append(1-getattr(metric_dict[spec], 'pos_class_ratio'))
                else:
                    continue

                dd2['species'].append(spec+' %S: '+str(round(getattr(metric_dict[spec], 'pos_class_ratio'),2)))

                if metr=='prauc_baseline':
                    dd2['metric'].append('PR-AUC S baseline')
                elif metr=='prauc0_baseline':
                    dd2['metric'].append('PR-AUC R/I baseline')

        df1 = pd.DataFrame(dd1)
        df2 = pd.DataFrame(dd2)
        
        ax = locals()['ax{}'.format(i+1)]

        sns.pointplot(x="value", y="species", hue="metric",
                  data=df1, dodge=0.0, join=False, palette="dark",
                  markers=['d','d','d'], ax=ax,
                  hue_order=['PR-AUC S','PR-AUC R/I','ROC-AUC'],
                  scale=1.0, ci=None)

        sns.pointplot(x="value", y="species", hue="metric",
                  data=df2, dodge=0.0, join=False, palette="dark",
                  scale=2.0, yerr=4.0, ci=680,
                  markers=['|','|'], ax=ax,
                  hue_order=['PR-AUC S baseline','PR-AUC R/I baseline'])

        # Add a legend and informative axis label

        ax.set(ylabel=phenotype, 
               xlim=(0.0, 1.0), 
               xlabel='metric value')

        ax.get_legend().set_visible(False)


        # sns.despine(left=True, bottom=True)
        ax.vlines([0.5], *ax.get_ylim())
        ax.hlines(range(len(l_species)), *ax.get_xlim(), alpha=0.3)
        plt.legend(bbox_to_anchor=(1.15, 0.5), loc='center left')

    plt.tight_layout()
    plt.savefig(plot_name+'_panel_6pointplots.pdf')
    plt.savefig(plot_name+'_panel_6pointplots.png')





# def plot_Dataset_AMRprecentage(Dataset, task, plot_name, top_num=10):
    data = Dataset.copy()
    data = data.to_singletask(task)
    task = anglicize_labels(task)

    # make lists with sizes
    list_species = data.sample_species
    list_class = data.y
    species_counts = count_elements(data.sample_species)

    list_species_names_total = [s[0] for s in species_counts][:top_num]

    list_species_posratio_total = []
    for i,spec in enumerate(list_species_names_total):
        data_spec = data.copy()
        data_spec.reduce_to_species(spec)
        list_species_posratio_total.append(data_spec.pos_class_ratio[0])
    print list_species_posratio_total

    # plot
    plt.close('all')
    f, ax = plt.subplots()

    # plot for all labeled samples (looks like negative samples)
    sns.barplot(x=[1]*len(list_species_posratio_total), y=list_species_names_total, color='cornflowerblue')
    # plot for all class 1 samples
    sns.barplot(x=list_species_posratio_total, y=list_species_names_total, color='darkblue')

    # Add a legend and informative axis label
    plt.title(task)
    ax.set(ylabel='', 
           xlabel='')
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(plot_name+'_{}_AMRprecentage.pdf'.format(task))


def plot_Dataset_AMRprecentage_overlay(Dataset_main, Datasets_overlay, overlay_names, task, plot_name, pvalues=True, top_num=10, min_num=5):
    
    Dataset_main = Dataset_main.to_singletask(task)
    Dataset_main.remove_nans()

    # make lists with sizes
    list_species = Dataset_main.sample_species
    list_class = Dataset_main.y

    species_counts = count_elements(Dataset_main.sample_species)

    list_species_names_complete = [s[0] for s in species_counts][:top_num]
    list_species_names_total = []

    # -----------
    # calc pos_class_ratio for each spec in Dataset_main
    list_species_posratio_total = []
    for i,spec in enumerate(list_species_names_complete):
        if np.sum(np.isin(Dataset_main.sample_species, spec))>=min_num:
            remove_idx = ~np.isin(Dataset_main.sample_species, spec)
            pos_class_ratio = calc_pos_class_ratio(Dataset_main.y[~remove_idx])
            list_species_posratio_total.append(float(pos_class_ratio))
            list_species_names_total.append(list_species_names_complete[i])

    # -----------
    # calc pos_class_ratio and p-value w.r.t. to Dataset_main, for each spec in Datasets_overlay
    for j, dataset in enumerate(Datasets_overlay):

        # if task not in Dataset_overlay[j], skip Dataset_overlay[j]
        if task not in dataset.tasks:
            locals()['list_ratios_{}'.format(j)] = [None]*len(list_species_names_total)
            locals()['list_pval_{}'.format(j)] = [None]*len(list_species_names_total)
            continue
        
        dataset = dataset.to_singletask(task)
        dataset.remove_nans()

        locals()['list_ratios_{}'.format(j)] = []
        locals()['list_pval_{}'.format(j)] = []

        
        for i,spec in enumerate(list_species_names_total):
            if not np.sum(np.isin(Dataset_main.sample_species, spec))>=min_num:
                continue

            if spec in dataset.sample_species:
                
                if np.sum(np.isin(dataset.sample_species, spec))>=min_num:
                    data_ref_spec = Dataset_main.copy()
                    data_ref_spec.reduce_to_species(spec)                    
                    data_overlay_spec = dataset.copy()
                    data_overlay_spec.reduce_to_species(spec)

                    pos_class_ratio = data_overlay_spec.pos_class_ratio[0]
                    pvalue = PosRatio_pval(data_ref_spec, data_overlay_spec)

                    locals()['list_ratios_{}'.format(j)].append(float(pos_class_ratio))
                    # p-value is not defined if all freqeuncies are 0
                    if np.isnan(pvalue):
                        locals()['list_pval_{}'.format(j)].append(None)
                    else:
                        locals()['list_pval_{}'.format(j)].append(float(round(pvalue, 2)))
                    
                    
                else:
                    locals()['list_ratios_{}'.format(j)].append(None)
                    locals()['list_pval_{}'.format(j)].append(None)

            else:
                locals()['list_ratios_{}'.format(j)].append(None)
                locals()['list_pval_{}'.format(j)].append(None)


    # check all list_ratios are the same length
    for j, _ in enumerate(Datasets_overlay):
        assert len(locals()['list_ratios_{}'.format(j)]) == len(list_species_posratio_total) == len(list_species_names_total)
        assert len(locals()['list_pval_{}'.format(j)]) == len(list_species_posratio_total) == len(list_species_names_total)


    # ----------------------
    # plot

    print(zip(list_species_names_total, list_species_posratio_total, 
        locals()['list_ratios_{}'.format(0)], 
        locals()['list_pval_{}'.format(0)]))

    print('\n\n\tstart plotting..')
    plt.close('all')
    f, ax = plt.subplots(figsize=(18,16))
    task = anglicize_labels(task)

    # seaborn style
    # rc = {'axes.labelsize':12, 'xtick.major.size':10, 'ytick.major.size':10, 'legend.fontsize': 10}
    # rc = {'axes.labelsize':12, 'xtick.major.size':10, 'ytick.major.size':10}
    rc = matplotlib.rcParams
    rc = {'ytick.major.size':20}
    # plt.rcParams.update(**rc)
    sns.set(font_scale = 1.5, rc=rc) 
    sns.set_style("whitegrid")


    sns.barplot(x=list_species_posratio_total, y=list_species_names_total, label='USB %class-S', color='grey', ax=ax, zorder=-1)

    # pointplots
    for j, _ in enumerate(Datasets_overlay):
        print 'starting '+ str(j)

        sns.pointplot(x=locals()['list_ratios_{}'.format(j)], y=list_species_names_total, color=col_list_50[j+1],
                  dodge=True, join=False, scale=2.0, 
                  label=overlay_names[j],
                  markers='d', ax=ax, zorder=1000+j)
        err = np.array([0.1 for r in locals()['list_ratios_{}'.format(j)]])
        idx_none = np.array([True if r==None else False for r in locals()['list_ratios_{}'.format(j)]])

        if pvalues:
            if j%2==0:
                y_values = [r-0.25 for r in range(len(list_species_names_total))]
            else:
                y_values = [r+0.45 for r in range(len(list_species_names_total))]
            # y_values = [r-0.001 for r in locals()['list_ratios_{}'.format(j)] if r!=None]
            x_values = locals()['list_ratios_{}'.format(j)]
            x_values_text = [r-0.014 if r!=None else None for r in x_values ]
            for i, xy in enumerate(zip(x_values, y_values)): 
                xytext = (x_values_text[i], y_values[i])
                if y_values[i]!=None and x_values[i]!=None and locals()['list_pval_{}'.format(j)][i]!=None:                         
                    ax.annotate('{}'.format(locals()['list_pval_{}'.format(j)][i]), xy=xy, xytext=xytext, 
                        arrowprops=dict(arrowstyle='->'), color=col_list_50[j+1], fontsize=12, zorder=1050+j)

    change_width_horizontal_bars(ax, .50)

    # legend
    col =   ['grey']+col_list_50[1:]
    texts = ['USB class S']+overlay_names
    mark =  [(4, 0, 45)]*(1+len(Datasets_overlay))
    patches = [plt.plot([],[], marker=mark[i], ms=15, ls='', color=col[i], label="{:s} %class-S".format(texts[i]))[0] for i in range(len(texts))]        
    plt.legend(handles=patches, bbox_to_anchor=(1.01, 0.5), loc='center left')


    # Add a legend and informative axis label
    plt.title(task)
    ax.set(ylabel='', 
           xlabel='')
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(plot_name+'top{}_{}_PosPerc_pval.png'.format(top_num,task))
    plt.savefig(plot_name+'top{}_{}_PosPerc_pval.pdf'.format(top_num,task))


