import glasbey
import numpy as np
import textalloc as ta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import anndata as ad
import scanpy as sc

def latex_str(string):
    string_new = list(string)
    for i, char in enumerate(string):
        if char == 'δ':
            string_new[i] = '$\delta$'
        elif char == 'γ':
            string_new[i] = '$\gamma$'            
        elif char == 'Φ':
            string_new[i] = '$\Phi$'   
    return r''.join(string_new)

def return_palette(data_set):
    if data_set == 'liver':
        palette = {
            'B cells': '#964b00',
            'B/Plasma': '#895a00',    
            'B/Plasma cells': '#895a00',             
            'Basophils': '#000000',
            'CCL2+ fibro': '#ed1c24',
            'CD4+ KLRB1 T': '#00fa9a',
            'CD8 eff. memory T': '#44d7a8',
            'CV and capsule': '#00b7eb',
            'Capsule fibroblasts': '#cf1020',
            'Central vein ETCs': '#fcc200',
            'Cholangiocytes': '#c90016',
            'Cholangio': '#c90016',    
            'Circulating NK': '#d9e650',
            'Circulating NK/NKT': '#008000',    
            'Circulating eff. memory T': '#195905',
            'Conflict: mono/neutrophils': '#cec8ef',
            'Cytotoxic CD8+': '#00ff00',
            'Effector Memory T': '#76ff7a',
            'Endothelial': '#ffbf00',
            'Fibroblast 1': '#f08080',
            'Fibroblast 2': '#ff4500',    
            'Fibroblasts': '#ff3800',
            'Hepatocytes': '#ff0090',
            'Hepatocytes NucSeq': '#ca1f7b',
            'HsPCs': '#e62020',
            'ILCs': '#7fff00',
            'KCs': '#00bfff',
            'Lipid associated MΦ': '#b9f2ff',
            'Liver sinusoidal ETCs': '#ffa700',
            'LSECs': '#ffa700',    
            'Lymphatic ETCs': '#ed7000',
            'Macrophages': '#1034a6',    
            'Mesothelial cells': '#e66771',    
            'Migratory cDCs': '#8f00ff',
            'MoMac1': '#4169e1',
            'MoMac2': '#45b1e8',
            'Monocytes': '#89cff0',   
            'Mono': '#89cff0',     
            'Mono/mono derived': '#4f86f7',
            'NK': '#568203',
            'NK cells': '#568203',   
            'NK/NKT cells': '#829b17',
            'NKT': '#afb42b',   
            'Naive CD4': '#3fff00',
            'Naive CD4+ T': '#3fff00',
            'Naive CD8': '#90ee90',
            'Naive CD8+ T': '#90ee90',
            'Naive/CM CD4+ T': '#85bb65',     
            'Neutrophils': '#838996',
            'Patrolling MΦ': '#00e5ff',
            'Patrolling monocytes': '#214fc6',
            'Peritoneal MΦ': '#73c2fb',    
            'Plasma': '#c19a6b',
            'Plasma cells': '#c19a6b',    
            'Platelets': '#A3AEA7',
            'Portal vein ETCs': '#ffd700',
            'Pre-moKCs/moKCs': '#2a52be',
            'Regulatory T': '#b2ec5d',
            'Stellate': '#ff033e',
            'Stromal cells': '#ff3800',
            'T cells': '#32cd32',
            'T helper': '#009698',
            'Th17s': '#3b7a57',
            'Th1s': '#0bda51',
            'Tissue res. CD8+ T': '#99e6b3',
            'Tissue res. NK': '#3eb489',
            'Trans. mono. 1': '#0000cd',
            'Trans. mono. 2': '#0f52ba',
            'VSMCs': '#ff0038',    
            'Tissue res. NK': '#3cd070',
            'cDC1s': '#800080',
            'cDC2s': '#bf00ff',
            'cDCs': '#a000c0',
            'immLAMs': '#318ce7',
            'matLAMs': '#a1caf1',    
            'pDCs': '#d891ef',
            'resKCs': '#5b92e5',
            'γδ T cells': '#9acd32',
            'Circ. eff. memory T':'#195905',  
            'Mono/Neutrophils': '#4f86f7',   
            }
        
    elif data_set == 'species':
        palette = {
            'Mouse' : 'tab:orange',
            'Human' : 'tab:purple',
            'Mouse Nafld' : 'tab:red',
            'Pig' : 'tab:olive',
            'Chicken' : 'tab:pink',
            'Hamster' : 'tab:blue',
            'Monkey' : 'tab:green',                      
            }

    elif data_set == 'adipose':
        palette = {
            'B cell': '#964b00',
            'T cell': '#348810',    
            'dendritic cell': '#000000',
            'endothelial cell': '#ffae42',
            'endothelial cell of lymphatic vessel': '#fedf00',
            'epithelial cell': '#6244ae',
            'mammary gland epithelial cell': '#937ad6',
            'fat cell': '#ff43a4',
            'immature NK T cell': '#9acd32',
            'macrophage': '#1034a6',
            'mast cell': '#e9f16d',
            'mesothelial cell': '#cc2568',
            'monocyte': '#89cff0',
            'neutrophil': '#838996',
            'pericyte cell': '#9d794e',
            'preadipocyte': '#ff4444',
            'smooth muscle cell': '#39a78e',
            'stromal cell of endometrium': '#990012'}
        
    elif data_set == 'glioblastoma':
        palette = {
            'B':  '#964b00', 
            'B cells': '#964b00',     
            'Mast': '#664228', 
            'Mast cells 1': '#846651', 
            'Mast cells 2': '#8b7e66',
            'Mono 1': '#89cff0', 
            'Mono 2': '#4f86f7', 
            'Mono/TAM': '#8c034c',
            'NK': '#62f251', 
            'NK cells': '#62f251', 
            'Neutrophils': '#838996',
            'Oligo': '#fff200',
            'Proliferating T cells': '#a3e92c',
            'Prolif T cells': '#a3e92c',
            'Proliferating TAM': '#ffae42', 
            'Prolif TAM': '#ffae42', 
            'Plasma cells': '#c19a6b',
            'TAM 1': '#ff0000', 
            'TAM 2': '#ff780a', 
            'TAM 3': '#ff6666', 
            'TAM 4': '#9d0000', 
            'TAM 5': '#ff006a',
            'Tcells 1': '#6fcc47', 
            'T cells 1': '#6fcc47',
            'Tcells 2': '#348810', 
            'T cells 2': '#348810',
            'Tcells 3': '#0d6443', 
            'T cells 3': '#0d6443',             
            'Treg 1': '#fcb4f6',
            'Treg 2': '#d3aee9',
            'Treg': '#eac8c8', 
            'cDC1': '#800080',
            'cDC2': '#bf00ff',
            'pDC': '#d891ef',
            'mDC': '#000000',
        }        

    else:
        palette = None

    return palette    


def create_palette(names):
    name_list = np.unique(names)
    lenght = 10 + len(name_list)
    colors = glasbey.extend_palette("tab10", palette_size=lenght)
    return {name : colors[j] for j, name in enumerate(name_list)}


def ret_sign(number):
    if number >= 0:
        sign_str = "+"
    else:
        sign_str = "-"
    return  sign_str  

def is_bright(hex_color):
    hex_color = hex_color.lstrip('#')
    R, G, B = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    luminance = 0.299 * R + 0.587 * G + 0.114 * B
    threshold = 100 
    if luminance < threshold:
        return 'white'
    else:
        return 'black'   
    
def bar_plot(model, fs=15.9, add = 0, dataset='liver'):
    knn_adj=round(model.mdata.mod[model.target_dataset_key].uns['metrics']['adjusted_rand_score_nns_hom_genes'],3)
    model_adj=round(model.mdata.mod[model.target_dataset_key].uns['metrics']['adjusted_rand_score_nns_aligned_latent_space'],3)

    knn_mis = round(model.mdata.mod[model.target_dataset_key].uns['metrics']['adjusted_mutual_info_score_nns_hom_genes'],3)
    model_mis = round(model.mdata.mod[model.target_dataset_key].uns['metrics']['adjusted_mutual_info_score_nns_aligned_latent_space'],3)

    df_knn = model.mdata.mod[model.target_dataset_key].uns['prediction_df_nns_hom_genes']
    df_model = model.mdata.mod[model.target_dataset_key].uns['prediction_df_nns_aligned_latent_space']

    context_cell_labels = model.mdata.mod[model.context_dataset_key].obs[model.mdata.mod[model.context_dataset_key].uns['dataset_cell_key']].to_numpy()
    target_cell_labels = model.mdata.mod[model.target_dataset_key].obs[model.mdata.mod[model.target_dataset_key].uns['dataset_cell_key']].to_numpy()
    context_cell_types = np.unique(context_cell_labels)
    target_cell_types = np.unique(target_cell_labels)

    palette = return_palette(dataset)

    if palette == None:
        palette = create_palette(np.concatenate((context_cell_types,target_cell_types)))

    common_labels, _, c_ind_b = np.intersect1d(context_cell_types, target_cell_types, return_indices=True)
    all_labels = np.unique(np.concatenate((context_cell_types,target_cell_types)))
    improvement = np.array([df_model.loc[cell_type, cell_type] - df_knn.loc[cell_type, cell_type] for cell_type in common_labels])

    cells_old = np.array([df_knn.loc[key].sort_values(ascending=False).index for key in df_knn.index])
    cells_new = np.array([df_model.loc[key].sort_values(ascending=False).index for key in df_model.index])
    values_old = np.array([df_knn.loc[key].sort_values(ascending=False) for key in df_knn.index])
    values_new = np.array([df_model.loc[key].sort_values(ascending=False) for key in df_model.index])

    str_old = [str(i)+' - '+str(j) for (i,j) in zip(np.array([np.where(i == all_labels)[0][0]+1 for i in cells_old[:,0]]), np.array(round(df_knn.max(1),1)))]
    str_new = [str(i)+' - '+str(j) for (i,j) in zip(np.array([np.where(i == all_labels)[0][0]+1 for i in cells_new[:,0]]), np.array(round(df_model.max(1),1)))]

    labels = [' 'for i in target_cell_types]
    for j, ind in enumerate(c_ind_b):
        labels[ind] = ret_sign(improvement[j])+str(round(abs(improvement[j]),1))


    tick_colors = []
    for numb in labels:
        if numb == ' ':
            tick_colors += ['#212121'] 
        elif float(numb) < -20:
            tick_colors += ['#b71c1c']     
        elif float(numb) < -10:
            tick_colors += ['#c62828'] 
        elif float(numb) < -4:
            tick_colors += ['#e53935'] 
        elif float(numb) < 0:
            tick_colors += ['#e57373'] 
        elif float(numb) == 0.0:
            tick_colors += ['#212121'] 
        elif float(numb) < 4:
            tick_colors += ['#66bb6a'] 
        elif float(numb) < 10:
            tick_colors += ['#4caf50']         
        elif float(numb) < 20:
            tick_colors += ['#43a047']    
        elif float(numb) < 100:
            tick_colors += ['#388e3c']   
    tick_colors = np.array(tick_colors)    

    x = np.arange(len(df_knn.index))

    lab = np.unique(np.concatenate((context_cell_labels, target_cell_labels)))

    fig, ax = plt.subplots(figsize=(12,18), dpi=250)
    for i in range(len(target_cell_types)):
        n=0
        x_positions = np.flip(x+(0.5*n)-1.0/2)
        ax.axhline(y=i+0.5, color='black', linestyle='solid')
        p = ax.barh(x_positions, width=values_new[:,i], left=np.cumsum(np.insert(values_new, 0, 0, axis=1), axis=1)[:,i], edgecolor="white", height=0.5, align='edge', color=[palette[c] for c in cells_new[:,i]])
        n=1
        x_positions = np.flip(x+(0.5*n)-1.0/2)
        p_2 = ax.barh(x_positions, width=values_old[:,i], left=np.cumsum(np.insert(values_old, 0, 0, axis=1), axis=1)[:,i], edgecolor="white", height=0.5, align='edge', color=[palette[c] for c in cells_old[:,i]])    

    ax.set_yticks(x)
    ax.set_ylim(-0.5, len(df_knn.index)-0.5)
    ax.set_yticklabels(np.array([str(np.where(np.array(lab)==s)[0][0]+1)+': '+latex_str(s) for s in np.flip(df_knn.index)]), rotation=50, ha='right', fontsize=fs*0.95)#
    ax_t = ax.secondary_yaxis('right')
    ax_t.set_yticks(x)
    ax_t.tick_params(axis='y', direction='inout', length=10)
    ax_t.set_yticklabels(np.flip(labels), ha='left', fontsize=fs)#

    for i, color in enumerate(np.flip(tick_colors)):
        ax_t.get_yticklabels()[i].set_color(color)

    ax.set_xticks(np.arange(0,110,10))
    ax.set_xticklabels(np.array([i.astype(int).astype(str) for i in np.arange(0,110,10).astype(int).astype(str)]), fontsize=fs)#
    ax.xaxis.set_tick_params(length=10)

    ax_f = ax.secondary_xaxis('top')
    ax_f.set_xticks(np.arange(0,110,10))
    ax_f.set_xticklabels(np.array([i.astype(int).astype(str) for i in np.arange(0,110,10)]), fontsize=fs)#
    ax_f.xaxis.set_tick_params(length=10)

    [ax.axvline(x=i, color='lightgray', linestyle='dashed', lw=1.25) for i in np.arange(0,110,10)]


    for j,x_pos in enumerate(x_positions):
        ax.text(values_new[j,0]/2-5.5, x_pos-(0.45-add), str_new[j], fontsize=fs, color=is_bright(palette[cells_new[j,0]]))       
        ax.text(values_old[j,0]/2-5.5, x_pos+(0.05+add), str_old[j], fontsize=fs, color=is_bright(palette[cells_old[j,0]]))        

    pos_old = np.cumsum(np.insert(values_old, 0, 0, axis=1), axis=1) + np.insert(values_old, -1, 0, axis=1)/2 - 1 
    pos_new = np.cumsum(np.insert(values_new, 0, 0, axis=1), axis=1) + np.insert(values_new, -1, 0, axis=1)/2 - 1 

    def ret_val(value):
        if np.abs(value) >= 10:
            return 1.25
        else:
            return 0

    for i in range(1, 6):
        for j,x_pos in enumerate(x_positions):
            if values_new[j,i] > 4:
                lab = np.where(cells_new[j,i] == all_labels)[0][0]+1
                ax.text(pos_new[j,i]-ret_val(lab), x_pos-(0.45-add), str(lab), fontsize=fs, color=is_bright(palette[cells_new[j,i]]))

            if values_old[j,i] > 4:
                lab = np.where(cells_old[j,i] == all_labels)[0][0]+1            
                ax.text(pos_old[j,i]-ret_val(lab), x_pos+(0.05+add), str(lab), fontsize=fs, color=is_bright(palette[cells_old[j,i]]))        

    nn_old = round(np.mean([df_knn.loc[cell_type, cell_type] for cell_type in common_labels]),1)
    nn_new = round(np.mean([df_model.loc[cell_type, cell_type] for cell_type in common_labels]),1)

    ax.set_title('Neighbor search, accuracy: '+str(nn_old)+'%, ARI: '+str(knn_adj)+', MIS: '+str(knn_mis) +'.\n '
                'After training, accuracy: '+str(nn_new)+'%, ARI: '+str(model_adj)+', MIS: '+str(model_mis) +'.\n ', fontsize=fs*1.25)

    return fig, ax    


def plot_umap(model, legendncols=4, fontsize=14, dataset='liver'):

    context_labels = model.mdata.mod[model.context_dataset_key].obs[model.mdata.mod[model.context_dataset_key].uns['dataset_cell_key']].to_numpy()
    target_labels = model.mdata.mod[model.target_dataset_key].obs[model.mdata.mod[model.target_dataset_key].uns['dataset_cell_key']].to_numpy()

    palette = return_palette(dataset)
    if palette == None:
        palette = create_palette(np.concatenate((context_labels,target_labels)))

    colors_A = [palette[i] for i in context_labels]
    colors_B = [palette[i] for i in target_labels]

    fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=(12,9.5), constrained_layout=True, dpi=250, layout='constrained') 
    ax_A.set_title(model.mdata.mod[model.context_dataset_key].uns['dataset_name']+ ' (context)', fontsize=fontsize*1.5)
    ax_B.set_title(model.mdata.mod[model.target_dataset_key].uns['dataset_name']+ ' (target)', fontsize=fontsize*1.5)

    umap_together = ad.AnnData(np.concatenate((model.mdata.mod[model.context_dataset_key].obsm['latent_mu'], model.mdata.mod[model.target_dataset_key].obsm['latent_mu'])))
    sc.pp.neighbors(umap_together, n_neighbors=10)
    sc.tl.umap(umap_together, min_dist=0.5, spread=1.0)
    umap_together = umap_together.obsm['X_umap']

    ax_B.scatter(umap_together[:len(context_labels)][:,0], umap_together[:len(context_labels)][:,1], s=40000/len(colors_A), c='lightgray')
    ax_A.scatter(umap_together[len(context_labels):][:,0], umap_together[len(context_labels):][:,1], s=40000/len(colors_B), c='lightgray')

    ax_A.scatter(umap_together[:len(context_labels)][:,0], umap_together[:len(context_labels)][:,1], s=40000/len(colors_A), c=colors_A)
    ax_B.scatter(umap_together[len(context_labels):][:,0], umap_together[len(context_labels):][:,1], s=40000/len(colors_B), c=colors_B)

    ax_A.set_xticks([])
    ax_A.set_yticks([])
    ax_B.set_xticks([])
    ax_B.set_yticks([])

    l = np.unique(np.concatenate((context_labels, target_labels)))
    legend = fig.legend(handles=[mpatches.Patch(color=palette[cell], label=str(i+1)+': '+latex_str(cell)) for i,cell in enumerate(l)], loc='outside lower left', fontsize=13.2, ncol=legendncols, columnspacing=0.1)

    legend.get_frame().set_linewidth(1)  
    legend.get_frame().set_edgecolor('black') 


    cell_index_A = {c : np.where(context_labels == c)[0] for c in np.unique(context_labels)}
    cell_index_B = {c : np.where(target_labels == c)[0] for c in np.unique(target_labels)}

    centroids_A = np.stack([np.mean(umap_together[:len(context_labels)][cell_index_A[key]], axis=0) for key in cell_index_A.keys()])
    l_A = [str(np.where(np.array(l)==k)[0][0]+1) for k in np.unique(context_labels)]
    ta.allocate_text(fig, ax_A, x=centroids_A[:,0], y=centroids_A[:,1], text_list=l_A, x_scatter=umap_together[:len(context_labels)][:,0], y_scatter=umap_together[:len(context_labels)][:,1], 
                    nbr_candidates=200, verbose=True, textsize=fontsize, min_distance = 0.01, max_distance=0.25, margin=0.01, linecolor='#5a5255', linewidth=1)#,           

    centroids_B = np.stack([np.mean(umap_together[len(context_labels):][cell_index_B[key]], axis=0) for key in cell_index_B.keys()])
    l_B = [str(np.where(np.array(l)==k)[0][0]+1) for k in np.unique(target_labels)]
    ta.allocate_text(fig, ax_B, x=centroids_B[:,0], y=centroids_B[:,1], text_list=l_B, x_scatter=umap_together[len(context_labels):][:,0], y_scatter=umap_together[len(context_labels):][:,1], 
                    nbr_candidates=200, verbose=True, textsize=fontsize, min_distance = 0.01, max_distance=0.25, margin=0.01, linecolor='#5a5255', linewidth=1)#, 
    

def plot_lfc(model, alpha = 0.9):
    nde = []
    up = []
    down = []
    uplp = []
    downlp = []
    delta = model.mdata.mod[model.context_dataset_key].uns['lfc_delta']
    common_labels = list(model.mdata.mod[model.context_dataset_key].uns['lfc_df'].columns)
    gene_names = np.array(model.mdata.mod[model.context_dataset_key].uns['lfc_df'].index)

    fig = plt.figure(figsize=(13, 8*int(np.ceil(len(common_labels)/4))/4))
    for i,cell_type in enumerate(common_labels):

        ax = fig.add_subplot(int(np.ceil(len(common_labels)/4)), 4, i+1)

        
        lfcs = np.array(model.mdata.mod[model.context_dataset_key].uns['lfc_df'][cell_type])
        prob = np.array(model.mdata.mod[model.context_dataset_key].uns['prob_df'][cell_type])

        colors = np.array(['tab:gray'] * len(lfcs), dtype='<U20')
        colors[np.where(lfcs>delta)[0]] = 'tab:red' 
        colors[np.where(lfcs<-delta)[0]] = 'tab:blue' 
        colors[np.setdiff1d(np.where(prob<alpha)[0], np.where(lfcs<-delta)[0])] = 'salmon' 
        colors[np.setdiff1d(np.where(prob<alpha)[0], np.where(lfcs>delta)[0])] = 'lightskyblue' 
        colors[np.where(np.abs(lfcs)<delta)[0]] = 'tab:gray' 

        nde.append(len(np.where(colors == 'tab:gray')[0])*100/len(colors))
        up.append(len(np.where(colors == 'tab:red')[0])*100/len(colors))
        down.append(len(np.where(colors == 'tab:blue')[0])*100/len(colors))
        uplp.append(len(np.where(colors == 'salmon')[0])*100/len(colors))
        downlp.append(len(np.where(colors == 'lightskyblue')[0])*100/len(colors))

        ax.set_title(latex_str(common_labels[i]), fontsize=15)

        ax.scatter(lfcs, prob, s=0.25, c=colors)

        ax.text(0.95, 0.7, '{}'.format(gene_names[np.argsort(lfcs)][-7]), color='tab:red', transform=ax.transAxes, horizontalalignment='right', verticalalignment='center', fontstyle='italic')
        ax.text(0.95, 0.6, '{}'.format(gene_names[np.argsort(lfcs)][-6]), color='tab:red', transform=ax.transAxes, horizontalalignment='right', verticalalignment='center', fontstyle='italic')
        ax.text(0.95, 0.5, '{}'.format(gene_names[np.argsort(lfcs)][-5]), color='tab:red', transform=ax.transAxes, horizontalalignment='right', verticalalignment='center', fontstyle='italic')
        ax.text(0.95, 0.4, '{}'.format(gene_names[np.argsort(lfcs)][-4]), color='tab:red', transform=ax.transAxes, horizontalalignment='right', verticalalignment='center', fontstyle='italic') 
        ax.text(0.95, 0.3, '{}'.format(gene_names[np.argsort(lfcs)][-3]), color='tab:red', transform=ax.transAxes, horizontalalignment='right', verticalalignment='center', fontstyle='italic')
        ax.text(0.95, 0.2, '{}'.format(gene_names[np.argsort(lfcs)][-2]), color='tab:red', transform=ax.transAxes, horizontalalignment='right', verticalalignment='center', fontstyle='italic')    
        ax.text(0.95, 0.1, '{}'.format(gene_names[np.argsort(lfcs)][-1]), color='tab:red', transform=ax.transAxes, horizontalalignment='right', verticalalignment='center', fontstyle='italic')

        ax.text(0.05, 0.7, '{}'.format(gene_names[np.argsort(lfcs)][6]), color='tab:blue', transform=ax.transAxes, horizontalalignment='left', verticalalignment='center', fontstyle='italic')
        ax.text(0.05, 0.6, '{}'.format(gene_names[np.argsort(lfcs)][5]), color='tab:blue', transform=ax.transAxes, horizontalalignment='left', verticalalignment='center', fontstyle='italic')
        ax.text(0.05, 0.5, '{}'.format(gene_names[np.argsort(lfcs)][4]), color='tab:blue', transform=ax.transAxes, horizontalalignment='left', verticalalignment='center', fontstyle='italic')
        ax.text(0.05, 0.4, '{}'.format(gene_names[np.argsort(lfcs)][3]), color='tab:blue', transform=ax.transAxes, horizontalalignment='left', verticalalignment='center', fontstyle='italic') 
        ax.text(0.05, 0.3, '{}'.format(gene_names[np.argsort(lfcs)][2]), color='tab:blue', transform=ax.transAxes, horizontalalignment='left', verticalalignment='center', fontstyle='italic')
        ax.text(0.05, 0.2, '{}'.format(gene_names[np.argsort(lfcs)][1]), color='tab:blue', transform=ax.transAxes, horizontalalignment='left', verticalalignment='center', fontstyle='italic')    
        ax.text(0.05, 0.1, '{}'.format(gene_names[np.argsort(lfcs)][0]), color='tab:blue', transform=ax.transAxes, horizontalalignment='left', verticalalignment='center', fontstyle='italic')

        ax.hlines(alpha, np.min(lfcs)*1.05, np.max(lfcs)*1.05, color='black', ls='--')
        ax.vlines(-delta, np.min(prob)*1.05, np.max(prob)*1.05, color='black', ls='--')
        ax.vlines(delta, np.min(prob)*1.05, np.max(prob)*1.05, color='black', ls='--')

    nde = np.stack(nde)
    up = np.stack(up)
    down = np.stack(down)
    uplp = np.stack(uplp)
    downlp = np.stack(downlp)

    plt.suptitle('Median |LFC|>{}: {}%\n Up regulated, p≤{}: {}%, p>{}: {}%\n Down regulated, p≤{}: {}%, p>{}: {}%'.format(
        delta, np.round(100-np.mean(nde),1), alpha, np.round(np.mean(uplp),1), alpha, np.round(np.mean(up),1), alpha, np.round(np.mean(downlp),1), alpha, np.round(np.mean(down),1)), fontsize=20)

    plt.tight_layout()