import glasbey
import numpy as np
import textalloc as ta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import anndata as ad
import scanpy as sc
import torch

def return_palette(names):
        
        known_cell_types = {
            'B Cells': '#964b00',
            'Basophils': '#000000',
            'CD4+ KLRB1 T': '#00fa9a',
            'Capsule Fibroblasts': '#cf1020',
            'Central Vein ECs': '#fcc200',
            'Cholangiocytes': '#c90016',
            'Circ. Eff. Memory T': '#195905',
            'Circulating NK': '#d9e650',
            'Cytotoxic CD8+': '#00ff00',
            'Endothelial': '#ffbf00',
            'Fibroblast 1': '#f08080',
            'Fibroblast 2': '#ff4500',
            'Fibroblasts': '#ff3800',
            'Gamma delta T Cells': '#9acd32',
            'Hepatocytes': '#ff0090',
            'ILCs': '#7fff00',
            'Imm. LAMs': '#318ce7',
            'KCs': '#00bfff',
            'LSECs': '#ffa700',
            'Lymphatic ECs': '#ed7000',
            'Mat. LAMs': '#a1caf1',
            'Mesothelial': '#ff99af',
            'Migratory cDCs': '#8f00ff',
            'MoMac1': '#4169e1',
            'MoMac2': '#45b1e8',
            'Mono/Mono Derived': '#4f86f7',
            'Monocytes': '#89cff0',
            'NK': '#568203',
            'NK/NKT': '#839B17',
            'NKT': '#afb42b',
            'Naive CD4+ T': '#3fff00',
            'Naive CD8+ T': '#90ee90',
            'Naive/CM CD4+ T': '#85bb65',
            'Neutrophils': '#8c8784',
            'Patrolling Mono.': '#214fc6',
            'Peritoneal Mac.': '#73c2fb',
            'Plasma': '#c19a6b',
            'Portal Vein ECs': '#ffd700',
            'Pre-moKCs/moKCs': '#2a52be',
            'Regulatory T': '#b2ec5d',
            'Stellate': '#ff033e',
            'Stromal': '#ff3800',
            'T Cells': '#32cd32',
            'Th17s': '#3b7a57',
            'Th1s': '#0bda51',
            'Tissue Res. CD8+ T': '#99e6b3',
            'Tissue Res. NK': '#3cd070',
            'Trans. Mono. 1': '#0000cd',
            'Trans. Mono. 2': '#0f52ba',
            'cDC1s': '#800080',
            'cDC2s': '#bf00ff',
            'cDCs': '#a000c0',
            'pDCs': '#d891ef'
        }

        name_list = np.unique(names)
        lenght = 10 + len(name_list)
        colors = glasbey.extend_palette("tab10", palette_size=lenght)

        palette = {}
        j = 0

        for name in name_list:       
            if name in known_cell_types.keys():
                palette[name] = known_cell_types[name]
            else: 
                palette[name] = colors[j]
                j += 1

        return palette

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
    
def bar_plot(model, save_path=None, cell_key= 'cell_type_coarse', key='_aligned_latent_space', name=''):

    context_cell_labels = model.mdata.mod[model.context_dataset_key].obs[cell_key].to_numpy()
    target_cell_labels = model.mdata.mod[model.target_dataset_key].obs[cell_key].to_numpy()
    context_cell_types = np.unique(context_cell_labels)
    target_cell_types = np.unique(target_cell_labels)

    figsize=(12,0.9+len(target_cell_types)*11/16)

    fig, ax = plt.subplots(figsize=figsize, dpi=250)
    add = 0.01
    fs=20

    context_and_target_labels = np.unique(np.concatenate((context_cell_types,target_cell_types)))
    palette = return_palette(np.concatenate((context_cell_types,target_cell_types)))

    legend = fig.legend(handles=[mpatches.Patch(color=palette[cell], label=str(i+1)+': '+cell) for i,cell in enumerate(context_and_target_labels) if cell in context_cell_types], loc='lower center', bbox_to_anchor=(0.5, -0.07), fontsize=fs*0.8, ncol=4, columnspacing=0.1)
    legend.get_frame().set_linewidth(1)  
    legend.get_frame().set_edgecolor('black') 

    knn_adj = round(model.mdata.mod[model.target_dataset_key].uns['metrics_'+cell_key]['adjusted_rand_score_nns_hom_genes'],3)
    model_adj = round(model.mdata.mod[model.target_dataset_key].uns['metrics_'+cell_key]['adjusted_rand_score_nns'+key],3)

    knn_mis = round(model.mdata.mod[model.target_dataset_key].uns['metrics_'+cell_key]['adjusted_mutual_info_score_nns_hom_genes'],3)
    model_mis = round(model.mdata.mod[model.target_dataset_key].uns['metrics_'+cell_key]['adjusted_mutual_info_score_nns'+key],3)

    df_knn = model.mdata.mod[model.target_dataset_key].uns['prediction_df_nns_hom_genes'+'_'+cell_key]
    df_model = model.mdata.mod[model.target_dataset_key].uns['prediction_df_nns'+key+'_'+cell_key]

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
    ax.set_yticklabels(np.array([str(np.where(np.array(lab)==s)[0][0]+1)+': '+s for s in np.flip(df_knn.index)]), rotation=50, ha='right', fontsize=fs*0.95)#
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
            return 0.15
        else:
            return -0.25

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

    ax.set_title('Data-level neighbor search. Accuracy: '+str(nn_old)+'%, ARI: '+str(knn_adj)+', MI: '+str(knn_mis) +'.\n '
                'Latent space neighbor search. Accuracy: '+str(nn_new)+'%, ARI: '+str(model_adj)+', MI: '+str(model_mis) +'.\n ', fontsize=fs*1.25)

    if save_path != None:
        plt.savefig(save_path+'/figures/nns_accuracy_'+cell_key+'_'+name+'.jpeg', bbox_inches='tight')   

def plot_umap_three_species(model_1, model_2, context_cell_key = 'cell_type_coarse', target_cell_key_1 = 'cell_type_coarse', target_cell_key_2 = 'cell_type_coarse',  fontsize = 17, iterations = 200, save_path=None, name=''):

    model_1.initialize_prototypes(context_cell_key = context_cell_key, target_cell_key = target_cell_key_1)
    model_2.initialize_prototypes(context_cell_key = context_cell_key, target_cell_key = target_cell_key_2)
    
    with torch.no_grad():
        model_1.context_encoder_inner.eval()
        model_1.context_encoder_outer.eval()
        model_1.target_encoder_inner.eval()
        model_1.target_encoder_outer.eval()
        model_2.target_encoder_inner.eval()
        model_2.target_encoder_outer.eval()

        context_prototypes = model_1.context_encoder_inner.encode(model_1.context_encoder_outer(model_1.context_mean, model_1.context_label))[0].cpu().numpy()
        target_prototypes_1 = model_1.target_encoder_inner.encode(model_1.target_encoder_outer(model_1.target_mean, model_1.target_label))[0].cpu().numpy()
        target_prototypes_2 = model_2.target_encoder_inner.encode(model_2.target_encoder_outer(model_2.target_mean, model_2.target_label))[0].cpu().numpy()    

    print('Computing UMAP coordinates.')
    umap_together = ad.AnnData(np.concatenate((model_1.mdata.mod[model_1.context_dataset_key].obsm['latent_mu'], model_1.mdata.mod[model_1.target_dataset_key].obsm['latent_mu'], model_2.mdata.mod[model_2.target_dataset_key].obsm['latent_mu'], context_prototypes, target_prototypes_1, target_prototypes_2), axis=0))
    sc.pp.neighbors(umap_together, n_neighbors=9)
    sc.tl.umap(umap_together, min_dist=0.5, spread=1.0)
    print('Done.')    

    n_obs_c = model_1.mdata.mod[model_1.context_dataset_key].n_obs
    n_obs_1 = model_1.mdata.mod[model_1.target_dataset_key].n_obs    
    n_obs_2 = model_2.mdata.mod[model_2.target_dataset_key].n_obs   
    n_obs_pc = len(context_prototypes)
    n_obs_p1 = len(target_prototypes_1)
    n_obs_p2 = len(target_prototypes_2)

    umap_c = umap_together.obsm['X_umap'][:n_obs_c]
    umap_1 = umap_together.obsm['X_umap'][n_obs_c:n_obs_c+n_obs_1]
    umap_2 = umap_together.obsm['X_umap'][n_obs_c+n_obs_1:n_obs_c+n_obs_1+n_obs_2]

    umap_c_prototypes = umap_together.obsm['X_umap'][n_obs_c+n_obs_1+n_obs_2:n_obs_c+n_obs_1+n_obs_2+n_obs_pc]
    umap_1_prototypes = umap_together.obsm['X_umap'][n_obs_c+n_obs_1+n_obs_2+n_obs_pc:n_obs_c+n_obs_1+n_obs_2+n_obs_pc+n_obs_p1]
    umap_2_prototypes = umap_together.obsm['X_umap'][n_obs_c+n_obs_1+n_obs_2+n_obs_pc+n_obs_p1:]

    cell_labels_c = model_1.mdata.mod[model_1.context_dataset_key].obs[context_cell_key]
    cell_labels_1 = model_1.mdata.mod[model_1.target_dataset_key].obs[target_cell_key_1]
    cell_labels_2 = model_2.mdata.mod[model_2.target_dataset_key].obs[target_cell_key_2]
    labels = np.unique(np.concatenate((cell_labels_c, cell_labels_1, cell_labels_2)))

    palette = return_palette(labels)

    figsize=(17.225,6.5)
    dpi=250
    fig, (ax_A, ax_B, ax_C) = plt.subplots(1, 3, figsize=figsize, constrained_layout=True, dpi=dpi, layout='constrained') 

    legend = fig.legend(handles=[mpatches.Patch(color=palette[cell], label=str(i+1)+': '+cell) for i,cell in enumerate(labels)], loc='outside lower left', fontsize=fontsize*0.9, ncol=5, columnspacing=0.15)
    legend.get_frame().set_linewidth(1)  
    legend.get_frame().set_edgecolor('black') 
    plt.gcf().canvas.draw()
    bbox = legend.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    height = bbox.height

    fig.set_size_inches(12.225,6+height)

    colors_c = [palette[i] for i in cell_labels_c]
    colors_1 = [palette[i] for i in cell_labels_1]
    colors_2 = [palette[i] for i in cell_labels_2]

    ax_A.set_title('Context dataset', fontsize=fontsize*1.3)
    ax_B.set_title('Target dataset 1', fontsize=fontsize*1.3)
    ax_C.set_title('Target dataset 2', fontsize=fontsize*1.3)

    perm_c = np.random.permutation(len(colors_c))
    perm_1 = np.random.permutation(len(colors_1))
    perm_2 = np.random.permutation(len(colors_2))

    ax_A.scatter(umap_together.obsm['X_umap'][:,0], umap_together.obsm['X_umap'][:,1], s=40000/len(colors_c), c='lightgray')
    ax_B.scatter(umap_together.obsm['X_umap'][:,0], umap_together.obsm['X_umap'][:,1], s=40000/len(colors_1), c='lightgray')
    ax_C.scatter(umap_together.obsm['X_umap'][:,0], umap_together.obsm['X_umap'][:,1], s=40000/len(colors_2), c='lightgray')

    ax_A.scatter(umap_c[perm_c,0], umap_c[perm_c,1], s=40000/len(colors_c), c=np.array(colors_c)[perm_c])
    ax_B.scatter(umap_1[perm_1,0], umap_1[perm_1,1], s=40000/len(colors_1), c=np.array(colors_1)[perm_1])
    ax_C.scatter(umap_2[perm_2,0], umap_2[perm_2,1], s=40000/len(colors_2), c=np.array(colors_2)[perm_2])

    ax_A.scatter(umap_c_prototypes[:,0], umap_c_prototypes[:,1], s=30, linewidths=2, edgecolors='black', c=[palette[i] for i in np.unique(cell_labels_c)])
    ax_B.scatter(umap_1_prototypes[:,0], umap_1_prototypes[:,1], s=30, linewidths=2, edgecolors='black', c=[palette[i] for i in np.unique(cell_labels_1)])
    ax_C.scatter(umap_2_prototypes[:,0], umap_2_prototypes[:,1], s=30, linewidths=2, edgecolors='black', c=[palette[i] for i in np.unique(cell_labels_2)])

    ax_A.set_xticks([])
    ax_A.set_yticks([])
    ax_B.set_xticks([])
    ax_B.set_yticks([])
    ax_C.set_xticks([])
    ax_C.set_yticks([])

    ax_A.spines['top'].set_visible(False)
    ax_A.spines['right'].set_visible(False)
    ax_A.spines['bottom'].set_visible(False)
    ax_A.spines['left'].set_visible(False)

    ax_B.spines['top'].set_visible(False)
    ax_B.spines['right'].set_visible(False)
    ax_B.spines['bottom'].set_visible(False)
    ax_B.spines['left'].set_visible(False)

    ax_C.spines['top'].set_visible(False)
    ax_C.spines['right'].set_visible(False)
    ax_C.spines['bottom'].set_visible(False)
    ax_C.spines['left'].set_visible(False)

    l_A = [str(np.where(np.array(labels)==k)[0][0]+1) for k in np.unique(cell_labels_c)]
    ta.allocate_text(fig, ax_A, x=umap_c_prototypes[:,0], y=umap_c_prototypes[:,1], text_list=l_A, x_scatter=umap_c[:,0], y_scatter=umap_c[:,1], 
                nbr_candidates=iterations, verbose=True, textsize=fontsize*0.8, min_distance=0.01, max_distance=0.3, margin=0.01, linecolor='#5a5255', linewidth=1)#,           

    l_B = [str(np.where(np.array(labels)==k)[0][0]+1) for k in np.unique(cell_labels_1)]
    ta.allocate_text(fig, ax_B, x=umap_1_prototypes[:,0], y=umap_1_prototypes[:,1], text_list=l_B, x_scatter=umap_1[:,0], y_scatter=umap_1[:,1], 
                nbr_candidates=iterations, verbose=True, textsize=fontsize*0.8, min_distance=0.01, max_distance=0.3, margin=0.01, linecolor='#5a5255', linewidth=1)#, 

    l_C = [str(np.where(np.array(labels)==k)[0][0]+1) for k in np.unique(cell_labels_2)]
    ta.allocate_text(fig, ax_C, x=umap_2_prototypes[:,0], y=umap_2_prototypes[:,1], text_list=l_C, x_scatter=umap_2[:,0], y_scatter=umap_2[:,1], 
                nbr_candidates=iterations, verbose=True, textsize=fontsize*0.8, min_distance=0.01, max_distance=0.3, margin=0.01, linecolor='#5a5255', linewidth=1)#, 

    if save_path != None:
        plt.savefig(save_path+'/figures/umap_aligned_lat_space_three_species'+name+'.jpeg', bbox_inches='tight')          


def plot_umap(model, context_cell_key = 'cell_type_fine', target_cell_key = 'cell_type_fine', legendncols = 4, fontsize = 17, columnspacing = 0.15, iterations = 200, save_path=None, name=''):

    model.initialize_prototypes(context_cell_key = context_cell_key, target_cell_key = target_cell_key)
    with torch.no_grad():
        model.context_encoder_inner.eval()
        model.context_encoder_outer.eval()
        model.target_encoder_inner.eval()
        model.target_encoder_outer.eval()
        context_prototypes = model.context_encoder_inner.encode(model.context_encoder_outer(model.context_mean, model.context_label))[0].cpu().numpy()
        target_prototypes = model.target_encoder_inner.encode(model.target_encoder_outer(model.target_mean, model.target_label))[0].cpu().numpy()

    print('Computing UMAP coordinates.')
    umap_together = ad.AnnData(np.concatenate((model.mdata.mod[model.context_dataset_key].obsm['latent_mu'], model.mdata.mod[model.target_dataset_key].obsm['latent_mu'], context_prototypes, target_prototypes), axis=0))
    sc.pp.neighbors(umap_together, n_neighbors=9)
    sc.tl.umap(umap_together, min_dist=0.5, spread=1.0)
    print('Done.')

    n_obs_context = model.mdata.mod[model.context_dataset_key].n_obs
    n_obs_target = model.mdata.mod[model.target_dataset_key].n_obs    
    umap_mouse = umap_together.obsm['X_umap'][:n_obs_context]
    umap_human = umap_together.obsm['X_umap'][n_obs_context:n_obs_context+n_obs_target]
    umap_mouse_prototypes = umap_together.obsm['X_umap'][n_obs_context+n_obs_target:n_obs_context+n_obs_target+len(context_prototypes)]
    umap_human_prototypes = umap_together.obsm['X_umap'][n_obs_context+n_obs_target+len(context_prototypes):]

    mouse_cell_labels = model.mdata.mod[model.context_dataset_key].obs[context_cell_key]
    human_cell_labels = model.mdata.mod[model.target_dataset_key].obs[target_cell_key]
    labels = np.unique(np.concatenate((mouse_cell_labels, human_cell_labels)))

    palette = return_palette(labels)

    figsize=(12.225,6.5)
    dpi=250
    fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True, dpi=dpi, layout='constrained') 

    legend = fig.legend(handles=[mpatches.Patch(color=palette[cell], label=str(i+1)+': '+cell) for i,cell in enumerate(labels)], loc='outside lower left', fontsize=fontsize*0.9, ncol=legendncols, columnspacing=columnspacing)
    legend.get_frame().set_linewidth(1)  
    legend.get_frame().set_edgecolor('black') 
    plt.gcf().canvas.draw()
    bbox = legend.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    height = bbox.height

    fig.set_size_inches(12.225,6+height)

    colors_mouse = [palette[i] for i in mouse_cell_labels]
    colors_human = [palette[i] for i in human_cell_labels]

    ax_A.set_title('Context dataset', fontsize=fontsize*1.3)
    ax_B.set_title('Target dataset', fontsize=fontsize*1.3)

    perm_mouse = np.random.permutation(len(colors_mouse))
    perm_human = np.random.permutation(len(colors_human))

    ax_B.scatter(umap_mouse[:,0], umap_mouse[:,1], s=40000/len(colors_mouse), c='lightgray')
    ax_A.scatter(umap_human[:,0], umap_human[:,1], s=40000/len(colors_human), c='lightgray')

    ax_A.scatter(umap_mouse[perm_mouse,0], umap_mouse[perm_mouse,1], s=40000/len(colors_mouse), c=np.array(colors_mouse)[perm_mouse])
    ax_B.scatter(umap_human[perm_human,0], umap_human[perm_human,1], s=40000/len(colors_human), c=np.array(colors_human)[perm_human])
    ax_A.scatter(umap_mouse_prototypes[:,0], umap_mouse_prototypes[:,1], s=30, linewidths=2, edgecolors='black', c=[palette[i] for i in np.unique(mouse_cell_labels)])
    ax_B.scatter(umap_human_prototypes[:,0], umap_human_prototypes[:,1], s=30, linewidths=2, edgecolors='black', c=[palette[i] for i in np.unique(human_cell_labels)])

    ax_A.set_xticks([])
    ax_A.set_yticks([])
    ax_B.set_xticks([])
    ax_B.set_yticks([])

    ax_A.spines['top'].set_visible(False)
    ax_A.spines['right'].set_visible(False)
    ax_A.spines['bottom'].set_visible(False)
    ax_A.spines['left'].set_visible(False)

    ax_B.spines['top'].set_visible(False)
    ax_B.spines['right'].set_visible(False)
    ax_B.spines['bottom'].set_visible(False)
    ax_B.spines['left'].set_visible(False)

    l_A = [str(np.where(np.array(labels)==k)[0][0]+1) for k in np.unique(mouse_cell_labels)]
    ta.allocate_text(fig, ax_A, x=umap_mouse_prototypes[:,0], y=umap_mouse_prototypes[:,1], text_list=l_A, x_scatter=umap_mouse[:,0], y_scatter=umap_mouse[:,1], 
                    nbr_candidates=iterations, verbose=True, textsize=fontsize*0.8, min_distance=0.01, max_distance=0.3, margin=0.01, linecolor='#5a5255', linewidth=1)#,           

    l_B = [str(np.where(np.array(labels)==k)[0][0]+1) for k in np.unique(human_cell_labels)]
    ta.allocate_text(fig, ax_B, x=umap_human_prototypes[:,0], y=umap_human_prototypes[:,1], text_list=l_B, x_scatter=umap_human[:,0], y_scatter=umap_human[:,1], 
                    nbr_candidates=iterations, verbose=True, textsize=fontsize*0.8, min_distance=0.01, max_distance=0.3, margin=0.01, linecolor='#5a5255', linewidth=1)#, 

    if save_path != None:
        plt.savefig(save_path+'/figures/umap_aligned_lat_space_'+name+'.jpeg', bbox_inches='tight')       


def plot_lfc(model, save_path, alpha = 0.9, name=''):
    nde = []
    up = []
    down = []
    uplp = []
    downlp = []
    delta = model.mdata.mod[model.context_dataset_key].uns['lfc_delta']
    common_labels = list(model.mdata.mod[model.context_dataset_key].uns['lfc_df'].columns)
    gene_names = np.array(model.mdata.mod[model.context_dataset_key].uns['lfc_df'].index)

    fig = plt.figure(figsize=(13, 8*int(np.ceil(len(common_labels)/4))/4+2.5/np.ceil(len(common_labels)) ))
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

        ax.set_title(common_labels[i], fontsize=15)

        ax.scatter(lfcs, prob, s=0.25, c=colors)
        gene_name_sort = gene_names[np.argsort(lfcs)]
        ax.text(0.95, 0.7, gene_name_sort[-7], color='tab:red', transform=ax.transAxes, horizontalalignment='right', verticalalignment='center')
        ax.text(0.95, 0.6, gene_name_sort[-6], color='tab:red', transform=ax.transAxes, horizontalalignment='right', verticalalignment='center')
        ax.text(0.95, 0.5, gene_name_sort[-5], color='tab:red', transform=ax.transAxes, horizontalalignment='right', verticalalignment='center')
        ax.text(0.95, 0.4, gene_name_sort[-4], color='tab:red', transform=ax.transAxes, horizontalalignment='right', verticalalignment='center') 
        ax.text(0.95, 0.3, gene_name_sort[-3], color='tab:red', transform=ax.transAxes, horizontalalignment='right', verticalalignment='center')
        ax.text(0.95, 0.2, gene_name_sort[-2], color='tab:red', transform=ax.transAxes, horizontalalignment='right', verticalalignment='center')    
        ax.text(0.95, 0.1, gene_name_sort[-1], color='tab:red', transform=ax.transAxes, horizontalalignment='right', verticalalignment='center')

        ax.text(0.05, 0.7, gene_name_sort[6], color='tab:blue', transform=ax.transAxes, horizontalalignment='left', verticalalignment='center')
        ax.text(0.05, 0.6, gene_name_sort[5], color='tab:blue', transform=ax.transAxes, horizontalalignment='left', verticalalignment='center')
        ax.text(0.05, 0.5, gene_name_sort[4], color='tab:blue', transform=ax.transAxes, horizontalalignment='left', verticalalignment='center')
        ax.text(0.05, 0.4, gene_name_sort[3], color='tab:blue', transform=ax.transAxes, horizontalalignment='left', verticalalignment='center') 
        ax.text(0.05, 0.3, gene_name_sort[2], color='tab:blue', transform=ax.transAxes, horizontalalignment='left', verticalalignment='center')
        ax.text(0.05, 0.2, gene_name_sort[1], color='tab:blue', transform=ax.transAxes, horizontalalignment='left', verticalalignment='center')    
        ax.text(0.05, 0.1, gene_name_sort[0], color='tab:blue', transform=ax.transAxes, horizontalalignment='left', verticalalignment='center')

        ax.hlines(alpha, np.min(lfcs)*1.05, np.max(lfcs)*1.05, color='black', ls='--')
        ax.vlines(-delta, np.min(prob)*1.05, np.max(prob)*1.05, color='black', ls='--')
        ax.vlines(delta, np.min(prob)*1.05, np.max(prob)*1.05, color='black', ls='--')

    nde = np.stack(nde)
    up = np.stack(up)
    down = np.stack(down)
    uplp = np.stack(uplp)
    downlp = np.stack(downlp)

    plt.suptitle('Median |LFC|>'+'{}: {}%\n Up regulated, p≤{}: {}%, p>{}: {}%\n Down regulated, p≤{}: {}%, p>{}: {}%'.format(
        delta, np.round(100-np.mean(nde),1), alpha, np.round(np.mean(uplp),1), alpha, np.round(np.mean(up),1), alpha, np.round(np.mean(downlp),1), alpha, np.round(np.mean(down),1)), fontsize=20)

    plt.tight_layout()

    if save_path!=None:
        plt.savefig(save_path+'/figures/LFC_'+name+'.jpeg', bbox_inches='tight')


def plot_prototypes(model, save_key=None, name=''):
    labels_c = model.mdata.mod[model.context_dataset_key].obs[model.mdata.mod[model.context_dataset_key].uns['dataset_cell_key']].to_numpy()
    labels_t = model.mdata.mod[model.target_dataset_key].obs[model.mdata.mod[model.target_dataset_key].uns['dataset_cell_key']].to_numpy()
    _, ind_t, ind_c = np.intersect1d(np.unique(labels_t), np.unique(labels_c), return_indices=True)

    fontsize = 20
    cols=3
    rows=int(np.ceil(len(ind_t)/cols))
    figsize=(12, 1+rows*3)

    palette = return_palette(labels_c)

    num_iter = len(np.stack(model.nlog_likeli_neighbors))

    x_ticks = [10**i for i in range(1, len(str(num_iter)))]
    x_tick_labels = [str(10**i) for i in range(1, len(str(num_iter)))]

    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    list_1 = []
    for j, c in enumerate(ind_t):
        row = j % rows
        col = j // rows
        data = np.stack(model.nlog_likeli_neighbors)[:,c,:]  
        list_2 = []
        k=0  
        for i in range(data.shape[1]):
            if i != ind_c[j]:
                averaged_plot = np.array([np.mean(data[:, i][k-min(int(5+min(k,5)+k*0.1), 6000)+1:k+1]) for k in range(len(data))])
                list_2.append(averaged_plot)
                axs[row, col].plot(averaged_plot, c=palette[np.unique(labels_c)[i]], alpha=0.5)
                k+=1 
                
        i = ind_c[j]
        averaged_plot = np.array([np.mean(data[:, i][k-min(int(5+min(k,5)+k*0.1), 6000)+1:k+1]) for k in range(len(data))])
        list_2.append(averaged_plot) 
        list_1.append(list_2)
        axs[row, col].plot(averaged_plot, c=palette[np.unique(labels_c)[i]], linewidth=4.25)                      

        axs[row, col].set_xscale('log')
        axs[row, col].set_xticks(x_ticks)
        axs[row, col].set_xticklabels(x_tick_labels)
        y_ticks = [str(int(y)) for y in axs[row, col].get_yticks()]   
        axs[row, col].set_yticklabels(y_ticks)    
        axs[row, col].tick_params(axis='both', labelsize=12.5)     
        axs[row, col].set_title(np.unique(labels_t)[c], fontsize=19)

    fig.legend()
    fig.tight_layout()

    legend = fig.legend(handles=[mpatches.Patch(color=palette[cell], label=cell) for cell in np.unique(labels_c)], fontsize=fontsize, loc='upper left', bbox_to_anchor=(0.99, 0.983), labelspacing=0.64)
    legend.get_frame().set_linewidth(1)  
    legend.get_frame().set_edgecolor('black') 

    if save_key != None:
        plt.savefig(save_key+'/figures/likelihood_prototypes_'+name+'.jpeg', bbox_inches='tight') 
