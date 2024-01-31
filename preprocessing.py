from typing import Union
import os

import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import muon as mu

from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors

class create_mdata():
    def __init__(self,        
                 context_adata: ad.AnnData,
                 context_batch_key: str,
                 context_cell_key: str,      
                 context_dataset_name : str = 'mouse',
                 context_gene_naming_convention: str = 'mouse', 
                 context_n_top_genes: Union[int, None] = 4000,                   
                 min_non_zero_genes: float = 0.025, 
                 max_genes_expr: float = 1,                                   
                 min_cell_type_size: Union[int, None] = 20,   
                 max_cell_type_size: Union[int, None] = None,                   
                 min_batch_size: int = 20, 
                 max_dataset_size: Union[int, None] = None,   
                 ): 
        """
        Initializes and configures a multi-dataset analysis (mdata) object for scPecies.

        This class is designed to process and integrate multiple AnnData objects, facilitating comparative
        analysis across different datasets (e.g., between species). It includes methods for filtering cells,
        encoding batch labels, computing library size normalization parameters, translating gene names between
        species, and identifying highly variable genes, among others.

        Parameters:
        context_adata (ad.AnnData): AnnData object of the context dataset.
        context_batch_key (str): Key for storing experimental batch effects in the context dataset.
        context_cell_key (str): Key for storing cell type labels in the context dataset.
        context_dataset_name (str): Designated name for the context dataset in the mdata object. 
        context_gene_naming_convention ('mouse', 'human'): Gene naming convention ('mouse' or 'human') for the context dataset.
        context_n_top_genes (int > 0 or None): Number of top highly variable genes to select in the context dataset. Set to None to skip this computation.
        min_non_zero_genes (0 <= float <= 1): Minimum required fraction of non-zero genes in a cell for it to be retained.
        min_cell_type_size (int >= 0): Minimum size threshold for retaining a cell type in the dataset.
        min_batch_size (int >= 0): Minimum size threshold for retaining an experimental batch effect.

        Methods:
        save_mdata: Saves the processed mdata object.
        setup_target_adata: Prepares and processes a target AnnData object for comparison and analysis.
        create_translation_dict: Creates a dictionary to translate gene names between mouse and human conventions.
        filter_cells: Filters cells based on gene expression and cell type size criteria.
        compute_library_prior_params: Computes prior parameters for scVI's library size encoder.
        translate_gene_names: Translates gene names between mouse and human conventions.
        encode_batch_labels: Encodes and filters experimental batch effects.
        subset_to_highly_variable_genes: Selects and retains highly variable genes in the dataset.
        """
        self.create_translation_dict()

        self.context_dataset_name = context_dataset_name
        context_adata.uns['dataset_batch_key'] = context_batch_key
        context_adata.uns['dataset_cell_key'] = context_cell_key
        context_adata.uns['dataset_name'] = context_dataset_name
        context_adata.uns['gene_naming_convention'] = context_gene_naming_convention

        context_adata = self.encode_batch_labels(context_adata, min_batch_size)
        context_adata = self.compute_library_prior_params(context_adata)        
        context_adata = self.translate_gene_names(context_adata)

        context_adata = self.filter_cells_and_genes(context_adata, 0, min_cell_type_size, max_cell_type_size, max_genes_expr, None)
        if context_n_top_genes != None:
            context_adata = self.subset_to_highly_variable_genes(context_adata, context_n_top_genes)
        context_adata = self.filter_cells_and_genes(context_adata, min_non_zero_genes, min_cell_type_size, max_cell_type_size, 1, max_dataset_size)
        self.dataset_collection = {context_adata.uns['dataset_name']: context_adata}
        print('Done!\n'+'-'*100)


    def save_mdata(self, save_path: str, name: str):  
        """
        Saves self.mdata to a given save_path under the povided name.
        Creates the directory if it does not exist.

        Parameters:
        save_path (str): The path to the folder where to save the mdata object.
        name (str): The name of the mdata object.
        """

        if not os.path.exists(save_path+'/dataset'):
            os.makedirs(save_path+'/dataset')
            print(f"\nCeated directory '{save_path}'.")

        mdata = mu.MuData(self.dataset_collection)
        mdata.write(save_path+'/dataset/'+name+'.h5mu') 
        print('Saved mdata to {}/dataset/{}.h5mu.'.format(save_path, name))


    def setup_target_adata(self,
                 target_adata: ad.AnnData,
                 target_batch_key: str,
                 target_cell_key: Union[str, None] = None,       
                 target_dataset_name: str = 'human',
                 target_gene_naming_convention: str = 'human', 
                 target_n_top_genes: Union[int, None] = 4000,                  
                 neighbors: int = 250,
                 metric: str = 'cosine',
                 compute_log1p: bool = True,   
                 min_non_zero_genes: float = 0.025, 
                 max_genes_expr: float = 1,                                   
                 min_cell_type_size: Union[int, None] = 20,   
                 max_cell_type_size: Union[int, None] = None,                   
                 min_batch_size: int = 20, 
                 max_dataset_size: Union[int, None] = None                       
                 ):   
        """
        Prepares and processes a target AnnData object for analysis.

        This method configures the target AnnData object by setting up various keys and attributes,
        filtering cells, encoding batch labels, computing library prior parameters, translating gene
        names, and subsetting to highly variable genes.
         
        This method also computes the nearest neighbors on the homologous genes with the context dataset.
        
        Parameters:
        target_adata (ad.AnnData): The AnnData object to be prepared and processed.
        target_batch_key (str): Key for batch information in 'target_adata'.
        target_cell_key (str or None): Key for cell type information in 'target_adata'. Set to None if cell labels are unknown
        target_dataset_name (str): Name of the target dataset. 
        target_gene_naming_convention ('human' or 'mouse): Gene naming convention. 
        target_n_top_genes (int > 0 or None): Number of highly variable gene of which to subset the target dataset. Set to None to skip hvg computation.
        neighbors (int > 0): Number of neighbors to consider in nearest neighbors calculation. 
        metric (str): Metric used for nearest neighbors calculation. 
        compute_log1p (bool): Whether to compute log1p transformed counts for the NNS. 
        """

        target_adata.uns['dataset_batch_key'] = target_batch_key
        target_adata.uns['dataset_cell_key'] = target_cell_key
        target_adata.uns['dataset_name'] = target_dataset_name
        target_adata.uns['gene_naming_convention'] = target_gene_naming_convention


        target_adata = self.encode_batch_labels(target_adata, min_batch_size)
        target_adata = self.compute_library_prior_params(target_adata)           
        target_adata = self.translate_gene_names(target_adata)

        target_adata = self.filter_cells_and_genes(target_adata, 0, min_cell_type_size, max_cell_type_size, max_genes_expr, None)
        if target_n_top_genes != None:
            target_adata = self.subset_to_highly_variable_genes(target_adata, target_n_top_genes)
        target_adata = self.filter_cells_and_genes(target_adata, min_non_zero_genes, min_cell_type_size, max_cell_type_size, 1, max_dataset_size)

        if target_gene_naming_convention == 'human':
            context_gene_names = self.dataset_collection[self.context_dataset_name].var['human_gene_names'].to_numpy()

        elif target_gene_naming_convention == 'mouse':
            context_gene_names = self.dataset_collection[self.context_dataset_name].var['mouse_gene_names'].to_numpy()

        _, context_ind, target_ind = np.intersect1d(context_gene_names, target_adata.var_names.to_numpy(), return_indices=True)
        
        if compute_log1p:
            print('Compute {} neighbors with the context dataset on the log1p transformed counts of {} homologous genes. Using {} metric.'.format(
                str(neighbors), str(len(context_ind)), metric))            
            context_neigh = np.log1p(self.dataset_collection[self.context_dataset_name].X.toarray()[:, context_ind])
            target_neigh = np.log1p(target_adata.X.toarray()[:, target_ind])

        else:    
            print('Compute {} neighbors with the context dataset on the counts of {} homologous genes. Using {} metric.'.format(
                str(neighbors), str(len(context_ind)), metric))              
            context_neigh = self.dataset_collection[self.context_dataset_name].X.toarray()[:, context_ind]
            target_neigh = target_adata.X.toarray()[:, target_ind]

        neigh = NearestNeighbors(n_neighbors=neighbors, metric=metric)
        neigh.fit(context_neigh)

        _, indices_whole = neigh.kneighbors(target_neigh)
        indices_whole = np.squeeze(indices_whole)

        target_adata.obsm['ind_nns_hom_genes'] = indices_whole.astype(np.int32)

        self.dataset_collection[target_adata.uns['dataset_name']] = target_adata
        print('Done!\n'+'-'*80)


    def create_translation_dict(self):
        """
        Creates a dictionary for translating gene names between human and mouse.

        This function fetches a gene dataset from the Jackson Laboratory's website,
        specifically the Mouse Genome Informatics (MGI) report on mouse-human sequence homology.
        It processes this dataset to isolate entries corresponding to human and mouse genes.
        The function then identifies genes that are common to both species and creates a
        dictionary mapping human gene names to their mouse counterparts.

        The resulting translation dictionary is stored as a DataFrame in the 'translation_dict'
        attribute of the class. This dictionary has two columns: 'gene_names_human' and
        'gene_names_mouse', representing the gene symbols in human and mouse, respectively.
        """

        print('Creating the gene name translation dictionary.')

        atlas = pd.read_csv("http://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt",sep="\t")
        h_ind = np.where(atlas.loc[:,"Common Organism Name"]=="human")[0]
        m_ind = np.where(atlas.loc[:,"Common Organism Name"]!="human")[0]
        human_atlas = atlas.iloc[h_ind]
        mouse_atlas = atlas.iloc[m_ind]
        human_gene_key = human_atlas.loc[:,"DB Class Key"].to_numpy()
        mouse_gene_key = mouse_atlas.loc[:,"DB Class Key"].to_numpy()
        _, x_ind, y_ind = np.intersect1d(human_gene_key, mouse_gene_key, return_indices=True)
        human_atlas = human_atlas.iloc[x_ind]
        mouse_atlas = mouse_atlas.iloc[y_ind]
        self.translation_dict = pd.DataFrame(data={"gene_names_human": human_atlas.loc[:,"Symbol"].to_numpy(), "gene_names_mouse": mouse_atlas.loc[:,"Symbol"].to_numpy()})  


    @staticmethod
    def filter_cells_and_genes(
                     adata: ad.AnnData, 
                     min_non_zero_genes: float, 
                     min_cell_type_size: int,
                     max_cell_type_size: Union[int, None],
                     max_genes_expr: float,
                     max_dataset_size: Union[int, None],
                     ):
        """
        Filters cells in an AnnData object based on gene expression and cell type size criteria.

        This method performs two main filtering steps on the AnnData object:
        1. It removes cells that have a lower number of non-zero genes than a specified threshold.
        The threshold is given as a fraction (min_non_zero_genes) of the total number of variables (genes) in 'adata'.
        2. If the 'dataset_cell_key' attribute in 'adata.uns' is not None, the method further filters out cell types 
        that are underrepresented based on the 'min_cell_type_size' parameter. Only cell types with a count 
        greater than 'min_cell_type_size' are retained.

        Parameters:
        adata (ad.AnnData): The AnnData object to be filtered.
        min_non_zero_genes (float): The minimum fraction of non-zero genes required for a cell to be retained.
        min_cell_type_size (int): The minimum size (number of cells) required for a cell type to be retained.

        Returns:
        ad.AnnData: The filtered AnnData object.
        """

        old_n_obs = adata.n_obs
        old_n_vars = adata.n_vars        
        sc.pp.filter_cells(adata, min_genes=adata.n_vars*min_non_zero_genes)
        sc.pp.filter_genes(adata, max_cells=adata.n_obs*max_genes_expr)

        if adata.uns['dataset_cell_key'] != None:
            cell_type_counts = adata.obs[adata.uns['dataset_cell_key']].value_counts()>min_cell_type_size
            cell_type_counts = cell_type_counts[cell_type_counts==True].index
            adata = adata[adata.obs[adata.uns['dataset_cell_key']].isin(cell_type_counts)]    

        if adata.uns['dataset_cell_key'] != None and max_cell_type_size != None:
            unique_cell_types = adata.obs[adata.uns['dataset_cell_key']].unique()
            samples_per_type = max_cell_type_size // len(unique_cell_types)
            selected_indices = []

            for cell_type in unique_cell_types:
                indices = np.where(adata.obs[adata.uns['dataset_cell_key']] == cell_type)[0]
                n_samples = min(len(indices), samples_per_type)
                selected_indices.extend(np.random.choice(indices, n_samples, replace=False))

            adata = adata[selected_indices]  

        if max_dataset_size != None:
            adata = adata[np.random.choice(np.arange(adata.n_obs), max_dataset_size, replace=False)]  

        print('Filtering cells and genes. Kept {} cells, removed {}. Kept {} genes, removed {}.'.format(
            str(adata.n_obs), str(int(old_n_obs-adata.n_obs)), str(adata.n_vars), str(int(old_n_vars-adata.n_vars))))

        return adata
    @staticmethod
    def compute_library_prior_params(adata: ad.AnnData):
        """
        Computes the batchwise prior parameters for the library encoder for each cell in an AnnData object.

        This method calculates the log mean and log standard deviation of the library size for each cell.
        The library size is defined as the total sum of gene expression counts per cell. These calculations
        are performed separately for each batch as defined by the 'dataset_batch_key' in 'adata.obs'.

        The method updates the AnnData object by adding two new columns to 'adata.obs':
        - 'library_log_mean': Contains the log mean of the library size for each cell.
        - 'library_log_std': Contains the log standard deviation of the library size for each cell.

        Parameters:
        adata (ad.AnnData): The AnnData object for which library prior parameters are computed.

        Returns:
        ad.AnnData: The updated AnnData object with new 'library_log_mean' and 'library_log_std' columns in 'adata.obs'.
        """

        print('Compute prior parameters for the library encoder.')
        library_log_mean = np.zeros(shape=(adata.n_obs, 1))
        library_log_std = np.ones(shape=(adata.n_obs, 1))  
        log_sum = np.log(adata.X.sum(axis=1))

        for batch in np.unique(adata.obs[adata.uns['dataset_batch_key']]):
            ind = np.where(adata.obs[adata.uns['dataset_batch_key']] == batch)[0]
            library_log_mean[ind]  = np.mean(log_sum[ind])
            library_log_std[ind] = np.std(log_sum[ind])   

        adata.obs['library_log_mean'] = library_log_mean.astype(np.float32) 
        adata.obs['library_log_std'] = library_log_std.astype(np.float32) 

        return adata

    def translate_gene_names(self, adata: ad.AnnData):
        """
        Translates gene names in an AnnData object from mouse to human naming conventions or vice versa.

        This function checks the 'gene_naming_convention' in 'adata.uns' to determine whether the current
        gene names are in mouse or human convention. Based on this, it translates the gene names to the
        other species using a pre-defined translation dictionary stored in 'self.translation_dict'.

        The function handles two scenarios:
        1. If the gene naming convention is 'mouse', it translates mouse gene names to human names.
        The translated names are stored in 'adata.var['human_gene_names']'.
        The original mouse gene names are preserved in 'adata.var['mouse_gene_names']'.
        2. If the gene naming convention is 'human', it translates human gene names to mouse names.
        The translated names are stored in 'adata.var['mouse_gene_names']'.
        The original human gene names are preserved in 'adata.var['human_gene_names']'.

        For genes that do not have a homolog in the other species, the function assigns a unique identifier
        starting with 'non_hom_' followed by the dataset name and a numerical index.

        Parameters:
        adata (ad.AnnData): The AnnData object containing the gene names to be translated.

        Returns:
        ad.AnnData: The updated AnnData object with new columns for translated gene names.

        Raises:
        ValueError: If the gene naming convention is not set to 'mouse' or 'human'.
        """


        if adata.uns['gene_naming_convention'] == 'mouse':
            print('Translating the gene names of the {} dataset from the mouse to human gene naming convention.'.format(adata.uns['dataset_name']))            
            gene_names = adata.var_names.to_numpy().copy()
            _, x_ind_m, y_ind_m = np.intersect1d(gene_names, self.translation_dict.loc[:,"gene_names_mouse"].to_numpy(), return_indices=True)
            gene_names = np.array(['non_hom_'+adata.uns['dataset_name']+'_'+str(i) for i in range(len(gene_names))])            
            gene_names[x_ind_m] = self.translation_dict.loc[y_ind_m, "gene_names_human"].to_numpy()
            adata.var['human_gene_names'] = gene_names             
            adata.var['mouse_gene_names'] = adata.var_names.to_numpy().copy()

        elif adata.uns['gene_naming_convention'] == 'human':
            print('Translating the gene names of the {} dataset from the human to mouse gene naming convention.'.format(adata.uns['dataset_name']))            
            gene_names = adata.var_names.to_numpy().copy()
            _, x_ind_m, y_ind_m = np.intersect1d(gene_names, self.translation_dict.loc[:,"gene_names_human"].to_numpy(), return_indices=True)
            gene_names = np.array(['non_hom_'+adata.uns['dataset_name']+'_'+str(i) for i in range(len(gene_names))])
            gene_names[x_ind_m] = self.translation_dict.loc[y_ind_m, "gene_names_mouse"].to_numpy()
            adata.var['human_gene_names'] = adata.var_names.to_numpy().copy()
            adata.var['mouse_gene_names'] = gene_names

        else:
            raise ValueError('Can translate only mouse and human gene naming conventions.')
        
        return adata
        
    @staticmethod
    def encode_batch_labels(
                            adata: ad.AnnData, 
                            min_batch_size: int
                            ):
        """
        Encodes batch labels in an AnnData object using one-hot encoding for use in a scVI model.

        This method first filters out batches in the AnnData object that have a size smaller than
        the specified 'min_batch_size'. It uses the 'dataset_batch_key' in 'adata.uns' to identify
        the batches and removes those with insufficient cell counts.

        After filtering, the method applies one-hot encoding to the batch labels. The encoded labels
        are stored in 'adata.obsm['batch_label_enc']'.

        The method also handles the 'dataset_cell_key' to create a dictionary ('batch_dict') of encoded
        batch labels for each cell type. This dict will be used to average over batches in the normalized
        gene expression values when computing the log2-fold change.
        If 'cell_key' is None (no cell types defined), a default 'unknown' category is used.

        Parameters:
        adata (ad.AnnData): The AnnData object to be processed.
        min_batch_size (int): The minimum number of cells required for a batch to be included.

        Returns:
        ad.AnnData: The AnnData object with updated batch label encodings.
        """        


        batch_key = adata.uns['dataset_batch_key']
        cell_key = adata.uns['dataset_cell_key']

        batch_counts = adata.obs[batch_key].value_counts()    
        batch_counts = list(batch_counts[batch_counts < min_batch_size].index)
        adata = adata[~adata.obs[batch_key].isin(batch_counts)]   

        batch_labels = adata.obs[batch_key].to_numpy().reshape(-1, 1)

        print('Registering experimental batches for the {} dataset. Kept {}, removed {}.'.format(
            adata.uns['dataset_name'], str(len(np.unique(batch_labels))), str(len(batch_counts))))

        enc = OneHotEncoder()
        enc.fit(batch_labels)

        adata.obsm['batch_label_enc'] = enc.transform(batch_labels).toarray().astype(np.float32) 
        
        if cell_key == None:
            batches = {'unknown': enc.transform(np.unique(batch_labels).reshape(-1, 1)).toarray().astype(np.float32)}

        else:
            cell_types = adata.obs[cell_key].cat.categories.to_numpy()
            batches = {c : adata[adata.obs[cell_key] == c].obs[batch_key].value_counts() > 3 for c in cell_types}
            batches = {c : batches[c][batches[c]].index.to_numpy() for c in cell_types}
            batches = {c : enc.transform(batches[c].reshape(-1, 1)).toarray().astype(np.float32)  for c in cell_types}
            batches['unknown'] = enc.transform(np.unique(batch_labels).reshape(-1, 1)).toarray().astype(np.float32)

        adata.uns['batch_dict'] = batches

        return adata

    @staticmethod
    def subset_to_highly_variable_genes(
                                        adata: ad.AnnData,                                       
                                        n_top_genes: int,
                                        ):
        
        """
        Subsets an AnnData object to include only the most highly variable genes.

        This method selects the top 'n_top_genes' highly variable genes based on the specified
        'seurat' method. For other hvg computation methods subset manually, pass the 
        AnnData object to the class and set n_top_genes to None.

        The method cleans up additional metadata related to high variability gene selection 
        that are stored in 'adata.var' and 'adata.uns'.

        Parameters:
        adata (ad.AnnData): The AnnData object to be processed.
        n_top_genes (int): The number of top highly variable genes to select.
        flavor (str, optional): The method used for selecting highly variable genes. Defaults to 'seurat'.

        Returns:
        ad.AnnData: The updated AnnData object with only the most highly variable genes.
        """

        print('Subsetting the {} dataset to the {} most highly variable genes using seurat.'.format(adata.uns['dataset_name'], str(n_top_genes)))

        adata.layers["raw_counts"] = adata.X.copy() 
        sc.pp.log1p(adata)

        sc.pp.highly_variable_genes(
            adata,
            batch_key=adata.uns['dataset_batch_key'],
            n_top_genes=n_top_genes,
            subset=True,
            flavor='seurat',
        )

        adata.X = adata.layers['raw_counts'].copy()#.toarray()
        del adata.layers['raw_counts']        
        
        del adata.var['highly_variable_intersection']
        del adata.var['dispersions_norm']
        del adata.var['dispersions']
        del adata.var['means']
        del adata.var['highly_variable']
        del adata.var['highly_variable_nbatches']       
        del adata.uns['log1p']
        del adata.uns['hvg'] 

        return adata    