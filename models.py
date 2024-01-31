import numpy as np
import pandas as pd
import muon as mu
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import time
from datetime import timedelta
import os
import pickle
from typing import Union, List
from collections import Counter

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import fowlkes_mallows_score, adjusted_rand_score, balanced_accuracy_score, adjusted_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi2

def create_structure(structure: List[int], 
                     layer_order: list
                     ):
    """
    Constructs a neural network structure based on specified layers and operations.

    This function takes as input a list of integers
    representing the structure (number of neurons in each layer) and a list of operations to be applied
    in each layer. The operations include linear transformation, activation functions, dropout, batch normalization,
    and layer normalization. Activation functions can be optionally bounded to a specified range. 
    The function assembles and returns the layers as a PyTorch Sequential model.

    Parameters:
    structure (List[int]): A list of integers where each integer represents the number of neurons in a layer.
    layer_order (list): A list of tuples or strings specifying the operations to be applied. Supported operations are
                        'linear', 'batch_norm', 'layer_norm', 'act' (activation), and 'dropout'.
                        For 'dropout' and 'act', a tuple must be provided with the operation name and its parameter.
                        ('dropout', dropout_rate), e.g.: ('dropout', 0.1)
                        For 'act', an optional third element (tuple) can be provided to specify activation clipping.
                        ('act', activation_function, [a, b]), e.g.: ('act', torch.nn.ReLU(), [0, 7]), ('act', 'PReLU', [-6, 6]) or ('act', torch.nn.Tanh())

                        The operations are chained in the order they appear in the list. Omit any operation that is unwanted. E.g.:
                        ['linear', 'layer_norm', ('act', nn.ReLU(), [0, 6]), ('dropout', 0.1)]
                        ['linear', ('act', 'PReLU', [-6, 6]), 'batch_norm']
                        ['linear', ('act', torch.nn.Tanh())]

    Returns:
    nn.Sequential: A PyTorch Sequential model consisting of the specified layers and operations.
    """

    layer_operations = [l if type(l) == str else l[0] for l in layer_order]
    
    expected_strings = {'linear', 'batch_norm', 'layer_norm', 'act', 'dropout'}
    if set(layer_operations).issubset(expected_strings) != True:
        raise ValueError(f"layer_order can only contain these elements {expected_strings}")

    if 'dropout' in layer_operations: 
        dr_ind = layer_operations.index('dropout')
        dropout = layer_order[dr_ind][1]
        if dropout < 0.0:
            raise ValueError("The dropout rate must be a non-negative float.  Example: ('dropout', 0.1)")

    act_ind = layer_operations.index('act')
    act = layer_order[act_ind][1]

    if len(layer_order[act_ind]) == 3:
        clip_act = layer_order[act_ind][-1]

        if len(clip_act) != 2:
            raise ValueError("clip_act must be a list of exactly two floats. Example: ('act', torch.nn.ReLU(), [0, 6])")
        
        if clip_act[1] <= clip_act[0]:
            raise ValueError("The second float of clip_act must be greater than the first. Example: ('act', torch.nn.ReLU(), [0, 6])")

    layers = []
    for neurons_in, neurons_out in zip(structure, structure[1:]):
        for operation in layer_operations:
            if operation == 'linear':
                layers.append(nn.Linear(neurons_in, neurons_out))
            elif operation == 'act':
                if act == 'PReLU': act = nn.PReLU(num_parameters=neurons_out)
                else: act = act

                if clip_act != False:
                    layers.append(make_act_bounded(act, min=clip_act[0], max=clip_act[1]))
                else:
                    layers.append(act)                      
            elif operation == 'dropout':
                layers.append(nn.Dropout(dropout))
            elif operation == 'layer_norm':
                layers.append(nn.LayerNorm(neurons_out))
            elif operation == 'batch_norm':
                layers.append(nn.BatchNorm1d(neurons_out))                    
    return nn.Sequential(*layers)


class Encoder_outer(nn.Module):
    def __init__(self, 
                 param_dict: dict):
        """
        Defines the outer encoder neural network, the weight of this network are not shared between 
        context and target dataset. 

        This class uses the provided parameter dictionary to configure
        the network's structure and layers. The network is built using the 'create_structure' function
        which assembles a sequence of layers based on the provided specifications.

        Parameters:
        param_dict (dict): A dictionary containing the configuration parameters for the encoder. 
                           Expected keys include:
                           - 'data_dim': The dimensionality of the gene expression input data.
                           - 'batch_dim': The dimensionality of the experimental batch information.
                           - 'dims_enc_outer': A list of integers defining the number of neurons in the hidden layers.
                           - 'layer_order': A list specifying the types of layers and their configurations.
        """     
        super(Encoder_outer, self).__init__()
           
        structure = [param_dict['data_dim']+param_dict['batch_dim']] + param_dict['dims_enc_outer']
        self.model = create_structure(structure=structure, 
                                      layer_order=param_dict['layer_order'],
                                      )

    def forward(self, data, label_inp):
        """
        The forward pass concatenates the gene expression input data with the experimental batch information 
        along the last dimension and then passes this combined input through the model. 

        Parameters:
        data (Tensor): The input gene expression count tensor.
        label_inp (Tensor): The one-hot encoded experimental bacht label tensor.

        Returns:
        Tensor: The intermediate representation.
        """

        x = torch.cat((data, label_inp), dim=-1)
        x = self.model(x)
        return x 

    
class Encoder_inner(nn.Module):
    def __init__(self, 
                 device: str,
                 param_dict: dict
                 ):
        super(Encoder_inner, self).__init__()
        """
        This class defines the later layers of the scPecies encoder model 
        with shared weights between context and target dataset. 

        Parameters:
        device (str): The device (e.g., 'cpu' or 'cuda') on which the tensors should be allocated.
        param_dict (dict): A dictionary containing configuration parameters for the network. 
                           Expected keys include:
                           - 'dims_enc_outer': Dimensions of the last layer of the preceding outer encoder network.
                           - 'dims_lense': A list of integers defining the number of neurons in the hidden layers.
                           - 'layer_order': A list specifying the types of layers and their configurations.
                           - 'latent_dim': The dimensionality of the latent space.
        """
        structure = [param_dict['dims_enc_outer'][-1]] + param_dict['dims_lense']

        self.model = create_structure(structure=structure, 
                                      layer_order=param_dict['layer_order'],
                                      )

        self.mu = nn.Linear(structure[-1], param_dict['latent_dim'])
        self.log_sig = nn.Linear(structure[-1], param_dict['latent_dim'])

        self.sampling_dist = Normal(
            torch.zeros(torch.Size([param_dict['latent_dim']]), device=torch.device(device)), 
            torch.ones(torch.Size([param_dict['latent_dim']]), device=torch.device(device)))

    def encode(self, inter):
        """
        Encodes the intermediate representation into the variational parameters.

        This method processes the input through the model, and then computes the mean (mu) and log standard deviation (log_sig)
        for the latent space representation.

        Parameters:
        inter (Tensor): Intermediate representation, output of the context or target Encoder_outer class.

        Returns:
        tuple: A tuple containing the mean and log standard deviation of the latent space representation.
        """

        x = self.model(inter)
        mu = self.mu(x)
        log_sig = self.log_sig(x)
        return mu, log_sig

    def forward(self, inter):
        """
        The forward pass encodes the input data and label input into the (mu, log_sig) tuple using the 'encode' method, 
        then applies the reparametrization trick to obtain latent variables, 
        and finally computes the KL divergence for the ELBO loss function.

        Parameters:
        inter (Tensor): Intermediate representation.

        Returns:
        tuple: A tuple containing the latent variables 'z' and the KL divergence 'kl_div'.
        """

        mu, log_sig = self.encode(inter)
        eps = self.sampling_dist.sample(torch.Size([log_sig.size(dim=0)])) 
        kl_div = torch.mean(0.5 * torch.sum(mu.square() + torch.exp(2.0 * log_sig) - 1.0 - 2.0 * log_sig, dim=1))

        z = mu + log_sig.exp() * eps
        return z, kl_div

class Library_encoder(nn.Module):
    def __init__(self, 
                 device: str,
                 param_dict: dict
                 ):
        """
        This class defines the encoder neural network model for encoding gene expression data into the library size latent variables.

        Parameters:
        device (str): The device (e.g., 'cpu' or 'cuda') on which the tensors should be allocated.
        param_dict (dict): A dictionary containing configuration parameters for the network. Expected keys include:
                           - 'data_dim': The dimensionality of the gene expression input data.
                           - 'batch_dim': The dimensionality of the experimental batch information.
                           - 'dims_l_enc': A list of integers defining the number of neurons in the hidden layers.
                           - 'layer_order': A list specifying the types of layers and their configurations.
                           - 'lib_mu_add': A scalar value to be added to the mean output for stable convergence.
                           Set it a mean of all log library sizes in the dataset.
        """
        super(Library_encoder, self).__init__()
        self.device = device 

        structure = [param_dict['data_dim']+param_dict['batch_dim']] + param_dict['dims_l_enc']

        self.model = create_structure(structure=structure, 
                                      layer_order=param_dict['layer_order'],
                                      )        

        self.mu_add = param_dict['lib_mu_add']

        self.mu = nn.Linear(structure[-1], 1)
        self.log_sig = nn.Linear(structure[-1], 1)

        self.sampling_dist = Normal(
            torch.zeros(torch.Size([1]), device=torch.device(device)), 
            torch.ones(torch.Size([1]), device=torch.device(device))) 

    def encode(self, data, label_inp):
        """
        Encodes the input data and label input into library size variational parameters.

        This method processes the input through the model, and then computes the mean (mu) and log standard deviation (log_sig)
        for the latent library size variable, and shifts the mean by a predefined scalar ('mu_add').

        Parameters:
        data (Tensor): The input gene expression count tensor.
        label_inp (Tensor): The one-hot encoded experimental bacht label tensor.

        Returns:
        tuple: A tuple containing the adjusted mean and log variance of the library size representation.
        """

        x = torch.cat((data, label_inp), dim=-1)
        x = self.model(x)
        mu = self.mu(x)
        log_sig = self.log_sig(x)
        return mu + self.mu_add, log_sig 

    def forward(self, data, label_inp, prior_mu, prior_sig):
        """
        Defines the forward pass of the Library_encoder network.

        The forward pass encodes the input data and label input into library size (mu, log_sig) tuple using the 'encode' method, 
        then applies the reparametrization trick to obtain latent variables, 
        and finally computes the KL divergence between the prior and the encoded distribution for the ELBO loss function.

        Parameters:
        data (Tensor): The input gene expression count tensor.
        label_inp (Tensor): The one-hot encoded experimental bacht label tensor.
        prior_mu (Tensor): The precalculated mean of the prior distribution.
        prior_sig (Tensor): The precalculated standard deviation of the prior distribution.

        Returns:
        tuple: A tuple containing the encoded library size 'l' and the KL divergence 'kl_div'.
        """

        mu, log_sig = self.encode(data, label_inp)
        eps = self.sampling_dist.sample(torch.Size([log_sig.size(dim=0)])) 
        kl_div = torch.mean(prior_sig.log() - log_sig.squeeze() + (1 / torch.clamp((2.0 * prior_sig.square()), min=1e-7)) * ((mu.squeeze() - prior_mu) ** 2 + torch.exp(2.0 * log_sig.squeeze()) - prior_sig.square()))

        l = torch.exp(mu + log_sig.exp() * eps)
        return l, kl_div

class Decoder(nn.Module):
    def __init__(self, 
                 param_dict: dict
                 ):
        """
        This class defines the decoder neural network model. 
        It supports two types of distributions for the decoded output: Negative Binomial (NB)
        and Zero-Inflated Negative Binomial (ZINB).

        Parameters:
        param_dict (dict): A dictionary containing configuration parameters for the decoder network. Expected keys include:
                           - 'latent_dim': The dimensionality of the latent space.
                           - 'data_dim': The dimensionality of the gene expression input data.
                           - 'batch_dim': The dimensionality of the experimental batch information.
                           - 'dims_dec': A list of integers defining the number of neurons in each layer of the decoder.
                           - 'layer_order': Layer configurations for the decoder.
                           - 'data_distr': The type of distribution of the data ('zinb' or 'nb').
                           - 'dispersion':  How to compute the dispersion parameter. ('dataset', 'batch', or 'cell').
                                            'dataset' keeps the dispersion of a gene constant for the whole dataset.
                                            'batch' keeps the dispersion of a gene constat for every cell of a batch
                                            'cell' allows for individual dispersion for every cell.
                           - 'homologous_genes': An array of indices for homologous genes in the dataset.
        """

        super(Decoder, self).__init__()

        structure = [param_dict['latent_dim']+param_dict['batch_dim']] + param_dict['dims_dec']

        self.data_distr = param_dict['data_distr']
        self.dispersion = param_dict['dispersion']
        self.homologous_genes = np.array(param_dict['homologous_genes'])
        self.non_hom_genes = np.setdiff1d(np.arange(param_dict['data_dim']), self.homologous_genes)
        self.gene_ind = np.argsort(np.concatenate((self.homologous_genes, self.non_hom_genes)))
        self.data_dim = param_dict['data_dim']

        if self.data_distr not in ['zinb', 'nb']:
            raise ValueError(f"data_distr must be a list containing these strings: {'zinb', 'nb'}")        

        if self.dispersion not in ['dataset', 'batch', 'cell']:
            raise ValueError(f"dispersion must be a list containing these strings: {'dataset', 'batch', 'cell'}")     

        self.model = create_structure(structure=structure, 
                                      layer_order=param_dict['layer_order'],
                                      )   
          
        self.rho_pre = nn.Linear(structure[-1], self.data_dim)
        
        if self.dispersion == "dataset":
            self.log_alpha = torch.nn.parameter.Parameter(data=torch.randn(self.data_dim)*0.1, requires_grad=True)
        elif self.dispersion == "batch":
            self.log_alpha = torch.nn.parameter.Parameter(data=torch.randn((param_dict['batch_dim'], self.data_dim))*0.1, requires_grad=True)    
        elif self.dispersion == "cell":
            self.log_alpha = nn.Linear(structure[-1], self.data_dim)

        if self.data_distr == 'zinb':
            self.pi_nlogit = nn.Linear(structure[-1], self.data_dim)    

    def calc_nlog_likelihood(self, dec_outp, library, x, eps=1e-7): 
        """
        This method computes the negative log likelihood of the observed data given the decoder output for either the NB or ZINB distribution.
        It is used as part of the ELBO loss function computation during training.
        It is also used to evaluate the learned latent space manifold during the latent space NNS.

        Parameters:
        dec_outp (tuple): The output from the decoder, containing parameters of the chosen distribution.
        library (Tensor): The library size for scaling the output.
        x (Tensor): The observed data.
        eps (float): A small epsilon value to prevent numerical instability.

        Returns:
        Tensor: The negative log likelihood of the observed data given the decoder output.
        """

        if self.data_distr == 'nb':
            alpha, rho = dec_outp 
            alpha = torch.clamp(alpha, min=eps)
            rho = torch.clamp(rho, min=1e-8, max=1-eps)
            mu = rho * library
            p = torch.clamp(mu / (mu + alpha), min=eps, max=1-eps)            
            log_likelihood = x * torch.log(p) + alpha * torch.log(1.0 - p) - torch.lgamma(alpha) - torch.lgamma(1.0 + x) + torch.lgamma(x + alpha)   

        elif self.data_distr == 'zinb':
            alpha, rho, pi_nlogit = dec_outp  
            alpha = torch.clamp(alpha, min=eps)
            rho = torch.clamp(rho, min=1e-8, max=1-eps)            
            mu = rho * library
            log_alpha_mu = torch.log(alpha + mu)

            log_likelihood = torch.where(x < eps,
                F.softplus(pi_nlogit + alpha * (torch.log(alpha) - log_alpha_mu)) - F.softplus(pi_nlogit),
                - F.softplus(pi_nlogit) + pi_nlogit 
                + alpha * (torch.log(alpha) - log_alpha_mu) + x * (torch.log(mu) - log_alpha_mu) 
                + torch.lgamma(x + alpha) - torch.lgamma(alpha) - torch.lgamma(1.0 + x))
   
        return - torch.sum(log_likelihood, dim=-1) 

    def decode(self, z, label_inp):
        """
        Decodes the latent space representation and experimental batch information through the decoder network 
        into the parameters for the chosen output distribution (NB or ZINB).

        Parameters:
        z (Tensor): The latent space representation.
        label_inp (Tensor): The label input tensor.

        Returns:
        list: A list of tensors representing the parameters of the chosen output distribution.
        alpha - dispersion parameter
        rho - normalized gene expression parameter
        pi - dropout parameter for zero inflation
        ([alpha, rho] for NB, [alpha, rho, pi] for ZINB)

        The rho parameter is calculated separately for homologous and non homologous genes.
        """

        x = torch.cat((z, label_inp), dim=-1)
        x = self.model(x)

        if self.dispersion == "dataset":
            alpha = self.log_alpha.exp()
        elif self.dispersion == "batch":
            alpha = self.log_alpha[torch.argmax(label_inp, dim=-1)].exp()
        elif self.dispersion == "cell":
            alpha = self.log_alpha(x).exp()

        rho_pre = self.rho_pre(x)
        rho_pre_hom = F.softmax(rho_pre[:, self.homologous_genes], dim=-1) * len(self.homologous_genes)/self.data_dim
        rho_pre_nonhom = F.softmax(rho_pre[:, self.non_hom_genes], dim=-1) * len(self.non_hom_genes)/self.data_dim
        rho = torch.cat((rho_pre_hom, rho_pre_nonhom), dim=-1)[:, self.gene_ind]

        outputs = [alpha, rho]

        if self.data_distr == 'zinb':
            pi_nlogit = self.pi_nlogit(x)
            outputs.append(pi_nlogit)
        return outputs  
    
    def decode_homologous(self, z, label_inp):
        """
        Decodes the latent variables and label input into gene expression for homologous genes.
        This method is specifically used to asess and compare the log2-fold change between species.

        Parameters:
        z (Tensor): The latent space representation.
        label_inp (Tensor): The label input tensor.

        Returns:
        Tensor: The decoded gene expression probabilities for homologous genes.
        """

        x = torch.cat((z, label_inp), dim=-1)
        x = self.model(x)
        rho_pre = self.rho_pre(x)
        rho_hom = F.softmax(rho_pre[:, self.homologous_genes], dim=-1)
        return rho_hom  

    def forward(self, z, label_inp, library, x):    
        """
        Defines the forward pass of the Decoder network.
        This method combines the decoding and likelihood calculation steps. 
        It is used during training to compute the ELBO loss for the model.

        Parameters:
        z (Tensor): The latent space representation.
        label_inp (Tensor): The experimental batch information input tensor.
        library (Tensor): The library size latent variable tensor.
        x (Tensor): The observed data tensor.

        Returns:
        Tensor: The mean negative log likelihood across the input batch.
        """

        outputs = self.decode(z, label_inp)
        n_log_likeli = self.calc_nlog_likelihood(outputs, library, x).mean()
        return n_log_likeli     



class make_act_bounded(nn.Module):
    def __init__(self, act, min, max):
        """
        This class is designed to apply an activation function to an input and then bound the output 
        within a specified range. It can be used to modify standard activation functions so that their 
        output values are constrained between a minimum and maximum value.
        This is important to guarantee stability for unbounded activation functions like ReLU
        when not using layer normalization.

        Parameters:
        act (nn.Module): The activation function to be applied. It should be a PyTorch module representing an activation function (e.g., nn.ReLU).
        min (float): The minimum value to which the activation output should be clamped.
        max (float): The maximum value to which the activation output should be clamped.

        The constructor stores the provided activation function and the specified minimum and maximum bounds. 
        These are used in the forward pass to process and constrain the output.
        """        
        super().__init__()

        self.act = act         
        self.min = min   
        self.max = max    

    def forward(self, x):
        x = self.act(x)
        return torch.clamp(x, min=self.min, max=self.max)

class scSpecies():
 
    def __init__(self, 
                 device: str,
                 mdata: mu.MuData, 
                 directory: str,  
                 random_seed: int = 369963, 

                 context_dataset_key: str = 'mouse', 
                 target_dataset_key: str = 'human',      

                 context_optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 target_optimizer: torch.optim.Optimizer = torch.optim.Adam,   

                 context_hidden_dims_enc_outer: List[int] = [300],
                 target_hidden_dims_enc_outer: List[int] = [300],

                 hidden_dims_lense: List[int] = [200],

                 context_hidden_dims_l_enc: List[int] = [200],
                 target_hidden_dims_l_enc: List[int] = [200],

                 context_hidden_dims_dec: List[int] = [200, 300],
                 target_hidden_dims_dec: List[int] = [200, 300],

                 context_layer_order: list = ['linear', 'layer_norm', ('act', nn.ReLU(), [0, 6]), ('dropout', 0.1)],
                 target_layer_order: list = ['linear', 'layer_norm', ('act', nn.ReLU(), [0, 6]), ('dropout', 0.1)],

                 context_b_s: int = 128,
                 target_b_s: int = 128,          

                 context_data_distr: str = 'zinb',
                 target_data_distr: str = 'zinb',

                 latent_dim: int = 10,

                 context_dispersion: str = 'batch',
                 target_dispersion: str = 'batch',

                 alignment: int = 'inter',
                 k_neigh: int = 25,
                 top_percent: float = 20,

                 context_beta_start: float = 0.1,                
                 context_beta_max: float  = 1,
                 context_beta_epochs_raise: int = 10, 

                 target_beta_start: float = 0.1,                
                 target_beta_max: float  = 1,
                 target_beta_epochs_raise: int = 10, 

                 eta_start: float = 10,
                 eta_max: float = 30,
                 eta_epochs_raise: int = 10,              
                 ):
        
        self.device = device
        self.mdata = mdata  
        self.context_dataset_key = context_dataset_key
        self.target_dataset_key = target_dataset_key        
        self.directory = directory

        self.context_beta = context_beta_start
        self.target_beta = target_beta_start
        self.eta = eta_start   
        self.alignment = alignment

        # set the random seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.rng = np.random.default_rng(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed) 
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # computes the indices of the homologous genens
        _, hom_ind_context, hom_ind_target = np.intersect1d(np.array(mdata.mod[context_dataset_key].var['human_gene_names']), np.array(mdata.mod[target_dataset_key].var['human_gene_names']), return_indices=True)    

        # hyperparameter dictionary for the context model
        self.context_param_dict = {
            'random_seed': random_seed, 
            'dims_lense': hidden_dims_lense,
            'latent_dim': latent_dim,
            'optimizer': context_optimizer,
            'homologous_genes': list(hom_ind_context),
            'data_dim': self.mdata.mod[context_dataset_key].n_vars,
            'batch_dim': np.shape(self.mdata.mod[context_dataset_key].obsm['batch_label_enc'])[1],  
            'lib_mu_add': round(np.mean(self.mdata.mod[self.context_dataset_key].obs['library_log_mean']),5),    
            'dims_enc_outer': context_hidden_dims_enc_outer,
            'dims_l_enc': context_hidden_dims_l_enc,
            'dims_dec': context_hidden_dims_dec,
            'layer_order': context_layer_order,
            'data_distr': context_data_distr,
            'dispersion': context_dispersion,
            'b_s': context_b_s,
            'beta_start': context_beta_start, 
            'beta_max': context_beta_max,   
            'beta_epochs_raise': context_beta_epochs_raise,  
        }     

        # hyperparameter dictionary for the target model
        self.target_param_dict = {
            'random_seed': random_seed, 
            'dims_lense': hidden_dims_lense,
            'latent_dim': latent_dim,
            'optimizer': target_optimizer,
            'homologous_genes': list(hom_ind_target),
            'data_dim': self.mdata.mod[target_dataset_key].n_vars,
            'batch_dim': np.shape(self.mdata.mod[target_dataset_key].obsm['batch_label_enc'])[1], 
            'lib_mu_add': round(np.mean(self.mdata.mod[self.target_dataset_key].obs['library_log_mean']),5), 
            'dims_enc_outer': target_hidden_dims_enc_outer,   
            'dims_l_enc': target_hidden_dims_l_enc,
            'dims_dec': target_hidden_dims_dec,
            'layer_order': target_layer_order,
            'data_distr': target_data_distr,
            'dispersion': target_dispersion,            
            'b_s': target_b_s,            
            'beta_start': target_beta_start,                                      
            'beta_max': target_beta_max,            
            'beta_epochs_raise': target_beta_epochs_raise, 
            'k_neigh': k_neigh,
            'top_percent': top_percent,            
            'eta_start': eta_start,     
            'eta_max': eta_max,
            'eta_epochs_raise': eta_epochs_raise, 
        }     

        if self.context_param_dict['dims_enc_outer'][-1] != self.target_param_dict['dims_enc_outer'][-1]:
            raise ValueError("Context and target dims_enc_outer have the same output dimensions.")       

        self.create_directory()
        self.initialize()    
        self.pred_labels_nns_hom_genes(self.mdata.mod[self.context_dataset_key].uns['dataset_cell_key'])

    def initialize(self, initialize='both'):

        if initialize in ['context', 'both']:
            print('\nInitializing context scVI model.')
            self.context_encoder_inner = Encoder_inner(device=self.device,  param_dict=self.context_param_dict).to(self.device)
            self.context_encoder_outer = Encoder_outer(param_dict=self.context_param_dict).to(self.device)
            self.context_lib_encoder = Library_encoder(device=self.device, param_dict=self.context_param_dict).to(self.device)       
            self.context_decoder = Decoder(param_dict=self.context_param_dict).to(self.device) 
            self.context_optimizer = self.context_param_dict['optimizer'](
                list(self.context_encoder_outer.parameters()) + list(self.context_lib_encoder.parameters()) + list(self.context_decoder.parameters()) + list(self.context_encoder_inner.parameters()))

            self.context_encoder_inner.__name__ = 'context_encoder_inner'
            self.context_encoder_outer.__name__ = 'context_encoder_outer'
            self.context_lib_encoder.__name__ = 'context_lib_encoder'
            self.context_decoder.__name__ = 'context_decoder'        
            self.context_optimizer.__name__ = 'context_optimizer'   

        if initialize in ['target', 'both']:
            print('\nInitializing target scVI model.')
            self.target_encoder_outer = Encoder_outer(param_dict=self.target_param_dict).to(self.device)
            self.target_lib_encoder = Library_encoder(device=self.device, param_dict=self.target_param_dict).to(self.device)       
            self.target_decoder = Decoder(param_dict=self.target_param_dict).to(self.device) 
            
            if self.alignment == 'latent':
                self.target_encoder_inner = Encoder_inner(device=self.device,  param_dict=self.context_param_dict).to(self.device)
                self.target_optimizer = self.target_param_dict['optimizer'](
                    list(self.target_encoder_outer.parameters()) + list(self.target_lib_encoder.parameters()) + list(self.target_decoder.parameters()) + list(self.target_encoder_inner.parameters()))
           

            elif self.alignment == 'inter':
                self.target_encoder_inner = self.context_encoder_inner
                self.target_optimizer =self.target_param_dict['optimizer'](
                    list(self.target_encoder_outer.parameters()) + list(self.target_lib_encoder.parameters()) + list(self.target_decoder.parameters()))

            self.target_encoder_inner.__name__ = 'target_encoder_inner' 
            self.target_encoder_outer.__name__ = 'target_encoder_outer'
            self.target_lib_encoder.__name__ = 'target_lib_encoder'
            self.target_decoder.__name__ = 'target_decoder'        
            self.target_optimizer.__name__ = 'target_optimizer'   


    def create_directory(self):

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            print(f"\nCeated directory '{self.directory}'.")

        subfolders = ['figures', 'params', 'dataset']
        for folder in subfolders:
            if not os.path.exists(self.directory+'/'+folder):
                subfolder_path = os.path.join(self.directory, folder)
                os.makedirs(subfolder_path)
                print(f"\nCeated directory '{self.directory+'/'+folder}'.")


    def save_params(self, save='both', name=''):

        model_list = []
        if save in ['context', 'both']:
            model_list += [self.context_encoder_inner, self.context_encoder_outer, self.context_decoder, self.context_lib_encoder, self.context_optimizer]
            with open(self.directory+'/params/context_param_dict'+name+'.pkl', 'wb') as pickle_file:
                pickle.dump(self.context_param_dict, pickle_file)

        if save in ['target', 'both']:
            model_list += [self.target_encoder_inner, self.target_encoder_outer, self.target_lib_encoder, self.target_decoder, self.target_optimizer]
            with open(self.directory+'/params/target_param_dict'+name+'.pkl', 'wb') as pickle_file:
                pickle.dump(self.target_param_dict, pickle_file) 

        for model in model_list:
            torch.save(model.state_dict(), self.directory+'/params/'+model.__name__+name+'.pth')
        print('\nSaved models to path.')

    def save_mdata(self, name: str):  

        self.mdata.write(self.directory+'/dataset/'+name+'.h5mu') 
        print('\nSaved mdata {}.'.format(self.directory))

    # Helper for load_params
    def compare_elements(self, elem1, elem2):
        if type(elem1) != type(elem2):
            return False
        if isinstance(elem1, nn.Module):
            return type(elem1) == type(elem2)
        if isinstance(elem1, tuple):
            return all(self.compare_elements(sub_elem1, sub_elem2) for sub_elem1, sub_elem2 in zip(elem1, elem2))
        return elem1 == elem2

    # Helper for load_params
    def compare_lists(self, list1, list2):
        if len(list1) != len(list2):
            return False
        return all(self.compare_elements(elem1, elem2) for elem1, elem2 in zip(list1, list2))


    def load_params(self, load='both', name=''): 

        model_list = []
        if load in ['context', 'context_encoder', 'both']:
            model_list += [self.context_encoder_inner, self.context_encoder_outer, self.context_lib_encoder]
            if load != 'context_encoder':
                model_list += [self.context_optimizer, self.context_decoder]

            with open(self.directory+'/params/context_param_dict'+name+'.pkl', 'rb') as pickle_file:
                loaded_param_dict = pickle.load(pickle_file)   

            conflicts = 0
            for key in loaded_param_dict.keys() & self.context_param_dict.keys():
                if load == 'context_encoder' and key == 'homologous_genes':
                    loaded_param_dict[key] = self.context_param_dict[key]

                if isinstance(loaded_param_dict[key], list):
                    if self.compare_lists(loaded_param_dict[key], self.context_param_dict[key]) != True:
                        conflicts += 1                    
                        print(f"\nConflicting hyperparameters {key} has a different value in the loaded dictionary: {loaded_param_dict[key]} vs {self.context_param_dict[key]}")                        

                elif loaded_param_dict[key] != self.context_param_dict[key]:
                    conflicts += 1                    
                    print(f"\nConflicting hyperparameters {key} has a different value in the loaded dictionary: {loaded_param_dict[key]} vs {self.context_param_dict[key]}")  

            self.context_param_dict = loaded_param_dict  
            self.context_beta = self.context_param_dict['beta_max']      

            if conflicts > 0:
                self.initialize('context')

        if load in ['target', 'both']:
            model_list += [self.target_encoder_inner, self.target_encoder_outer, self.target_lib_encoder, self.target_decoder, self.target_optimizer]
  
            with open(self.directory+'/params/target_param_dict'+name+'.pkl', 'rb') as pickle_file:
                loaded_param_dict = pickle.load(pickle_file)   

            conflicts = 0
            for key in loaded_param_dict.keys() & self.target_param_dict.keys():
                if isinstance(loaded_param_dict[key], list):
                    if self.compare_lists(loaded_param_dict[key], self.target_param_dict[key]) != True:
                        conflicts += 1                    
                        print(f"\nConflicting hyperparameters {key} has a different value in the loaded dictionary: {loaded_param_dict[key]} vs {self.target_param_dict[key]}")                        

                elif loaded_param_dict[key] != self.target_param_dict[key]:
                    conflicts += 1                    
                    print(f"\nConflicting hyperparameters {key} has a different value in the loaded dictionary: {loaded_param_dict[key]} vs {self.target_param_dict[key]}")
            
            self.target_param_dict = loaded_param_dict         
            self.target_beta = self.target_param_dict['beta_max']    
            self.eta = self.target_param_dict['eta_max']                

            if conflicts > 0:
                self.initialize('target')

        if load == 'context_encoder':
            self.context_optimizer = self.context_param_dict['optimizer'](list(self.context_decoder.parameters()))
            self.context_optimizer.__name__ = 'context_optimizer'   

        for model in model_list:
            model.load_state_dict(torch.load(self.directory+'/params/'+model.__name__+name+'.pth'))              
        print('\nLoaded models from path.') 

    # helper for the pred_labels_nns_aligned_latent_space method.
    @staticmethod
    def most_frequent(arr):
        values, counts = np.unique(arr, return_counts=True)
        return values[np.argmax(counts)]

    def calc_likelihood_on_aligned_latent_space(self, metric='euclidean', k_neigh=200, top_neigh=25, b_s=50, nns_key='_nns_aligned_latent_space'):
        print('\nComputing latent space NNS with {} neighbors.\n'.format(str(k_neigh)))

        neigh = NearestNeighbors(n_neighbors=k_neigh, metric=metric)
        neigh.fit(self.mdata.mod[self.context_dataset_key].obsm['latent_mu'])

        _, indices = neigh.kneighbors(self.mdata.mod[self.target_dataset_key].obsm['latent_mu'])
        self.mdata.mod[self.target_dataset_key].obsm['ind'+nns_key] = indices

        self.target_lib_encoder.eval()        
        self.target_decoder.eval()

        steps = int(np.ceil(self.mdata.mod[self.target_dataset_key].n_obs/b_s+1e-10))

        likelihoods = []

        with torch.no_grad():
            tic = time.time()
            for step in range(steps):   
                if time.time() - tic > 0.5:
                    tic = time.time()
                    print('\rCalculate likelihoods for computed neighbors. Step {}/{}     '.format(str(step), str(steps)), end='', flush=True) 

                batch_adata = self.mdata.mod[self.target_dataset_key][step*b_s:(step+1)*b_s]
                b_s = batch_adata.n_obs
                data_batch = torch.from_numpy(batch_adata.X.toarray()).to(self.device)
                label_batch = torch.from_numpy(batch_adata.obsm['batch_label_enc']).to(self.device) 
                lib_mu_batch = torch.from_numpy(np.array(batch_adata.obs['library_log_mean'])).to(self.device)          
                lib_sig_batch = torch.from_numpy(np.array(batch_adata.obs['library_log_std'])).to(self.device)  

                l, _ = self.target_lib_encoder(data_batch, label_batch, lib_mu_batch, lib_sig_batch)               

                ind_neigh = np.reshape(self.mdata.mod[self.target_dataset_key].obsm['ind'+nns_key][step*batch_adata.n_obs:(step+1)*batch_adata.n_obs], (batch_adata.n_obs*k_neigh))
                neigh_latent = torch.from_numpy(self.mdata.mod[self.context_dataset_key].obsm['latent_mu'][ind_neigh]).to(self.device)  

                label_interl = torch.repeat_interleave(label_batch, repeats=k_neigh, dim=0)
                lib_interl = torch.repeat_interleave(l, repeats=k_neigh, dim=0)
                data_interl = torch.repeat_interleave(data_batch, repeats=k_neigh, dim=0)

                outp = self.target_decoder.decode(neigh_latent, label_interl)

                nlog_likeli_neighbors = self.target_decoder.calc_nlog_likelihood(outp, lib_interl, data_interl).reshape(batch_adata.n_obs, k_neigh)

                likelihoods.append(nlog_likeli_neighbors.cpu().numpy())

        likelihoods = np.concatenate(likelihoods)
        self.mdata.mod[self.target_dataset_key].obsm['nlog_likeli'+nns_key] = likelihoods

    def pred_labels_nns_hom_genes(self, context_cell_key, calculate_top_percent=True):
        print('\nEvaluating data level NNS for context label key {}.'.format(context_cell_key))

        ind_neigh = self.mdata.mod[self.target_dataset_key].obsm['ind_nns_hom_genes']  
        ind_neigh = ind_neigh[:,:self.target_param_dict['k_neigh']]
        context_cell_labels = self.mdata.mod[self.context_dataset_key].obs[context_cell_key].to_numpy()

        cell_type_counts = [dict(Counter(context_cell_labels[ind_neigh[i]])) for i in range(np.shape(ind_neigh)[0])]
        top_dict = {}

        cell_type_counts = [max(cell_type_counts[i].items(), key=lambda x: x[1]) + (i, ) for i in range(np.shape(ind_neigh)[0])]

        top_dict = {c: [] for c in np.unique(context_cell_labels)}
        for i in range(len(cell_type_counts)):
            top_dict[cell_type_counts[i][0]] += [cell_type_counts[i]]

        for key in top_dict.keys():
            top_dict[key] = sorted(top_dict[key], key=lambda x: x[1])
            num_samples = len(top_dict[key])
            top_dict[key] = [top_dict[key][i]+(1-(i+1)/num_samples,) for i in range(len(top_dict[key]))] 

        cell_type_counts = sorted([item for sublist in top_dict.values() for item in sublist], key=lambda x: x[-2]) 

        if calculate_top_percent:
            print('\nCalculating cells with the highest agreement.')
            self.mdata.mod[self.target_dataset_key].obs['top_percent_nns_hom_genes'] = np.array([cell_type_counts[i][-1] for i in range(len(cell_type_counts))])

        self.mdata.mod[self.target_dataset_key].obs['label_nns_hom_genes'] = np.array([cell_type_counts[i][0] for i in range(len(cell_type_counts))])

    def eval_label_transfer(self, cell_keys, metric='euclidean', k_neigh=200, top_neigh=25, b_s=50, nns_key='_nns_aligned_latent_space'):
        self.calc_likelihood_on_aligned_latent_space(metric='euclidean', k_neigh=200, top_neigh=25, b_s=50, nns_key='_nns_aligned_latent_space')

        for context_cell_key, target_cell_key in list(cell_keys):
            self.mdata.mod[self.context_dataset_key].uns['dataset_cell_key'] = context_cell_key
            self.mdata.mod[self.target_dataset_key].uns['dataset_cell_key'] = target_cell_key

            self.pred_labels_nns_hom_genes(context_cell_key, calculate_top_percent=False)

            likelihoods = self.mdata.mod[self.target_dataset_key].obsm['nlog_likeli'+nns_key]
            context_cell_labels = self.mdata.mod[self.context_dataset_key].obs[context_cell_key].to_numpy()
            predicted_cell_types = np.stack([self.most_frequent(context_cell_labels[self.mdata.mod[self.target_dataset_key].obsm['ind'+nns_key][i][np.argsort(likelihoods[i])]][:top_neigh]) for i in range(self.mdata.mod[self.target_dataset_key].n_obs)])

            self.mdata.mod[self.target_dataset_key].obs['label'+nns_key] = predicted_cell_types 

            target_cell_labels = self.mdata.mod[self.target_dataset_key].obs[target_cell_key].to_numpy()
            context_cell_types = np.unique(context_cell_labels)
            target_cell_types = np.unique(target_cell_labels)

            metrics_dict = {}
            for nns_key in ['_nns_hom_genes', '_nns_aligned_latent_space']:
                predicted_cell_types = self.mdata.mod[self.target_dataset_key].obs['label'+nns_key].to_numpy()

                joint_labels, _, _ = np.intersect1d(context_cell_labels, target_cell_labels, return_indices=True)
                joint_ind = np.where(np.array([cell_label in joint_labels for cell_label in target_cell_labels]))[0]
                metrics_dict['balanced_accuracy_score'+nns_key] = balanced_accuracy_score(target_cell_labels[joint_ind], predicted_cell_types[joint_ind])
                metrics_dict['adjusted_rand_score'+nns_key] = adjusted_rand_score(target_cell_labels, predicted_cell_types)
                metrics_dict['adjusted_mutual_info_score'+nns_key] = adjusted_mutual_info_score(target_cell_labels, predicted_cell_types)
                metrics_dict['fowlkes_mallows_score'+nns_key] = fowlkes_mallows_score(target_cell_labels, predicted_cell_types)

                df = pd.DataFrame(0, index=target_cell_types, columns=context_cell_types)
                for j,cell in enumerate(target_cell_labels): 
                    df.loc[cell][predicted_cell_types[j]] +=1

                df = (df.div(df.sum(axis=1), axis=0)) * 100
                self.mdata.mod[self.target_dataset_key].uns['prediction_df'+nns_key+'_'+target_cell_key] = df

            self.mdata.mod[self.target_dataset_key].uns['metrics_'+target_cell_key] = metrics_dict

            knn_acc = self.mdata.mod[self.target_dataset_key].uns['metrics_'+target_cell_key]['balanced_accuracy_score_nns_hom_genes']*100
            latent_acc = self.mdata.mod[self.target_dataset_key].uns['metrics_'+target_cell_key]['balanced_accuracy_score_nns_aligned_latent_space']*100

            knn_adj = self.mdata.mod[self.target_dataset_key].uns['metrics_'+target_cell_key]['adjusted_rand_score_nns_hom_genes']
            latent_adj = self.mdata.mod[self.target_dataset_key].uns['metrics_'+target_cell_key]['adjusted_rand_score_nns_aligned_latent_space']

            knn_mis = self.mdata.mod[self.target_dataset_key].uns['metrics_'+target_cell_key]['adjusted_mutual_info_score_nns_hom_genes']
            latent_mis = self.mdata.mod[self.target_dataset_key].uns['metrics_'+target_cell_key]['adjusted_mutual_info_score_nns_aligned_latent_space']

            print('\n Cell labely key: {}. KNN search on hom. genes --> Acc: {}%, ARI: {}, AMI: {}'.format(target_cell_key, round(knn_acc,2), round(knn_adj,3), round(knn_mis,3)))
            print('\n Cell labely key: {}. KNN search in lat. space --> Acc: {}%, ARI: {}, AMI: {}'.format(target_cell_key, round(latent_acc,2), round(latent_adj,3), round(latent_mis,3)))

    @staticmethod
    def average_slices(array, slice_sizes):
        averages = []
        start = 0
        for size in slice_sizes:
            end = start + size
            slice_avg = np.mean(array[start:end], axis=0)
            averages.append(slice_avg)
            start = end
        return np.stack(averages)

    @staticmethod
    def filter_outliers(data, confidence_level=0.9):
        mean = np.mean(data, axis=0)
        data_centered = data - mean
        cov_matrix = np.dot(data_centered.T, data_centered) / (data_centered.shape[0] - 1)
        cov_inv = np.linalg.inv(cov_matrix)

        m_dist = np.sqrt(np.sum(np.dot(data_centered, cov_inv) * data_centered, axis=1))

        df = mean.shape[0]  
        threshold = np.sqrt(chi2.ppf(confidence_level, df))

        filtered_data_ind = m_dist < threshold
        outlier_ind = m_dist >= threshold
        return filtered_data_ind, outlier_ind

    def compute_logfold_change(self, context_cell_key, target_cell_key, eps=1e-6, lfc_delta=1, samples=10000, b_s=128, confidence_level=0.9):
        self.mdata.mod[self.context_dataset_key].uns['lfc_delta'] = lfc_delta
        self.context_decoder.eval()   
        self.target_decoder.eval()     

        context_ind = np.array(self.context_param_dict['homologous_genes'])
        context_gene_names = self.mdata.mod[self.context_dataset_key].var_names.to_numpy()[context_ind]

        context_cell_labels = self.mdata.mod[self.context_dataset_key].obs[context_cell_key].to_numpy()
        context_cell_types = np.unique(context_cell_labels)
        context_cell_index = {c : np.where(context_cell_labels == c)[0] for c in context_cell_types}

        context_batch_key = self.mdata.mod[self.context_dataset_key].uns['dataset_batch_key']
        target_batch_key = self.mdata.mod[self.target_dataset_key].uns['dataset_batch_key']

        context_batch_labels = self.mdata.mod[self.context_dataset_key].obs[context_batch_key].to_numpy().reshape(-1, 1)
        target_batch_labels = self.mdata.mod[self.target_dataset_key].obs[target_batch_key].to_numpy().reshape(-1, 1)

        context_enc = OneHotEncoder()
        context_enc.fit(context_batch_labels)

        target_enc = OneHotEncoder()
        target_enc.fit(target_batch_labels)

        context_batches = {c : self.mdata.mod[self.context_dataset_key][self.mdata.mod[self.context_dataset_key].obs[context_cell_key] == c].obs[context_batch_key].value_counts() > 3 for c in context_cell_types}
        context_batches = {c : context_batches[c][context_batches[c]].index.to_numpy() for c in context_cell_types}
        context_batches = {c : context_enc.transform(context_batches[c].reshape(-1, 1)).toarray().astype(np.float32)  for c in context_cell_types}
        context_batches['unknown'] = context_enc.transform(np.unique(context_batch_labels).reshape(-1, 1)).toarray().astype(np.float32)

        if target_cell_key == None:
            joint_cell_types = context_cell_types

        else:
            target_cell_labels = self.mdata.mod[self.target_dataset_key].obs[target_cell_key].to_numpy()
            target_cell_types = np.unique(target_cell_labels)
            joint_cell_types = np.intersect1d(context_cell_types, target_cell_types, return_indices=True)[0]
            target_batches = {c : self.mdata.mod[self.target_dataset_key][self.mdata.mod[self.target_dataset_key].obs[target_cell_key] == c].obs[target_batch_key].value_counts() > 3 for c in target_cell_types}
            target_batches = {c : target_batches[c][target_batches[c]].index.to_numpy() for c in target_cell_types}
            target_batches = {c : target_enc.transform(target_batches[c].reshape(-1, 1)).toarray().astype(np.float32)  for c in target_cell_types}
            target_batches['unknown'] = target_enc.transform(np.unique(target_batch_labels).reshape(-1, 1)).toarray().astype(np.float32)


        df_lfc = pd.DataFrame(0, index=context_gene_names, columns=joint_cell_types)
        df_prob = pd.DataFrame(0, index=context_gene_names, columns=joint_cell_types)


        for cell_type in joint_cell_types:
            adata = self.mdata.mod[self.context_dataset_key][context_cell_index[cell_type]]

            filtered_data_ind, _ = self.filter_outliers(adata.obsm['latent_mu'], confidence_level=confidence_level)
            adata = adata[filtered_data_ind]

            steps = np.ceil(adata.n_obs/b_s).astype(int)    
            sampling_size = max(int(samples / adata.n_obs), 1)

            with torch.no_grad():
                logfold_list = []                
                tic = time.time()
                for step in range(steps):   
                    if time.time() - tic > 0.5:
                        tic = time.time()
                        print('\rCalculating LFC for cell type {}. Step {}/{}'.format(cell_type, str(step), str(steps))+ ' '*25, end='', flush=True) 
  
                    batch_adata = adata[step*b_s:(step+1)*b_s]

                    context_cell_type = batch_adata.obs[batch_adata.uns['dataset_cell_key']].to_numpy()
                    target_cell_type = np.array(['unknown']*batch_adata.n_obs)

                    context_labels = np.concatenate([context_batches[c] for c in context_cell_type])
                    target_labels = np.concatenate([target_batches[c] for c in target_cell_type])
                    context_labels = torch.from_numpy(context_labels).to(self.device)
                    target_labels = torch.from_numpy(target_labels).to(self.device)            

                    context_ind_batch = np.array([np.shape(context_batches[c])[0] for c in context_cell_type])
                    target_ind_batch = np.array([np.shape(target_batches[c])[0] for c in target_cell_type])
   

                    shape = np.shape(adata.obsm['latent_sig'])
                    for k in range(sampling_size):
                        z = np.float32(adata.obsm['latent_mu'] + adata.obsm['latent_sig'] * np.random.rand(shape[0], shape[1]))         
                        context_z = np.concatenate([np.tile(z[j], (i, 1)) for j, i in enumerate(context_ind_batch)])
                        target_z = np.concatenate([np.tile(z[j], (i, 1)) for j, i in enumerate(target_ind_batch)])

                        context_z = torch.from_numpy(context_z).to(self.device)
                        target_z = torch.from_numpy(target_z).to(self.device)

                        context_rho = self.context_decoder.decode_homologous(context_z, context_labels).cpu().numpy()
                        context_rho = self.average_slices(context_rho, context_ind_batch)

                        target_rho = self.target_decoder.decode_homologous(target_z, target_labels).cpu().numpy()
                        target_rho = self.average_slices(target_rho, target_ind_batch)

                        logfold_list.append(np.log2(context_rho+eps) - np.log2(target_rho+eps))

            logfold_list = np.concatenate(logfold_list)
            
            median_logfold = np.median(logfold_list, axis=0)
            lfc_prob = np.sum(np.where(np.abs(logfold_list)>lfc_delta, 1, 0), axis=0) / np.shape(logfold_list)[0]

            df_lfc[cell_type] = median_logfold
            df_prob[cell_type] = lfc_prob

        self.mdata.mod[self.context_dataset_key].uns['lfc_df'] = df_lfc        
        self.mdata.mod[self.context_dataset_key].uns['prob_df'] = df_prob        

    def eval_context(self):

        self.context_encoder_outer.eval()  
        self.context_encoder_inner.eval()

        b_s = self.context_param_dict['b_s']
        steps = int(np.ceil(self.mdata.mod[self.context_dataset_key].n_obs/b_s+1e-10))        
        mu_list, inter_list, sig_list = [], [], []


        with torch.no_grad():
            tic = time.time()
            for step in range(steps):   
                if time.time() - tic > 0.5:
                    tic = time.time()
                    print('\rCalculate context intermediate and latent variables. Step {}/{}     '.format(str(step), str(steps)), end='', flush=True) 

                batch_adata = self.mdata.mod[self.context_dataset_key][step*b_s:(step+1)*b_s]
                data_batch = torch.from_numpy(batch_adata.X.toarray()).to(self.device)
                label_batch = torch.from_numpy(batch_adata.obsm['batch_label_enc']).to(self.device) 

                inter = self.context_encoder_outer(data_batch, label_batch) 
                z_loc, z_log_sig = self.context_encoder_inner.encode(inter)             

                mu_list.append(z_loc.cpu().numpy())
                inter_list.append(inter.cpu().numpy())
                sig_list.append(z_log_sig.exp().cpu().numpy())
                
            self.mdata.mod[self.context_dataset_key].obsm['latent_mu'] = np.concatenate(mu_list)
            self.mdata.mod[self.context_dataset_key].obsm['latent_sig'] = np.concatenate(sig_list)
            self.mdata.mod[self.context_dataset_key].obsm['inter'] = np.concatenate(inter_list)
    

    def eval_target(self, save_intermediate=False):     

        if self.alignment == 'inter':
            self.target_encoder_inner = self.context_encoder_inner

        self.target_encoder_outer.eval()   
        self.target_encoder_inner.eval()

        b_s = self.target_param_dict['b_s']
        steps = int(np.ceil(self.mdata[self.target_dataset_key].n_obs/b_s+1e-10))
        mu_list, inter_list, sig_list = [], [], []

        with torch.no_grad():
            tic = time.time()
            for step in range(steps):   
                if time.time() - tic > 0.5:
                    tic = time.time()
                    if save_intermediate:
                        print('\rCalculate target intermediate and latent variables. Step {}/{}     '.format(str(step), str(steps)), end='', flush=True) 
                    else:
                        print('\rCalculate target latent variables. Step {}/{}     '.format(str(step), str(steps)), end='', flush=True) 

                batch_adata = self.mdata.mod[self.target_dataset_key][step*b_s:(step+1)*b_s]
                data_batch = torch.from_numpy(batch_adata.X.toarray()).to(self.device)
                label_batch = torch.from_numpy(batch_adata.obsm['batch_label_enc']).to(self.device) 

                inter = self.target_encoder_outer(data_batch, label_batch)           
                z_loc, z_log_sig = self.target_encoder_inner.encode(inter)  

                mu_list.append(z_loc.cpu().numpy())
                inter_list.append(inter.cpu().numpy())
                sig_list.append(z_log_sig.exp().cpu().numpy())
                
            self.mdata.mod[self.target_dataset_key].obsm['latent_mu'] = np.concatenate(mu_list)
            self.mdata.mod[self.target_dataset_key].obsm['latent_sig'] = np.concatenate(sig_list)

            if save_intermediate:
                self.mdata.mod[self.target_dataset_key].obsm['inter'] = np.concatenate(inter_list)

    @staticmethod
    def update_param(parameter, min_value, max_value, steps):
    
        if steps == 0 or min_value == max_value:
            return parameter

        parameter += (max_value - min_value) / steps
        return min(parameter, max_value)

    def initialize_prototypes(self, context_cell_key, target_cell_key):
        context_cell_labels = self.mdata.mod[self.context_dataset_key].obs[context_cell_key].to_numpy()
        context_cell_index = {c : np.where(context_cell_labels == c)[0] for c in np.unique(context_cell_labels)}

        target_cell_labels = self.mdata.mod[self.target_dataset_key].obs[target_cell_key].to_numpy()
        target_cell_index = {c : np.where(target_cell_labels == c)[0] for c in np.unique(target_cell_labels)}
        
        self.context_mean = []
        self.context_label = []
        for c in np.unique(context_cell_labels):
            self.context_mean.append(np.median(self.mdata.mod[self.context_dataset_key][context_cell_index[c]].X.toarray(), axis=0))
            self.context_label.append(np.mean(self.mdata.mod[self.context_dataset_key][context_cell_index[c]].obsm['batch_label_enc'], axis=0))       
        self.context_mean = torch.from_numpy(np.stack(self.context_mean)).to(self.device)
        self.context_label = torch.from_numpy(np.stack(self.context_label)).to(self.device)

        self.context_encoder_outer.eval()
        self.context_encoder_inner.eval()        
        with torch.no_grad():        
            self.mu_context = self.context_encoder_inner.encode(self.context_encoder_outer(self.context_mean, self.context_label))[0]
        self.context_encoder_outer.train()

        self.target_mean = []
        self.target_label = []
        for c in np.unique(target_cell_labels):
            self.target_mean.append(np.median(self.mdata.mod[self.target_dataset_key][target_cell_index[c]].X.toarray(), axis=0))
            self.target_label.append(np.mean(self.mdata.mod[self.target_dataset_key][target_cell_index[c]].obsm['batch_label_enc'], axis=0))    
        self.target_mean = torch.from_numpy(np.stack(self.target_mean)).to(self.device)
        self.target_label = torch.from_numpy(np.stack(self.target_label)).to(self.device)
        self.initialized = False

    def eval_prototypes(self):
        if self.initialized == False:
            self.nlog_likeli_neighbors = []
            self.param = torch.ones((self.target_mean.size(0), 1)).to(self.device)
            self.label_interl = torch.repeat_interleave(self.target_label, repeats=self.mu_context.size(0), dim=0)
            self.data_interl = torch.repeat_interleave(self.target_mean, repeats=self.mu_context.size(0), dim=0)
            self.context_rep = self.mu_context.repeat(self.target_mean.size(0), 1)
            self.initialized = True

        self.target_lib_encoder.eval()
        self.target_decoder.eval()
        with torch.no_grad():    
            l = self.target_lib_encoder(self.target_mean, self.target_label, self.param, self.param)[0]
            lib_interl = torch.repeat_interleave(l, repeats=self.mu_context.size(0), dim=0)

            outp = self.target_decoder.decode(self.context_rep, self.label_interl)
            likeli = self.target_decoder.calc_nlog_likelihood(outp, lib_interl, self.data_interl).reshape(self.target_mean.size(0), self.context_mean.size(0))
            self.nlog_likeli_neighbors.append(likeli.cpu().numpy())

        self.target_lib_encoder.train()
        self.target_decoder.train()

    def train_context(self, epochs=40, raise_beta=True, save_model=True, train_decoder_only=False, save_key=''):
        b_s = self.context_param_dict['b_s']
        n_obs = self.mdata.mod[self.context_dataset_key].n_obs

        steps_per_epoch = int(n_obs/b_s)

        progBar = Progress_Bar(epochs, steps_per_epoch, ['nELBO', 'nlog_likeli', 'KL-Div z', 'KL-Div l'])

        print(f'\nPretraining on the context dataset for {epochs} epochs (= {epochs*steps_per_epoch} iterations).')

        self.context_encoder_outer.train()
        self.context_lib_encoder.train()        
        self.context_encoder_inner.train()
        self.context_decoder.train()

        if train_decoder_only:
            self.context_encoder_outer.eval()
            self.context_lib_encoder.eval()        
            self.context_encoder_inner.eval()            

        for epoch in range(epochs):
            perm = self.rng.permutation(n_obs)  

            if raise_beta: 
                self.context_beta = self.update_param(self.context_beta, self.context_param_dict['beta_start'], self.context_param_dict['beta_max'], self.context_param_dict['beta_epochs_raise'])    

            for step in range(steps_per_epoch):         
                self.context_optimizer.zero_grad(set_to_none=True)

                batch_adata = self.mdata.mod[self.context_dataset_key][perm[step*b_s:(step+1)*b_s]]
                data_batch = torch.from_numpy(batch_adata.X.toarray()).to(self.device)
                label_batch = torch.from_numpy(batch_adata.obsm['batch_label_enc']).to(self.device)         
                lib_mu_batch = torch.from_numpy(batch_adata.obs['library_log_mean'].to_numpy()).to(self.device)          
                lib_sig_batch = torch.from_numpy(batch_adata.obs['library_log_std'].to_numpy()).to(self.device)  

                z, z_kl_div = self.context_encoder_inner(self.context_encoder_outer(data_batch, label_batch)) 
                l, l_kl_div = self.context_lib_encoder(data_batch, label_batch, lib_mu_batch, lib_sig_batch)         
                
                nlog_likeli = self.context_decoder(z, label_batch, l, data_batch)

                nelbo = self.context_beta * (z_kl_div + l_kl_div) + nlog_likeli

                nelbo.backward()
                self.context_optimizer.step() 

                progBar.update({'nELBO': nelbo.item(), 'nlog_likeli': nlog_likeli.item(), 'KL-Div z': (self.context_beta * z_kl_div).item(), 'KL-Div l': (self.context_beta * l_kl_div).item()})

        if save_model == True:    
            self.save_params('context',name=save_key)  

    def train_target(self, epochs=40, save_model=True, raise_beta=True, raise_eta=True, nns_key='_nns_hom_genes', save_key='', track_prototypes=False):

        if track_prototypes:
            self.initialize_prototypes(
                context_cell_key=self.mdata.mod[self.context_dataset_key].uns['dataset_cell_key'], 
                target_cell_key=self.mdata.mod[self.target_dataset_key].uns['dataset_cell_key'])    

        b_s = self.target_param_dict['b_s']
        n_obs = self.mdata.mod[self.target_dataset_key].n_obs
        k_neigh = self.target_param_dict['k_neigh']
        top_percent = self.target_param_dict['top_percent']


        steps_per_epoch = int(n_obs/b_s)

        progBar = Progress_Bar(epochs, steps_per_epoch, ['nELBO', 'nlog_likeli', 'KL-Div z', 'KL-Div l', 'Dist to neighbor']) 
        print(f'\nTraining on the target dataset for {epochs} epochs (= {epochs*steps_per_epoch} iterations).')

        self.target_encoder_outer.train()
        self.target_lib_encoder.train()        
        self.target_decoder.train()
        self.target_encoder_inner.train()
        
        if self.alignment == 'inter':
            self.target_encoder_inner = self.context_encoder_inner
            self.target_encoder_inner.eval()


        for epoch in range(epochs):
            perm = self.rng.permutation(n_obs)     
            if raise_beta:       
                self.target_beta = self.update_param(self.target_beta, self.target_param_dict['beta_start'], self.target_param_dict['beta_max'], self.target_param_dict['beta_epochs_raise'])    
            if raise_eta:
                self.eta = self.update_param(self.eta, self.target_param_dict['eta_start'], self.target_param_dict['eta_max'], self.target_param_dict['eta_epochs_raise'])    

            for step in range(steps_per_epoch): 
                if track_prototypes:
                    self.eval_prototypes()
                self.target_optimizer.zero_grad(set_to_none=True)

                batch_adata = self.mdata.mod[self.target_dataset_key][perm[step*b_s:(step+1)*b_s]]

                data_batch = torch.from_numpy(batch_adata.X.toarray()).to(self.device)
                label_batch = torch.from_numpy(batch_adata.obsm['batch_label_enc']).to(self.device)         
                lib_mu_batch = torch.from_numpy(batch_adata.obs['library_log_mean'].to_numpy()).to(self.device)          
                lib_sig_batch = torch.from_numpy(batch_adata.obs['library_log_std'].to_numpy()).to(self.device)  

                inter = self.target_encoder_outer(data_batch, label_batch)

                z, z_kl_div = self.target_encoder_inner(inter)                 
                l, l_kl_div = self.target_lib_encoder(data_batch, label_batch, lib_mu_batch, lib_sig_batch)   
                        
                nlog_likeli = self.target_decoder(z, label_batch, l, data_batch)                    
                ind_top = np.where(batch_adata.obs['top_percent'+nns_key].to_numpy()<top_percent/100)[0]  
                if np.shape(ind_top)[0] < 1: ind_top = np.reshape(np.random.randint(b_s), (1,))

                ind_neigh = batch_adata.obsm['ind'+nns_key][ind_top, :k_neigh]
                neigh_mu = torch.from_numpy(self.mdata.mod[self.context_dataset_key].obsm['latent_mu'][ind_neigh]).to(self.device)  
                neigh_sig = torch.from_numpy(self.mdata.mod[self.context_dataset_key].obsm['latent_sig'][ind_neigh]).to(self.device)    
                neigh_z = neigh_mu + neigh_sig * self.context_encoder_inner.sampling_dist.sample(torch.Size([neigh_sig.size(dim=0), neigh_sig.size(dim=1)]))
               
                  
                label_interl = torch.repeat_interleave(label_batch[ind_top], repeats=k_neigh, dim=0)
                lib_interl = torch.repeat_interleave(l[ind_top], repeats=k_neigh, dim=0)
                data_interl = torch.repeat_interleave(data_batch[ind_top], repeats=k_neigh, dim=0)

                outp = self.target_decoder.decode(neigh_z.view(-1, neigh_z.size(-1)), label_interl)

                nlog_likeli_neighbors = self.target_decoder.calc_nlog_likelihood(outp, lib_interl, data_interl).reshape(np.shape(ind_top)[0], k_neigh)
                best_pin_for_x = torch.argmin(nlog_likeli_neighbors, dim=1).cpu().numpy()

                if self.alignment == 'inter':
                    align_target = torch.from_numpy(self.mdata.mod[self.context_dataset_key].obsm['inter'][batch_adata.obsm['ind'+nns_key][ind_top, best_pin_for_x]]).to(self.device)
                    sqerror_align = torch.sum((inter[ind_top] - align_target)**2, dim=-1).mean()

                elif self.alignment == 'latent':
                    sqerror_align = torch.sum((z[ind_top] - neigh_z[np.arange(len(ind_top)),best_pin_for_x])**2, dim=-1).mean()

                nelbo = self.target_beta * (z_kl_div + l_kl_div) + nlog_likeli + self.eta * sqerror_align

                nelbo.backward()
                self.target_optimizer.step() 

                progBar.update({'nELBO': nelbo.item(), 'nlog_likeli': nlog_likeli.item(), 'KL-Div z': (self.target_beta * z_kl_div).item(), 'KL-Div l': (self.target_beta * l_kl_div).item(), 'Dist to neighbor': (self.eta * sqerror_align).item()})
       
        if save_model == True:            
            self.save_params('target',name=save_key)        


class Progress_Bar():
    def __init__(self, epochs, steps_per_epoch, metrics, avg_over_n_steps=100, sleep_print=0.5):

        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch 
        self.total_steps = self.epochs * steps_per_epoch
        self.remaining_steps = self.epochs * steps_per_epoch
        self.avg_over_n_steps = avg_over_n_steps
        self.tic = time.time() 
        self.sleep_print = sleep_print
        self.iteration = 0
        self.metrics = metrics
        self.time_metrics = ['Progress', 'ETA', 'Epoch', 'Iteration', 'ms/Iteration']
        
        self.dict = {
                    'Progress' : "0.000%",
                    'ETA' : 0.0,
                    'Epoch' : int(1),
                    'Iteration' : int(0),
                    'ms/Iteration' : 0.0,
                    'time': [time.time()]
                    }

        self.dict.update({metric: [] for metric in metrics})
        self.dict.update({metric+' last ep': [] for metric in metrics})
        self.dict.update({metric+' impr': 0.0 for metric in metrics})
        
    @staticmethod
    def format_number(number, min_length):
        decimal_count = len(str(number).split('.')[0])  
        decimal_places = max(min_length - decimal_count, 0) 

        formatted_number = "{:.{}f}".format(number, decimal_places)
        return formatted_number
    
    def ret_sign(self, number, min_length):
        if number > 0.0:
            sign_str = '\033[92m{}\033[00m'.format("+" + self.format_number(np.abs(number), min_length))
        elif number < 0.0:
            sign_str = '\033[91m{}\033[00m'.format("-" + self.format_number(np.abs(number), min_length))
        else:
            sign_str = '---'
        return  sign_str      

    def update(self, values):   
        self.remaining_steps -= 1   
        for key, value in values.items():
            self.dict[key].append(value) 
            
        if self.dict['Iteration'] == 1:
            for key, value in values.items():
                self.dict[key+' last ep'].append(value) 
             
        self.dict['Iteration'] += 1
        
        epoch = int(np.ceil(self.dict['Iteration'] / self.steps_per_epoch))

        if self.dict['Epoch'] < epoch:
            for key in self.metrics:
                self.dict[key+' last ep'].append(np.mean(self.dict[key][-self.steps_per_epoch:]))
                self.dict[key+' impr'] = self.dict[key+' last ep'][-2] - self.dict[key+' last ep'][-1]
            self.dict['Epoch'] = epoch
     
        self.dict['time'].append(time.time())
  
        avg_steps = np.min((self.dict['Iteration'], self.avg_over_n_steps))
        avg_time = (self.dict['time'][-1] - self.dict['time'][-avg_steps-1]) / avg_steps 
        
        self.dict['ETA'] = timedelta(seconds=int(self.remaining_steps * avg_time))         
        self.dict['ms/Iteration'] = self.format_number(avg_time*1000.0, 4)
        self.dict['Progress'] = self.format_number(100.0 * self.dict['Iteration'] / self.total_steps, 3)+'%'

        if time.time() - self.tic > self.sleep_print:
            metric_string =  [f'\033[95m{key}\033[00m: {self.dict[key]}' for key in self.time_metrics]       
            metric_string += [f'\033[33m{key}\033[00m: {self.format_number(np.mean(self.dict[key][-avg_steps:]), 5)} ({self.ret_sign(self.dict[key+" impr"], 4)})' for key in self.metrics]               
            metric_string =  "\033[96m - \033[00m".join(metric_string)
            print(f"\r{metric_string}.           ", end='', flush=True)   
            self.tic = time.time()   