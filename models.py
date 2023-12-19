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

class scPecies():
    """
    The scPecies class implements a model for aligning the latent representations of context and target datasets, 
    primarily used for cross-species or cross-condition analysis. This class facilitates 
    the integration of single-cell data from diverse sources, offering tools for training, evaluating, and 
    predicting based on single-cell variational inference models.

    Parameters:
    device (str): The device (e.g., 'cpu' or 'cuda') on which the tensors should be allocated.
    mdata (mu.MuData): The mdata object containing context and target datasets. Has to be created by the provided create_mdata class.
    random_seed (int): Random seed for reproducibility.
    hidden_dims_lense (List of ints > 0): List of integers specifying the hidden neuron sizes for the inner encoder model.
    latent_dim (int > 0): Dimensionality of the latent space.
    k_neigh (int > 0): Number of nearest neighbors for the alignment process.
    top_percent (0 <= float <= 100): The top percentage of cells with high NNS agreement to consider during alignment.
    eta_start (float >= 0): Weighting of the alignment term at the beginning of the training.
    eta_max (float >= 0 and >= eta_start): Weighting of the alignment term after eta_epochs_raise epochs.
    eta_epochs_raise (int >= 0): Specifies how many epochs are required to reach eta_max.      

    The following parameters can be set for the context and target model by setting 'context_' or 'target_' before the variable name.

    dataset_key (str): Modality keys for context and target datasets.
    optimizer (torch.optim.Optimizer): Optimizer for the training procedure.
    hidden_dims_enc_outer (List of ints > 0): List of integers specifying the hidden neuron sizes for the outer encoder model.
    hidden_dims_l_enc (List of ints > 0): List of integers specifying the hidden neuron sizes for the library encoder model.
    hidden_dims_dec (List of ints > 0): List of integers specifying the hidden neuron sizes for the decoder model.
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
    b_s (int > 0): Batch size during training.
    data_distr ('zinb' or 'nb'): The type of distribution of the data.
    dispersion ('dataset', 'batch', or 'cell'): How to compute the dispersion parameter.
                    'dataset' keeps the dispersion of a gene constant for the whole dataset.
                    'batch' keeps the dispersion of a gene constat for every cell of a batch
                    'cell' allows for individual dispersion for every cell.
    beta_start (float >= 0): Weighting of the KL terms at the beginning of the training.
    beta_max (float >= 0 and >= beta_start):  Weighting of the KL terms after beta_epochs_raise epochs.
    beta_epochs_raise (int >= 0): Specifies how many epochs are required to reach beta_max.    

    Class methods:

    initialize: Defines the neural network models according to the specified parameters.
    save_to_directory: Saves the model parameters.
    save_mdata: Saves the changes to self.mdata
    load_from_directory: Loads the model parameters.
    pred_labels_nns_aligned_latent_space: Predicts cell labels for the target dataset with the aligned latent representations.
    pred_labels_nns_hom_genes: Predicts cell labels for the target dataset with data level NNS.
    compute_metrics: Computes the label transfer accuracy.
    compute_logfold_change: Computes the logfold change between homologous genes.
    eval_context: Evaluates the scPecies context model.
    eval_target: Evaluates the scPecies target model.
    train_context: Trains the scPecies context model.
    train_target: Trains the scPecies target model.

    Helpers for class methods:

    most_frequent: Helper for pred_labels_nns_aligned_latent_space.
    create_directory: Helper for save_to_directory and load_from_directory.
    compare_elements: Helper for load_from_directory.
    compare_lists: Helper for load_from_directory.
    average_slices: Helper for compute_logfold_change.
    filter_outliers: Helper for compute_logfold_change.
    update_param: Helper for train_context and train_target

    Example Usage:  scPecies_instance = scPecies('cuda', mdata, save_path)
    """

    def __init__(self, 
                 device: str,
                 mdata: mu.MuData, 
                 directory: str,  
                 random_seed: int = 1234, 

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

                 context_layer_order: list = ['linear', 'layer_norm', ('act', nn.ReLU(), [-6, 6]), ('dropout', 0.1)],
                 target_layer_order: list = ['linear', 'layer_norm', ('act', nn.ReLU(), [-6, 6]), ('dropout', 0.1)],

                 context_b_s: int = 128,
                 target_b_s: int = 128,          

                 context_data_distr: str = 'zinb',
                 target_data_distr: str = 'zinb',

                 latent_dim: int = 10,

                 context_dispersion: str = 'batch',
                 target_dispersion: str = 'batch',

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

        # set the random seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.rng = np.random.default_rng(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed) 
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # defines dictionries which save the loss function trajectories
        self.context_history = {
            'Epoch' : [],
            'nELBO' : [],
            'nlog_likeli' : [],
            'KL-Div z' : [],
            'KL-Div l' : [],
            }

        self.target_history = {
            'Epoch' : [],
            'nELBO' : [],
            'nlog_likeli' : [],
            'KL-Div z' : [],
            'KL-Div l' : [],
            'Dist to neighbor' : [],
            }      

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
        self.pred_labels_nns_hom_genes()

    def initialize(self, initialize='both'):
        """
        Initializes the context and/or target scVI models based on the specified hyperparameter dictionaries.

        This method sets up the components of the scVI models (outer/inner encoder, library encoder, decoders, and optimizer) 
        and assigns them to the appropriate device. The initialization can be done for either the context model, 
        the target model, or both, depending on the argument passed.
        If the models are already defined their parameters are reinitialized.

        Parameters:
        initialize (str): A string parameter that determines which models to initialize. 
                        It accepts three values:
                        - 'context': Only initialize the context scVI model.
                        - 'target': Only initialize the target scVI model.
                        - 'both' (default): Initialize both context and target scVI models.

        Each model consists of an encoder (inner and outer), a library encoder, a decoder, and an optimizer. 
        The weights of the inner encoder are set when intializing the context model and shared with the target model. 
        These layer structures are configured with respective parameter dictionaries (`context_param_dict` or `target_param_dict`)

        Note: Both models are automatically initialized when defining a scPecies instance.

        Example Usage:
        scPecies_instance.initialize('context')  # Initializes only the context model
        scPecies_instance.initialize('target')   # Initializes only the target model
        scPecies_instance.initialize()           # Initializes both models
        """

        if initialize in ['context', 'both']:
            print('\nInitializing context scVI model.')
            self.encoder_inner = Encoder_inner(device=self.device,  param_dict=self.context_param_dict).to(self.device)
            self.context_encoder_outer = Encoder_outer(param_dict=self.context_param_dict).to(self.device)
            self.context_lib_encoder = Library_encoder(device=self.device, param_dict=self.context_param_dict).to(self.device)       
            self.context_decoder = Decoder(param_dict=self.context_param_dict).to(self.device) 
            self.context_optimizer = self.context_param_dict['optimizer'](
                list(self.context_encoder_outer.parameters()) + list(self.context_lib_encoder.parameters()) + list(self.context_decoder.parameters()) + list(self.encoder_inner.parameters()))

            self.encoder_inner.__name__ = 'encoder_inner'
            self.context_encoder_outer.__name__ = 'context_encoder_outer'
            self.context_lib_encoder.__name__ = 'context_lib_encoder'
            self.context_decoder.__name__ = 'context_decoder'        
            self.context_optimizer.__name__ = 'context_optimizer'   

        if initialize in ['target', 'both']:
            print('\nInitializing target scVI model.')
            self.target_encoder_outer = Encoder_outer(param_dict=self.target_param_dict).to(self.device)
            self.target_lib_encoder = Library_encoder(device=self.device, param_dict=self.target_param_dict).to(self.device)       
            self.target_decoder = Decoder(param_dict=self.target_param_dict).to(self.device) 
            self.target_optimizer =self.target_param_dict['optimizer'](
                list(self.target_encoder_outer.parameters()) + list(self.target_lib_encoder.parameters()) + list(self.target_decoder.parameters()))

            self.target_encoder_outer.__name__ = 'target_encoder_outer'
            self.target_lib_encoder.__name__ = 'target_lib_encoder'
            self.target_decoder.__name__ = 'target_decoder'        
            self.target_optimizer.__name__ = 'target_optimizer'   


    def create_directory(self):
        """
        Creates a specified directory along with predefined subdirectories if they do not already exist.

        This static method checks if the directory specified by `self.directory` exists. If not, it creates
        this directory. Additionally, it ensures the creation of three predefined subdirectories within this directory:

        - 'figures': Intended to store figure files.
        - 'params': Intended to store the scPecies parameters, hyperparameters and training history.
        - 'dataset': Intended to store the modifies mu.Mdata object.

        Note: A directory is automatically create when defining a scPecies instance.

        Example Usage:
        >>> scPecies_instance.create_directory() 
        """

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            print(f"\nCeated directory '{self.directory}'.")

        subfolders = ['figures', 'params', 'dataset']
        for folder in subfolders:
            if not os.path.exists(self.directory+'/'+folder):
                subfolder_path = os.path.join(self.directory, folder)
                os.makedirs(subfolder_path)
                print(f"\nCeated directory '{self.directory+'/'+folder}'.")


    def save_to_directory(self, save='both'):
        """
        Saves the model parameters, history, and configurations to the directory specified by `self.directory`.

        This method allows for saving the current state of the model, including its parameters, 
        training history, and configuration details. It supports selective saving for either the 
        'context', 'target', or 'both' models.

        Parameters:
        save (str):
            - 'context': Save only the context model's state.
            - 'target': Save only the target model's state.
            - 'both' (default): Save states of both context and target models.

        Example Usage:
        >>> scPecies_instance.save_to_directory('context')  # Saves only the context model's state
        >>> scPecies_instance.save_to_directory('target')   # Saves only the target model's state
        >>> scPecies_instance.save_to_directory()           # Saves states of both context and target models
        """

        model_list = []
        if save in ['context', 'both']:
            model_list += [self.encoder_inner, self.context_encoder_outer, self.context_decoder, self.context_lib_encoder, self.context_optimizer]
            with open(self.directory+'/params/context_history.pkl', 'wb') as pickle_file:
                pickle.dump(self.context_history, pickle_file)
            with open(self.directory+'/params/context_param_dict.pkl', 'wb') as pickle_file:
                pickle.dump(self.context_param_dict, pickle_file)

        if save in ['target', 'both']:
            model_list += [self.target_encoder_outer, self.target_lib_encoder, self.target_decoder, self.target_optimizer]
            with open(self.directory+'/params/target_history.pkl', 'wb') as pickle_file:
                pickle.dump(self.target_history, pickle_file) 
            with open(self.directory+'/params/target_param_dict.pkl', 'wb') as pickle_file:
                pickle.dump(self.target_param_dict, pickle_file) 

        for model in model_list:
            torch.save(model.state_dict(), self.directory+'/params/'+model.__name__+'.pth')
        print('\nSaved models to path.')

    def save_mdata(self, name: str):  
        """
        Saves the MuData object to a specified file path.

        This method is responsible for persisting the results of scPecies in the layers of the MuData object to disk. 
        The data is saved in the H5MU format, which is a specialized format for storing multi-modal data.

        Example Usage:
        >>> scPecies_instance.save_mdata('my_mdata')  
        # This will save the MuData object to 'self.directory/dataset/my_mdata.h5mu'
        """

        self.mdata.write(self.directory+'/dataset/'+name+'.h5mu') 
        print('\nSaved mdata {}.'.format(self.directory))

    # Helper for load_from_directory
    def compare_elements(self, elem1, elem2):
        if type(elem1) != type(elem2):
            return False
        if isinstance(elem1, nn.Module):
            return type(elem1) == type(elem2)
        if isinstance(elem1, tuple):
            return all(self.compare_elements(sub_elem1, sub_elem2) for sub_elem1, sub_elem2 in zip(elem1, elem2))
        return elem1 == elem2

    # Helper for load_from_directory
    def compare_lists(self, list1, list2):
        if len(list1) != len(list2):
            return False
        return all(self.compare_elements(elem1, elem2) for elem1, elem2 in zip(list1, list2))


    def load_from_directory(self, load='both'): 
        """
        Loads the model parameters, history, and configurations from a specified directory.

        This method is designed to reinstate the state of the scPecies model from previously saved files. 
        It supports selective loading for either the 'context', 'target', or 'both' models.

        Parameters:
        load (str): Determines which model's state to load. It accepts three values:
            - 'context': Load only the context model's state.
            - 'target': Load only the target model's state.
            - 'both' (default): Load states of both context and target models.

        The method performs the following actions:
        - For each selected model ('context', 'target', or both), the method iterates through the components 
        like encoders, decoders, optimizers, etc., and loads their state dictionaries from the specified directory.
        - It also deserializes and loads the training history and hyperparameter configuration from '.pkl' files.
        - In case of any discrepancies between the loaded and current hyperparameters, it prints conflict information.
        In case of a conflict it trys to initialize the saved model and overwrites the existing models

        Note:
        - The method assumes the existence of the directory defined in 'self.directory'.
        - The directory structure includes a 'params' folder where model states and configurations are expected to be found.

        Example Usage:
        >>> scPecies_instance.load_from_directory('context')  # Loads only the context model's state
        >>> scPecies_instance.load_from_directory('target')   # Loads only the target model's state
        >>> scPecies_instance.load_from_directory()           # Loads states of both context and target models
        """

        model_list = []
        if load in ['context', 'both']:
            model_list += [self.encoder_inner, self.context_encoder_outer, self.context_decoder, self.context_lib_encoder, self.context_optimizer]
            with open(self.directory+'/params/context_history.pkl', 'rb') as pickle_file:
                self.context_history = pickle.load(pickle_file)            
            with open(self.directory+'/params/context_param_dict.pkl', 'rb') as pickle_file:
                loaded_param_dict = pickle.load(pickle_file)   

            conflicts = 0
            for key in loaded_param_dict.keys() & self.context_param_dict.keys():
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
            model_list += [self.target_encoder_outer, self.target_lib_encoder, self.target_decoder, self.target_optimizer]
            with open(self.directory+'/params/target_history.pkl', 'rb') as pickle_file:
                self.target_history = pickle.load(pickle_file)            
            with open(self.directory+'/params/target_param_dict.pkl', 'rb') as pickle_file:
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

        for model in model_list:
            model.load_state_dict(torch.load(self.directory+'/params/'+model.__name__+'.pth'))              
        print('\nLoaded models from path.') 

    # helper for the pred_labels_nns_aligned_latent_space method.
    @staticmethod
    def most_frequent(arr):
        values, counts = np.unique(arr, return_counts=True)
        return values[np.argmax(counts)]

    def pred_labels_nns_aligned_latent_space(self, metric='euclidean', k_neigh=200, top_neigh=25, b_s=25, nns_key='_nns_aligned_latent_space'):
        """
        Predicts cell labels for the target dataset using a nearest neighbor search on the aligned latent space.

        This method utilizes the latent representations of cells in the target dataset to predict their cell types 
        based on the nearest neighbors from the context dataset. The predictions are made in a multi-step process 
        involving the computation of nearest neighbors, likelihood estimation, and label assignment.

        Parameters:
        metric (str): The distance metric to use for nearest neighbors computation. Default is 'euclidean'.
        k_neigh (int > 0): The number of nearest neighbors to consider for each cell in the target dataset. Default is 200.
        top_neigh (0 <= int <= 100): The number of top nearest neighbors to use for final label prediction. Default is 25.
        b_s (int > 0): Batch size to use for processing data. Default is 25 as this step is gpu intensive.
        nns_key (str): Key to store the indices of nearest neighbors in the target dataset's MuData object. Default is '_nns_aligned_latent_space'.

        The method performs the following actions:
        - Computes the k nearest neighbors for each cell in the target dataset's latent space using the specified metric.
        - Calculates the likelihood of these neighbors using the target model decoder.
        - Predicts the cell type for each cell in the target dataset based on the labels of the most likeli nearest neighbors.
        - Stores the indices of the nearest neighbors and the predicted labels in the target dataset's MuData object.

        Note:
        - The method assumes that the latent representations of context and target models have been computed and stored in the MuData objects.

        Example Usage:
        >>> scPecies_instance.pred_labels_nns_aligned_latent_space(metric='cosine', k_neigh=150, top_neigh=30, b_s=50)
        # This will predict cell labels using the cosine distance, considering 150 nearest neighbors and the top 30 for label prediction.
        """


        print('\n1) Computing latent space NNS with {} neighbors.\n'.format(str(k_neigh)))

        neigh = NearestNeighbors(n_neighbors=k_neigh, metric=metric)
        neigh.fit(self.mdata.mod[self.context_dataset_key].obsm['latent_mu'])

        _, indices = neigh.kneighbors(self.mdata.mod[self.target_dataset_key].obsm['latent_mu'])
        self.mdata.mod[self.target_dataset_key].obsm['ind'+nns_key] = indices

        self.target_lib_encoder.eval()        
        self.encoder_inner.eval()
        self.target_decoder.eval()

        steps = int(np.ceil(self.mdata.mod[self.target_dataset_key].n_obs/b_s+1e-10))

        likelihoods = []

        with torch.no_grad():
            tic = time.time()
            for step in range(steps):   
                if time.time() - tic > 0.5:
                    tic = time.time()
                    print('\r2) Calculate likelihoods for computed neighbors. Step {}/{}     '.format(str(step), str(steps)), end='', flush=True) 

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

        print('\n3) Predicting labels.')

        context_cell_labels = self.mdata.mod[self.context_dataset_key].obs[self.mdata.mod[self.context_dataset_key].uns['dataset_cell_key']].to_numpy()
        predicted_cell_type = np.stack([self.most_frequent(context_cell_labels[self.mdata.mod[self.target_dataset_key].obsm['ind'+nns_key][i][np.argsort(likelihoods[i])]][:top_neigh]) for i in range(self.mdata.mod[self.target_dataset_key].n_obs)])

        self.mdata.mod[self.target_dataset_key].obs['label'+nns_key] = predicted_cell_type     


    def pred_labels_nns_hom_genes(self):
        """
        Predicts cell labels in the target dataset based on data-level nearest neighbors (NNS) using homologous genes.

        This method employs a nearest neighbor search (NNS) approach, leveraging homologous gene expressions 
        to predict cell labels in the target dataset. It utilizes a voting mechanism based on the most 
        frequent labels among the nearest neighbors in the context dataset.
        The number of nearest neighbors is set when instanciating the scPecies class. 
        It can be retrieved and changed afterwards by calling scPecies_instance.target_param_dict['k_neigh'].

        The method performs the following actions:
        - Retrieves the indices of the nearest neighbors for each cell in the target dataset, computed based on homologous genes.
          The method assumes that the indices of nearest neighbors based on homologous genes are already computed and available.
          k_neigh has to be equal or lower than the number of neighbors during the computation in create_mdata_instance.setup_target_adata(neighbors)
        - Counts the occurrences of each cell type label among the nearest neighbors.
        - Assigns the most frequent label to each cell in the target dataset.
        - Calculates and stores the proportion of top agreement for each cell to quantify the confidence in label prediction.
          This is used during training of the target dataset. Alignment is performed only on samples with high confidence.
        - Saves the predicted labels and the agreement scores in the target dataset's MuData object.

        Example Usage:
        >>> scPecies_instance.pred_labels_nns_hom_genes()
        # This will use only one nearest neighbors for predicting cell labels based on homologous gene expressions.
        """

        print('\nEvaluating data level NNS and calculating cells with the highest agreement.')

        ind_neigh = self.mdata.mod[self.target_dataset_key].obsm['ind_nns_hom_genes']  
        ind_neigh = ind_neigh[:,:self.target_param_dict['k_neigh']]
        context_cell_labels = self.mdata.mod[self.context_dataset_key].obs[self.mdata.mod[self.context_dataset_key].uns['dataset_cell_key']].to_numpy()

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
        self.mdata.mod[self.target_dataset_key].obs['top_percent_nns_hom_genes'] = np.array([cell_type_counts[i][-1] for i in range(len(cell_type_counts))])
        self.mdata.mod[self.target_dataset_key].obs['label_nns_hom_genes'] = np.array([cell_type_counts[i][0] for i in range(len(cell_type_counts))])


    def compute_metrics(self):  
        """
        Computes various metrics to evaluate the label transfer accuracy between context and target datasets.
        
        It compares the performance of  data level and aligned latent space nearest neighbor strategies 
        for label prediction and computes metrics for each strategy.

        The method computes the following metrics:
        - Balanced Accuracy Score, Adjusted Rand Score, Adjusted Mutual Information Score, Fowlkes-Mallows Score.

        The results are stored in the MuData object of the target dataset for easy access and analysis.

        Note:
        - This function assumes that label predictions have already been performed using the corresponding methods.
        Call scPecies_instance.pred_labels_nns_hom_genes() and scPecies_instance.pred_labels_nns_aligned_latent_space() before.

        Example Usage:
        >>> scPecies_instance.compute_metrics()
        # Computes metrics for the label predictions and stores them in the target dataset's MuData object.
        """

        print('\nComputing metrics')
        context_cell_labels = self.mdata.mod[self.context_dataset_key].obs[self.mdata.mod[self.context_dataset_key].uns['dataset_cell_key']].to_numpy()
        target_cell_labels = self.mdata.mod[self.target_dataset_key].obs[self.mdata.mod[self.target_dataset_key].uns['dataset_cell_key']].to_numpy()
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
            self.mdata.mod[self.target_dataset_key].uns['prediction_df'+nns_key] = df

        self.mdata.mod[self.target_dataset_key].uns['metrics'] = metrics_dict

    # Helper for filter_outliers
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

    # Filters outliers from a latent_mu cell cluster based on the Mahalanobis distance and a specified confidence level.
    # Helper for compute_logfold_change
    @staticmethod
    def filter_outliers(data, confidence_level=0.9):
        mean = np.mean(data, axis=0)
        data_centered = data - mean
        cov_matrix = np.dot(data_centered.T, data_centered) / (data_centered.shape[0] - 1)
        cov_inv = np.linalg.inv(cov_matrix)

        # Compute Mahalanobis distance
        m_dist = np.sqrt(np.sum(np.dot(data_centered, cov_inv) * data_centered, axis=1))

        # Determine the threshold for the given confidence level using the chi-squared distribution
        df = mean.shape[0]  # Degrees of freedom (number of dimensions)
        threshold = np.sqrt(chi2.ppf(confidence_level, df))
        # Filter points within the threshold
        filtered_data_ind = m_dist < threshold
        outlier_ind = m_dist >= threshold
        return filtered_data_ind, outlier_ind

    def compute_logfold_change(self, eps=1e-6, lfc_delta=1, samples=10000, b_s=128, confidence_level=0.9):
        """
        Computes the log-fold change (LFC) for normalized gene expression modeled by the scVI models between context and target datasets.

        This method evaluates the differential expression of homologous genes between the context 
        and target datasets by calculating the log-fold change. It is useful for identifying genes 
        that are differentially expressed across species or conditions.

        Parameters:
        eps (float > 0): A small constant added to expression values to correct for lowly expressed genes with high LFC. Default is 1e-6.
        lfc_delta (float > 0): The threshold for considering significant log-fold changes. Default is 1.
        samples (int > 0): The number of samples to be taken from the cell type plugin estimator for computing the log-fold change. Default is 10000.
        b_s (int > 0): Batch size for processing the data. Default is 128.
        confidence_level (0 < float <= 1) Confidene level to filter cells from the context latent_mu cell clusters

        The method performs the following steps:
        - Retrieves the indices of homologous genes between context and target datasets.
        - For each cell type common to both datasets, calculates the log-fold change in expression levels.
          When the cell types of the traget dataset are unknown computes them for context every cell type. (Saved in 'lfc_df')
        - Estimates the probability of a gene being differentially expressed based on the lfc threshold. (Saved in 'prob_df')
        - Stores the results in the MuData object under the context modality in the .uns layer with keys 'lfc_df' and 'prob_df'.

        Example Usage:
        >>> scPecies_instance.compute_logfold_change()
        # Computes the log-fold change for homologous genes between context and target datasets.
        """

        self.mdata.mod[self.context_dataset_key].uns['lfc_delta'] = lfc_delta
        self.context_decoder.eval()   
        self.target_decoder.eval() 

        context_ind = np.array(self.context_param_dict['homologous_genes'])
        context_gene_names = self.mdata.mod[self.context_dataset_key].var_names.to_numpy()[context_ind]

        context_cell_labels = self.mdata.mod[self.context_dataset_key].obs[self.mdata.mod[self.context_dataset_key].uns['dataset_cell_key']].to_numpy()
        context_cell_types = np.unique(context_cell_labels)
        context_cell_index = {c : np.where(context_cell_labels == c)[0] for c in context_cell_types}

        if self.mdata.mod[self.target_dataset_key].uns['dataset_cell_key'] == None:
            joint_cell_types = context_cell_types

        else:
            target_cell_labels = self.mdata.mod[self.target_dataset_key].obs[self.mdata.mod[self.target_dataset_key].uns['dataset_cell_key']].to_numpy()
            target_cell_types = np.unique(target_cell_labels)
            joint_cell_types = np.intersect1d(context_cell_types, target_cell_types, return_indices=True)[0]

        df_lfc = pd.DataFrame(0, index=context_gene_names, columns=joint_cell_types)
        df_prob = pd.DataFrame(0, index=context_gene_names, columns=joint_cell_types)

        for cell_type in joint_cell_types:
            adata = self.mdata.mod[self.context_dataset_key][context_cell_index[cell_type]]

            filtered_data_ind, _ = self.filter_outliers(adata.obsm['latent_mu'], confidence_level=confidence_level)
            adata = adata[filtered_data_ind]

            steps = np.ceil(adata.n_obs/b_s).astype(int)    
            sampling_size = max(int(samples / adata.n_obs), 1)

            context_batches = self.mdata.mod[self.context_dataset_key].uns['batch_dict']
            target_batches = self.mdata.mod[self.target_dataset_key].uns['batch_dict']

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
        """
        Evaluates the context scVI model by computing latent and intermediate representations of the context dataset.
        - Stores these representations under the context modality (latent_mu, latent_sig, and inter) in the MuData object .obsm layer for later use.

        Note:
        - This method should be called after training the context model and before any downstream analysis.
        
        Example Usage:
        >>> scPecies_instance.eval_context()
        # Evaluates the context model and updates the MuData object with latent and intermediate representations.
        """

        self.context_encoder_outer.eval()  
        self.encoder_inner.eval()

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
                z_loc, z_log_sig = self.encoder_inner.encode(inter)             

                mu_list.append(z_loc.cpu().numpy())
                inter_list.append(inter.cpu().numpy())
                sig_list.append(z_log_sig.exp().cpu().numpy())
                
            self.mdata.mod[self.context_dataset_key].obsm['latent_mu'] = np.concatenate(mu_list)
            self.mdata.mod[self.context_dataset_key].obsm['latent_sig'] = np.concatenate(sig_list)
            self.mdata.mod[self.context_dataset_key].obsm['inter'] = np.concatenate(inter_list)
    

    def eval_target(self, save_intermediate=False):     
        """
        Evaluates the target scVI model by computing latent and intermediate representations of the target dataset.
        - Stores these representations under the target modality (latent_mu, latent_sig, and inter) in the MuData object .obsm layer for later use.
        If the by default it does not save the intermediate representations of the target dataset. 
        Set save_intermediate to true to store them. 

        Note:
        - This method should be called after training the target model and before any downstream analysis.
        
        Example Usage:
        >>> scPecies_instance.eval_target()
        # Evaluates the context model and updates the MuData object with latent and intermediate representations.
        """

        self.target_encoder_outer.eval()   
        self.encoder_inner.eval()

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
                z_loc, z_log_sig = self.encoder_inner.encode(inter)             

                mu_list.append(z_loc.cpu().numpy())
                inter_list.append(inter.cpu().numpy())
                sig_list.append(z_log_sig.exp().cpu().numpy())
                
            self.mdata.mod[self.target_dataset_key].obsm['latent_mu'] = np.concatenate(mu_list)
            self.mdata.mod[self.target_dataset_key].obsm['latent_sig'] = np.concatenate(sig_list)

            if save_intermediate:
                self.mdata.mod[self.target_dataset_key].obsm['inter'] = np.concatenate(inter_list)

    @staticmethod
    def update_param(parameter, min_value, max_value, steps):
        """
        Incrementally updates a parameter value during training towards a maximum value within specified steps.

        The increment is determined based on the range (max_value - min_value) and the number of steps provided.
        If the number of steps is zero or the minimum and maximum values are the same, the parameter is returned
        as is, without any modification.

        Parameters:
        parameter (float): The current value of the parameter to be updated.
        min_value (float): The value of the parameter at the beginning of the training.
        max_value (float): The maximum value of the range towards which the parameter should be updated.
        steps (int): The number of steps over which the parameter should reach the maximum value.

        Returns:
        float/int: The updated parameter value, which will not exceed the max_value.


        Example Usage:
        >>> ClassName.update_param(5, 0, 10, 20)  # Increments a parameter with value 5 from initial value 0 towards 10 in 20 steps. Returns 5.5
        """        

        if steps == 0 or min_value == max_value:
            return parameter

        parameter += (max_value - min_value) / steps
        return min(parameter, max_value)



    def train_context(self, epochs=None, raise_beta=True, save_model=True, early_stopping=True):
        """
        This method pretrains the context scVI on the context dataset. It involves multiple 
        training epochs where the model parameters are optimized to reduce the negative 
        evidence lower bound (nELBO).

        Parameters:
        - epochs (int, optional): The number of training epochs. If not specified, it is calculated based on 
        the dataset size.
        - raise_beta (bool): If True, gradually increases the weight of the KL-divergence 
        in the loss function to the provided maximum values according to the self.context_param_dict.
        - save_model (bool): If True, saves the model parameters and training history after 
        training.
        - early_stopping (bool): If True, stops training after there is no decrease in loss for 
        more than 5 epochs

        Note:
        - It is important to run this method before using the target model for any downstream tasks.

        Example Usage:
        >>> scPecies_instance.train_context(epochs=100, raise_beta=True, early_stopping=False)
        # Trains the context model for 100 epochs without early stopping.
        """
        
        b_s = self.context_param_dict['b_s']
        n_obs = self.mdata.mod[self.context_dataset_key].n_obs

        if epochs is None:        
            epochs = np.min([round((15000 / n_obs) * 400), 400])

        steps_per_epoch = int(n_obs/b_s)
        progBar = Progress_Bar(epochs, steps_per_epoch, ['nELBO', 'nlog_likeli', 'KL-Div z', 'KL-Div l'], ['nELBO last epoch', 'nlog last epoch'])
    
        print(f'\nPretraining on the context dataset for a maximum of {epochs} epochs and {epochs*steps_per_epoch} iterations.')

        self.context_encoder_outer.train()
        self.context_lib_encoder.train()        
        self.encoder_inner.train()
        self.context_decoder.train()

        avg_nelbo_list = []
        for epoch in range(epochs):
            perm = self.rng.permutation(n_obs)  

            if raise_beta: 
                self.context_beta = self.update_param(self.context_beta, self.context_param_dict['beta_start'], self.context_param_dict['beta_max'], self.context_param_dict['beta_epochs_raise'])    

            avg_nelbo, avg_nlog = 0, 0

            if epoch > 0:
                avg_nlog = np.mean(self.context_history['nlog_likeli'][-steps_per_epoch:])
                avg_nelbo = np.mean(self.context_history['nELBO'][-steps_per_epoch:])   
                avg_nelbo_list.append(avg_nelbo)

            if epoch >= 15 and early_stopping:
                if avg_nelbo_list[-5] < avg_nelbo:
                    print('\nStopped training prematurely as no progress was observed in the last five epochs.')
                    break


            for step in range(steps_per_epoch):         
                self.context_optimizer.zero_grad(set_to_none=True)

                batch_adata = self.mdata.mod[self.context_dataset_key][perm[step*b_s:(step+1)*b_s]]
                data_batch = torch.from_numpy(batch_adata.X.toarray()).to(self.device)
                label_batch = torch.from_numpy(batch_adata.obsm['batch_label_enc']).to(self.device)         
                lib_mu_batch = torch.from_numpy(np.array(batch_adata.obs['library_log_mean'])).to(self.device)          
                lib_sig_batch = torch.from_numpy(np.array(batch_adata.obs['library_log_std'])).to(self.device)  

                z, z_kl_div = self.encoder_inner(self.context_encoder_outer(data_batch, label_batch)) 
                l, l_kl_div = self.context_lib_encoder(data_batch, label_batch, lib_mu_batch, lib_sig_batch)         
                
                nlog_likeli = self.context_decoder(z, label_batch, l, data_batch)

                nelbo = self.context_beta * (z_kl_div + l_kl_div) + nlog_likeli
        
                nelbo.backward()
                self.context_optimizer.step() 

                self.context_history['Epoch'].append(epoch+1)
                self.context_history['nELBO'].append(nelbo.item())
                self.context_history['nlog_likeli'].append(nlog_likeli.item())
                self.context_history['KL-Div z'].append(z_kl_div.item())     
                self.context_history['KL-Div l'].append(l_kl_div.item())

                progBar.update(
                    self.context_history['Epoch'][-1], 
                    [self.context_history['nELBO'][-1], self.context_history['nlog_likeli'][-1], self.context_history['KL-Div z'][-1], self.context_history['KL-Div l'][-1]], 
                    [avg_nelbo, avg_nlog])

        if save_model == True:    
            self.save_to_directory('context')  

    def train_target(self, epochs=None, save_model=True, raise_beta=True, raise_eta=True, nns_key='_nns_hom_genes', alignment='inter', use_latent='z', early_stopping=True):
        """
        This method is responsible for training the target model using the specified target dataset. 
        It iterates over multiple epochs to optimize the model parameters by minimizing the 
        negative evidence lower bound (nELBO) along with an additional alignment term.

        Parameters:
        - epochs (int, optional): Number of epochs for training. If not provided, it is automatically 
        determined based on the size of the dataset.
        - save_model (bool): If True, saves the model parameters post-training.
        - raise_beta (bool): If True, the weight of the KL-divergence term in the loss 
        function is progressively increased to the provided maximum values according to the self.target_param_dict.
        - raise_eta (bool): If True, the weight of the alignment term in the loss 
        function is gradually increased to the provided maximum values according to the self.target_param_dict.
        - nns_key (str): Key for nearest neighbors indices used for alignment in the dataset.
        - alignment ('inter' or 'latent'): Where aligmnet should be performed, either 'inter' (intermediate space) or 'latent' (latent space).
        - use_latent ('z' or 'mu'): Determines whether to use 'z' (latent variables) or 'mu' (variational mean parameter) for alignment.
        - early_stopping (bool): If True, stops training after there is no decrease in loss for 
        more than 5 epochs

        Example Usage:
        >>> model_instance.train_target(epochs=50, save_model=True, raise_beta=True, 
                                        raise_eta=True, alignment='inter')
        # Trains the target model for 50 epochs with specified parameters and saves the model post-training.
        """


        b_s = self.target_param_dict['b_s']
        n_obs = self.mdata.mod[self.target_dataset_key].n_obs
        k_neigh = self.target_param_dict['k_neigh']
        top_percent = self.target_param_dict['top_percent']

        if epochs is None:        
            epochs = np.min([round((15000 / n_obs) * 400), 400])
        steps_per_epoch = int(n_obs/b_s)

        progBar = Progress_Bar(epochs, steps_per_epoch, ['nELBO', 'nlog_likeli', 'KL-Div z', 'KL-Div l', 'Dist to neighbor'], ['nELBO last epoch', 'nlog last epoch']) 
        print(f'\nTraining on the target dataset for a maximum of {epochs} epochs and {epochs*steps_per_epoch} iterations.')

        self.target_encoder_outer.train()
        self.target_lib_encoder.train()        
        self.target_decoder.train()
        self.encoder_inner.eval()

        avg_nelbo_list = []
        for epoch in range(epochs):
            perm = self.rng.permutation(n_obs)     
            if raise_beta:       
                self.target_beta = self.update_param(self.target_beta, self.target_param_dict['beta_start'], self.target_param_dict['beta_max'], self.target_param_dict['beta_epochs_raise'])    
            if raise_eta:
                self.eta = self.update_param(self.eta, self.target_param_dict['eta_start'], self.target_param_dict['eta_max'], self.target_param_dict['eta_epochs_raise'])    

            avg_nelbo, avg_nlog = 0, 0

            if epoch > 0:
                avg_nlog = np.mean(self.target_history['nlog_likeli'][-steps_per_epoch:])
                avg_nelbo = np.mean(self.target_history['nELBO'][-steps_per_epoch:])    
                avg_nelbo_list.append(avg_nelbo)


            if epoch > 15 and early_stopping:
                if avg_nelbo_list[-5] < avg_nelbo:
                    print('\nStopped training prematurely as no progress was observed in the last five epochs.')                    
                    break
   

            for step in range(steps_per_epoch): 
                self.target_optimizer.zero_grad(set_to_none=True)

                batch_adata = self.mdata.mod[self.target_dataset_key][perm[step*b_s:(step+1)*b_s]]

                data_batch = torch.from_numpy(batch_adata.X.toarray()).to(self.device)
                label_batch = torch.from_numpy(batch_adata.obsm['batch_label_enc']).to(self.device)         
                lib_mu_batch = torch.from_numpy(np.array(batch_adata.obs['library_log_mean'])).to(self.device)          
                lib_sig_batch = torch.from_numpy(np.array(batch_adata.obs['library_log_std'])).to(self.device)  

                inter = self.target_encoder_outer(data_batch, label_batch)

                z, z_kl_div = self.encoder_inner(inter) 
                l, l_kl_div = self.target_lib_encoder(data_batch, label_batch, lib_mu_batch, lib_sig_batch)   
                        
                nlog_likeli = self.target_decoder(z, label_batch, l, data_batch)                    
                ind_top = np.where(batch_adata.obs['top_percent'+nns_key].to_numpy()<top_percent/100)[0]  
                if np.shape(ind_top)[0] < 1: ind_top = np.reshape(np.random.randint(b_s), (1,))

                ind_neigh = np.reshape(batch_adata.obsm['ind'+nns_key][ind_top, :k_neigh], (np.shape(ind_top)[0]*k_neigh))

                if use_latent == 'z':
                    neigh_mu = torch.from_numpy(self.mdata.mod[self.context_dataset_key].obsm['latent_mu'][ind_neigh]).to(self.device)  
                    neigh_sig = torch.from_numpy(self.mdata.mod[self.context_dataset_key].obsm['latent_sig'][ind_neigh]).to(self.device)  
                    neigh_latent = neigh_mu + neigh_sig * self.encoder_inner.sampling_dist.sample(torch.Size([neigh_sig.size(dim=0)]))

                elif use_latent == 'mu':
                    neigh_latent = torch.from_numpy(self.mdata.mod[self.context_dataset_key].obsm['latent_mu'][ind_neigh]).to(self.device)  

                label_interl = torch.repeat_interleave(label_batch[ind_top], repeats=k_neigh, dim=0)
                lib_interl = torch.repeat_interleave(l[ind_top], repeats=k_neigh, dim=0)
                data_interl = torch.repeat_interleave(data_batch[ind_top], repeats=k_neigh, dim=0)

                outp = self.target_decoder.decode(neigh_latent, label_interl)

                nlog_likeli_neighbors = self.target_decoder.calc_nlog_likelihood(outp, lib_interl, data_interl).reshape(np.shape(ind_top)[0], k_neigh)
                best_pin_for_x = torch.argmin(nlog_likeli_neighbors, dim=1).cpu().numpy()

                if alignment == 'inter':
                    align_target = torch.from_numpy(self.mdata.mod[self.context_dataset_key].obsm['inter'][batch_adata.obsm['ind'+nns_key][ind_top, best_pin_for_x]]).to(self.device)
                    sqerror_align = torch.sum((inter[ind_top] - align_target)**2, dim=-1).mean()

                elif alignment == 'latent':
                    mu_h, _ = self.encoder_inner.encode(inter) 
                    align_target = torch.from_numpy(self.mdata.mod[self.context_dataset_key].obsm['latent_mu'][batch_adata.obsm['ind'+nns_key][ind_top, best_pin_for_x]]).to(self.device)
                    sqerror_align = torch.sum((mu_h[ind_top] - align_target)**2, dim=-1).mean() * (self.target_param_dict['dims_enc_outer'][-1] / self.target_param_dict['latent_dim']) / 5.0

                nelbo = self.target_beta * (z_kl_div + l_kl_div) + nlog_likeli + self.eta * sqerror_align

                nelbo.backward()
                self.target_optimizer.step() 
                
                self.target_history['Epoch'].append(epoch+1)
                self.target_history['nELBO'].append(nelbo.item())
                self.target_history['nlog_likeli'].append(nlog_likeli.item())
                self.target_history['KL-Div z'].append(z_kl_div.item())     
                self.target_history['KL-Div l'].append(l_kl_div.item())
                self.target_history['Dist to neighbor'].append(sqerror_align.item())                


                progBar.update(
                    self.target_history['Epoch'][-1], 
                    [self.target_history['nELBO'][-1], self.target_history['nlog_likeli'][-1], self.target_history['KL-Div z'][-1], self.target_history['KL-Div l'][-1], self.target_history['Dist to neighbor'][-1]], 
                    [avg_nelbo, avg_nlog])                
        
        if save_model == True:            
            self.save_to_directory('target')                                                         




class Progress_Bar():
    def __init__(self, epochs, steps, metrics_iter, metrics_batch=None, avg_over_n_steps=100, sleep_print=1):
        """
        Initializes a Progress_Bar class, a utility for displaying training progress during model training.
        It shows metrics such as epoch number, steps completed, estimated time remaining, and performance metrics.

        Parameters:
        epochs (int): Total number of epochs for training.
        steps (int): Number of steps per epoch.
        metrics_iter (list): A list of metrics to be displayed at each iteration.
        metrics_batch (list, optional): Additional metrics to be displayed after each epoch. Defaults to None.
        avg_over_n_steps (int): Number of steps over which to average the metrics. 
        sleep_print (int): Time interval (in seconds) between updates to the progress bar. 
        """
        self.epochs = epochs
        self.steps = steps     
        self.remaining_steps = self.epochs * steps
        self.avg_over_n_steps = avg_over_n_steps
        self.tic = time.time() 
        self.sleep_print = sleep_print
        self.dict = {'Progress' : "0.000%",
                    'ETA' : 0.0,
                    'Epoch' : int(1),
                    'Iteration' : int(0),
                    'ms/Iteration' : 0.0}
        self.avg = {key : [0.0] for key in metrics_iter + ['time']}
        self.avg['time'] = [time.time()]
        self.metrics_iter = metrics_iter
        self.metrics_batch = metrics_batch
        self.impr = {key : 0.0 for key in metrics_batch}

    @staticmethod
    def format_number(number, min_length):
        """
        Formats a number to a string with a specified minimum length.

        Parameters:
        number (float): The number to format.
        min_length (int): The minimum length of the formatted string.

        Returns:
        str: The formatted string representation of the number.
        """

        decimal_count = len(str(number).split('.')[0])  
        decimal_places = max(min_length - decimal_count, 0) 

        formatted_number = "{:.{}f}".format(number, decimal_places)
        return formatted_number
    
    def ret_sign(self, number, min_length):
        """
        Returns a string representation of a number with a sign, formatted to a specified minimum length.

        Parameters:
        number (float): The number to represent.
        min_length (int): The minimum length of the formatted string.

        Returns:
        str: A string representing the number with a sign (positive in red, negative in green, or '---' for zero).
        """

        if number > 0.0:
            sign_str = '\033[91m{}\033[00m'.format("+" + self.format_number(np.abs(number), min_length))
        elif number < 0.0:
            sign_str = '\033[92m{}\033[00m'.format("-" + self.format_number(np.abs(number), min_length))
        else:
            sign_str = '---'
        return  sign_str      

    def update(self, epoch, values, values_epoch=None):
        """
        Updates the progress bar with current training information.

        This method should be called within the training loop to update the progress bar's display.

        Parameters:
        epoch (int): The current epoch number.
        values (list): A list of current values for the metrics being tracked at each iteration.
        values_epoch (list, optional): A list of current values for the metrics being tracked after each epoch. Defaults to None.

        The method calculates the average time per step, the estimated time of arrival (ETA), updates the 
        progress metrics, and prints the progress bar.
        """

        toc = time.time()
        values.append(toc)
        self.remaining_steps -= int(1)

        for i, key in enumerate(self.avg.keys()):
            if len(self.avg[key]) >= self.avg_over_n_steps:
                del self.avg[key][0]
            self.avg[key].append(values[i])

        avg_time = (self.avg['time'][-1] - self.avg['time'][0]) / (len(self.avg['time']) - 1)
        self.dict['ETA'] = timedelta(seconds=int(self.remaining_steps * avg_time))        
        self.dict['ms/Iteration'] = self.format_number(avg_time*1000.0, 4)

        if epoch - self.dict['Epoch'] > 0 and epoch > 2 and values_epoch != None:
            self.impr = {key : values_epoch[j] - self.values_epoch_old[j]  for j, key in enumerate(self.metrics_batch)}
        elif epoch - self.dict['Epoch'] == 0 and values_epoch != None:
            self.values_epoch_old = values_epoch

        self.dict['Epoch'] = epoch

        if toc - self.tic > self.sleep_print:
            metric_string = [f'\033[95m{key}\033[00m: {self.dict[key]}' for key in self.dict.keys()] 
            if self.metrics_batch != None:
                metric_string += [f'\033[96m{key}\033[00m: {self.format_number(values_epoch[j], 5)} ({self.ret_sign(self.impr[key], 4)})' for j, key in enumerate(self.metrics_batch) if values_epoch[j] != 0]
            metric_string += [f'\033[33m{key}\033[00m: {self.format_number(np.mean(self.avg[key]), 5)}' for key in self.metrics_iter if np.mean(self.avg[key]) != 0]
            metric_string = "\033[96m - \033[00m".join(metric_string)
            print(f"\r{metric_string}.           ", end='', flush=True)   
            self.tic = time.time()          

        self.dict['Iteration'] += int(1)        
        self.dict['Progress']  = self.format_number(100.0 * self.dict['Iteration'] / (self.epochs * self.steps), 3)+'%'