#%%
# from init_rcParams import set_mpl_settings
# set_mpl_settings()
#%%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import network_from_pd as FHS_net

from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration

from dynsimf.models.components.PropertyFunction import PropertyFunction

class AmhaModel:
    

    def __init__(self, g = FHS_net.G0_relabled, alpha_i_mp=1, alpha_r_mp=1, beta_i_mp=1, beta_r_mp=1, mtp_a_i_individual=np.array([1, 1, 1]), 
                 mtp_a_r_individual=np.array([1, 1, 1]), mtp_b_i_individual=np.array([1, 1, 1, 1]), mtp_b_r_individual=np.array([1, 1, 1]), 
                 flat_ai=0, flat_ar=0, flat_bi=0, flat_br=0, number_of_iterations=200):
        
        self.g = g
        self.model = Model(self.g)
        
        self.alpha_i_mp = alpha_i_mp
        self.alpha_r_mp = alpha_r_mp
        self.beta_i_mp = beta_i_mp
        self.beta_r_mp = beta_r_mp
        self.mtp_a_i_individual = mtp_a_i_individual
        self.mtp_a_r_individual = mtp_a_r_individual
        self.mtp_b_i_individual = mtp_b_i_individual
        self.mtp_b_r_individual = mtp_b_r_individual
        self.flat_ai = flat_ai
        self.flat_ar = flat_ar
        self.flat_bi = flat_bi
        self.flat_br = flat_br
        self.number_of_iterations = number_of_iterations
        self.its = None

        
        mtp_ai = self.alpha_i_mp * self.mtp_a_i_individual # Multiplier alpha 
        mtp_ar = self.alpha_r_mp * self.mtp_a_r_individual # Multiplier alpha 
        mtp_bi = self.beta_i_mp * self.mtp_b_i_individual # Multiplier beta 
        mtp_br = self.beta_r_mp * self.mtp_b_r_individual # Multiplier beta 

        self.constants = {
            #Automatic, Social
            # DIVIDED BY 46, as 4.6 is the average in years between examinations, resulting in 10 iterations per year - so to run 5 years number_of_iterations should be 50
            'a_m': np.array([0.2153*mtp_ai[0] + flat_ai, 0.0189 * mtp_bi[0] + flat_bi]) / 46, 
            'a_h': np.array([0.0084*mtp_ai[1] + flat_ai, 0.0000 * mtp_bi[1] + flat_bi]) / 46,
            'm_a': np.array([0.1917*mtp_ar[0] + flat_ar, 0.0348 * mtp_br[0] + flat_br]) / 46,
            'm_h': np.array([0.0881*mtp_ai[2] + flat_ai, 0.0346 * mtp_bi[2] + flat_bi]) / 46,
            'h_a': np.array([0.0555*mtp_ar[1] + flat_ar, 0.0183 * mtp_br[1] + flat_br]) / 46,
            'h_m': np.array([0.3038*mtp_ar[2] + flat_ar, 0.0000 * mtp_br[2] + flat_br]) / 46,  # Social rate from A-> M ~ H
            'a_m_h': np.array([0.0*mtp_bi[3] + flat_bi]) / 46,
        }

        initial_state = {
            'state': list(nx.get_node_attributes(self.g, 'd_state_ego').values())
        }
        
        self.model.constants = self.constants
        self.model.set_states(['state'])
        self.model.add_update(self.update_state, {'constants': self.model.constants})
        self.model.set_initial_state(initial_state, {'constants': self.model.constants})

        self.correlations = PropertyFunction(
            'correlations',
            self.get_spatial_correlation,
            10,
            # {'nodes': self.model.get_nodes_state(self.model.nodes,'state')}
            {}
        )
        self.model.add_property_function(self.correlations)

    
    def update_state(self, constants):
        state = self.model.get_state('state')
        adjacency = self.model.get_adjacency()

        # Select different states
        abstain_indices = np.where(state == 0)[0]
        moderate_indices = np.where(state == 1)[0]
        heavy_indices = np.where(state == 2)[0]

        # Select all neighbours of each state
        abstain_nbs = adjacency[abstain_indices]
        moderate_nbs = adjacency[moderate_indices]
        heavy_nbs = adjacency[heavy_indices]

        # Get dummy vector for all nodes per state: e.g. n = 6, node 3  and 1 infected: [0,1,0,1,0,0]
        abstain_vec = np.zeros(len(state))
        abstain_vec[abstain_indices] = 1
        moderate_vec = np.zeros(len(state))
        moderate_vec[moderate_indices] = 1
        heavy_vec = np.zeros(len(state))
        heavy_vec[heavy_indices] = 1

        # Get vector of per type ego the adjacency for other type of friends
            # so if there is 3 abstain ego, who each only have 2 moderate friends, it will be
            # shape (3, n), filled with zeros except 2 ones when abstain_moderate
        abstain_moderate = abstain_nbs * moderate_vec
        abstain_heavy = abstain_nbs * heavy_vec
        moderate_abstain = moderate_nbs * abstain_vec
        moderate_heavy = moderate_nbs * heavy_vec
        heavy_abstain = heavy_nbs * abstain_vec
        heavy_moderate = heavy_nbs * moderate_vec

        # Get number of friends of certain type for each state
            #  size(n abstainers, int)
        num_a_m = abstain_moderate.sum(axis = 1)
        num_a_h = abstain_heavy.sum(axis = 1)
        num_m_a = moderate_abstain.sum(axis = 1)
        num_m_h = moderate_heavy.sum(axis = 1)
        num_h_a = heavy_abstain.sum(axis = 1)
        num_h_m = heavy_moderate.sum(axis = 1)

        # Get probability to change state:
            # size n abstainers, float
        a_to_m_prob = constants['a_m'][0] + num_a_m * constants['a_m'][1] + num_a_h * constants['a_m_h']
        a_to_h_prob = constants['a_h'][0] + num_a_h * constants['a_h'][1]
        m_to_a_prob = constants['m_a'][0] + num_m_a * constants['m_a'][1]
        m_to_h_prob = constants['m_h'][0] + num_m_h * constants['m_h'][1]
        h_to_a_prob = constants['h_a'][0] + num_h_a * constants['h_a'][1]
        h_to_m_prob = constants['h_m'][0] + num_h_m * constants['h_m'][1]

        # draw uniformly to see who makes transition
        draw_a_m = np.random.random_sample(len(a_to_m_prob))
        draw_a_h = np.random.random_sample(len(a_to_h_prob))
        draw_m_a = np.random.random_sample(len(m_to_a_prob))
        draw_m_h = np.random.random_sample(len(m_to_h_prob))
        draw_h_a = np.random.random_sample(len(h_to_a_prob))
        draw_h_m = np.random.random_sample(len(h_to_m_prob))

        # Node indicators for changing nodes
        nodes_a_to_m = abstain_indices[np.where(a_to_m_prob > draw_a_m)]
        nodes_a_to_h = abstain_indices[np.where(a_to_h_prob > draw_a_h)]
        nodes_m_to_a = moderate_indices[np.where(m_to_a_prob > draw_m_a)]
        nodes_m_to_h = moderate_indices[np.where(m_to_h_prob > draw_m_h)]
        nodes_h_to_a = heavy_indices[np.where(h_to_a_prob > draw_h_a)]
        nodes_h_to_m = heavy_indices[np.where(h_to_m_prob > draw_h_m)]

        # Update new state variable with changed states
        state[nodes_a_to_m] = 1
        state[nodes_a_to_h] = 2
        state[nodes_m_to_a] = 0
        state[nodes_m_to_h] = 2
        state[nodes_h_to_a] = 0
        state[nodes_h_to_m] = 1

        return {'state': state}

    


    def get_spatial_correlation(self):
        """
        Takes a list of node states (0, 1 or 2) and a networkx graph object and returns 
        the spatial correlation of state 0, 1 and 2.
        """
        state_list = self.model.get_state('state')
        # Get total number of nodes
        num_nodes = len(state_list)
        
        # Calculate the fraction of each state
        state0_frac = np.count_nonzero(state_list == 0) / num_nodes
        state1_frac = np.count_nonzero(state_list == 1) / num_nodes
        state2_frac = np.count_nonzero(state_list == 2) / num_nodes
        
        # Initialize spatial correlation values for each state
        state0_corr = 0
        state1_corr = 0
        state2_corr = 0
        
        # Iterate through all edges and calculate the spatial correlation of each state
        for edge in self.g.edges:
            i, j = edge
            if state_list[i] == 0 and state_list[j] == 0:
                state0_corr += 1
            elif state_list[i] == 1 and state_list[j] == 1:
                state1_corr += 1
            elif state_list[i] == 2 and state_list[j] == 2:
                state2_corr += 1
        
        # Normalize by the total number of edges and return the spatial correlations
        total_edges = self.g.number_of_edges()
        state0_corr /= total_edges
        state1_corr /= total_edges
        state2_corr /= total_edges
        
        return [state0_corr, state1_corr, state2_corr]


    def simulate(self, custom_iterations=None):
        # Check if user provided custom_iterations
        if custom_iterations:
            self.its = self.model.simulate(custom_iterations)
        else:
            self.its = self.model.simulate(self.number_of_iterations)

        # Get all states for all iterations
        iterations = self.its['states'].values()

        A = [np.count_nonzero(it == 0) for it in iterations]
        M = [np.count_nonzero(it == 1) for it in iterations]
        H = [np.count_nonzero(it == 2) for it in iterations]

        # Return number of ppl in each state for last iteration
        return np.array((A, M, H))

# %%

# amha_class = AmhaModel(G)
# %%
# amha_class.simulate(400)
#%%
# cor_dict = amha_class.model.get_properties()
# val = np.array(cor_dict['correlations'])

# # %%
# plt.plot(val[:,0])
# plt.plot(val[:,1])
# plt.plot(val[:,2])
# # %%
