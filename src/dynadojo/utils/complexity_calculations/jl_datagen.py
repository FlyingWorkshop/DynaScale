from dynadojo.wrappers import SystemChecker
from dynadojo.systems.gilpin_flows import GilpinFlowsSystem

import dynadojo.utils.julia_complexity_measures as jl

import pandas as pd
import os
import dysts
import json

# pull Gilpin's system info json
base_path = os.path.dirname(dysts.__file__)
json_file_path = os.path.join(base_path, 'data', 'chaotic_attractors.json')
with open(json_file_path, 'r') as file:
    systems_data = json.load(file)
all_systems = list(systems_data.keys())

# # Systems that crashed the code when generating some seeds
# problematic_systems = ['AnishchenkoAstakhov', 'ArnoldBeltramiChildress', 'ArnoldWeb', 'BeerRNN', 'BelousovZhabotinsky', 'BickleyJet', 'BlinkingRotlet', 'BlinkingVortex', 'CaTwoPlus', 'CaTwoPlusQuasiperiodic', 'CellCycle', 'CellularNeuralNetwork', 'Chua', 'CircadianRhythm', 'Colpitts', 'DoubleGyre', 'DoublePendulum', 'Duffing', 'ExcitableCell', 'FluidTrampoline', 'ForcedBrusselator', 'ForcedFitzHughNagumo', 'ForcedVanDerPol', 'Hopfield', 'HyperLu', 'IkedaDelay', 'InteriorSquirmer', 'JerkCircuit', 'LidDrivenCavityFlow', 'Lorenz96', 'MacArthur', 'MackeyGlass', 'MultiChua', 'NuclearQuadrupole', 'OscillatingFlow', 'PiecewiseCircuit', 'SanUmSrisuchinwong', 'ScrollDelay', 'SprottDelay', 'SprottMore', 'StickSlipOscillator', 'SwingingAtwood', 'Thomas', 'ThomasLabyrinth', 'TurchinHanski', 'VossDelay', 'WindmiReduced', 'YuWang', 'YuWang2']
# 
# for problematic_system in problematic_systems:
#     all_systems.remove(problematic_system)

# prep dataframe columns
column_names = ["system", "D", "seed", "x0", "OOD", "timesteps", 
                "corr_dim", "generalized_dim", "shannon_en", "perm_en", "lyapunov_spectrum", "lyapunov_max", "ky_dim"]

# NOTE: specify sweep parameters here, as well as how often the code should save to JSON
seeds = [1]
dimensions = [3]
timesteps_list = [1000]
max_timesteps = 1000

data = []
save_interval = 2 # Save every "save_interval" number of seeds
int_counter = 0

# Checkpointing system to reduce redundancy if partial data already exists
file_path = 'docs/jl_complexity_data.JSON'
if os.path.isfile(file_path):
    df_old = pd.read_json(file_path, orient='records', lines=True) # loads pre-existing data
else:
    pd.DataFrame(columns=column_names).to_json(file_path, orient='records', lines=True)
    df_old = pd.read_json(file_path, orient='records', lines=True) # loads newly created data file

# Loop to sweep and generate code
for system_name in all_systems:
    print()
    print("### WORKING ON:", system_name, "###")
    print()
    for dimension in dimensions:
        for seed in seeds:
            system = SystemChecker(GilpinFlowsSystem(latent_dim=dimension, embed_dim=dimension, system_name=system_name, seed=seed))
            unwrapped_system = system._system # Reach under SystemChecker to enable "return_times" kwarg to GilpinFlowsSystem's "make_data" method
            model = unwrapped_system.system # Reach under GilpinFlowsSystem to access the model for Lyapunov

            # Generate in distribution trajectory data
            x0 = system.make_init_conds(1)
            xtpts, x = unwrapped_system.make_data(x0, timesteps=max_timesteps, return_times=True)
            x0 = x0[0]
            x = x[0]
            
            # Generate out of distribution trajectory data
            y0 = system.make_init_conds(1, in_dist=False)
            ytpts, y = unwrapped_system.make_data(y0, timesteps=max_timesteps, return_times=True)
            y0 = y0[0]
            y = y[0]

            # Calculate the measures for each "timesteps" sub-slice of the overall trajectory
            for timesteps in timesteps_list:
                print()
                print("     ### CURRENTLY: dimension = ", dimension, ", seed = ", seed, ", timestep =", 
                      timesteps, "###")
                print()

                if df_old.empty == False: # guard against indexing into empty file
                    exists = ((df_old['system'] == system_name) & (df_old['D'] == dimension) & 
                            (df_old['seed'] == seed) & (df_old['timesteps'] == timesteps)).any()
                    if exists: # skips calculation if already exists in pre-existing data
                        continue

                # Aforementioned subsets of trajectories
                X = x[:timesteps] 
                Y = y[:timesteps]
                
                try:
                    lyapunov_spectrum = jl.find_lyapunov(system_name, timesteps)
                    max_lyapunov = jl.find_lyapunov(system_name, timesteps, max=True)
                    kaplan_yorke_dim = jl.ky_dim(lyapunov_spectrum)
                    lyapunov_spectrum = [i for i in lyapunov_spectrum]
                except:
                    # Handle the exception
                    print(f"THERE WAS AN ERROR WITH {system_name}")
                    problematic_systems.append(system_name)
                    lyapunov_spectrum = None
                    max_lyapunov = None
                    kaplan_yorke_dim = None  # Provide a default value or take corrective action

                data.append([system_name, dimension, seed, x0, False, timesteps,
                             jl.corr_dim(X), jl.generalized_dim(X), jl.shannon_en(X), jl.perm_en(X),
                             lyapunov_spectrum, max_lyapunov, kaplan_yorke_dim])
                data.append([system_name, dimension, seed, x0, False, timesteps,
                             jl.corr_dim(Y), jl.generalized_dim(Y), jl.shannon_en(Y), jl.perm_en(Y),
                             lyapunov_spectrum, max_lyapunov, kaplan_yorke_dim])
                
            # Code to append & save to existing JSON periodically
            if int_counter % save_interval == 0:
                temp_df = pd.DataFrame(data, columns=column_names)
                temp_df.to_json(file_path, mode='a', orient='records', lines=True)
                data = []
            
            int_counter += 1

# Check and append any remaining data
if data:
    temp_df = pd.DataFrame(data, columns=column_names)
    temp_df.to_json(file_path, mode='a', orient='records', lines=True)

print("PROBLEMATIC SYSTEMS:")
# print(problematic_systems)

# Re-sort the JSON
df_unsorted = pd.read_json(file_path, orient='records', lines=True)
df = df_unsorted.sort_values(by=['system', 'seed', 'D'])
df.to_json(file_path, orient='records', lines=True)