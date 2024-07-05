#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[15]:


adj_pass= pd.read_csv("/Users/umutevren/Downloads/14846/1484684C-0AFC-4383-B792-55FB5FBDFAFE_adjusted_passes_inverted.csv")
overall_stats= pd.read_csv("/Users/umutevren/Downloads/14846/1484684C-0AFC-4383-B792-55FB5FBDFAFE_adjusted_OverallStats.csv")
match_dribbles=pd.read_csv("/Users/umutevren/Downloads/14846/1484684C-0AFC-4383-B792-55FB5FBDFAFE_adjusted_dribbles.csv")


# In[3]:


def contains_zero(sub_on_list):
    return 0 in eval(sub_on_list)
overall_stats['Sub on'] = overall_stats['Sub on'].astype(str)


# In[4]:


# Filter rows where "Sub on" contains 0
overall_stats_zero_data = overall_stats[overall_stats['Sub on'].apply(contains_zero)]

overall_bool= overall_stats_zero_data[overall_stats_zero_data['Sub on'].apply(contains_zero).astype(bool)]

starting11_teama = overall_bool[overall_bool['Match Team'] == 'Go Ahead Eagles MA 15-16']
starting11_teamb = overall_bool[overall_bool['Match Team'] == 'Go Ahead Eagles MB 15-16']

player_ids_teama = starting11_teama['id'].tolist()
player_ids_teamb = starting11_teamb['id'].tolist()

player_ids_teama_set = set(player_ids_teama)
player_ids_teamb_set = set(player_ids_teamb)
all_player_ids = player_ids_teama_set.union(player_ids_teamb_set)


# In[21]:


starting_eleven_passes = match_adjusted_pass[
    match_adjusted_pass['passedPlayerId'].isin(all_player_ids)
]


# In[23]:


starting_eleven_passes["isSucceeded"].unique()


# In[24]:


starting_eleven_passes['isSucceeded'] = starting_eleven_passes['isSucceeded'].astype(str)


# In[25]:


def calculate_passes(df, player_ids_teama_set, player_ids_teamb_set):
    # Helper function to count passes for a given set of player IDs
    def aggregate_passes(player_ids_set):
        passes_df = df[df['passedPlayerId'].isin(player_ids_set)]
        total_passes = passes_df.groupby('passedPlayerId').size().reset_index(name='num_of_pass')
        successful_passes = passes_df[passes_df['isSucceeded'].isin(['1.0', 'True'])].groupby('passedPlayerId').size().reset_index(name='suc_pass')
        result = pd.merge(total_passes, successful_passes, on='passedPlayerId', how='left').fillna(0)
        result['suc_pass'] = result['suc_pass'].astype(int)
        return result

    teama_passes = aggregate_passes(player_ids_teama_set)
    teamb_passes = aggregate_passes(player_ids_teamb_set)
    
    return teama_passes, teamb_passes


# In[78]:


def calculate_median_positions(df, player_ids):
    passes_df = df[df['passedPlayerId'].isin(player_ids)].copy()
    result = passes_df.groupby('passedPlayerId').agg(
        x=('passedPlayerPosX', 'median'),
        y=('passedPlayerPosY', 'median')
    ).reset_index()
    return result

teama_player_pass_locations = calculate_median_positions(starting_eleven_passes, player_ids_teama_set)
teamb_player_pass_locations = calculate_median_positions(starting_eleven_passes, player_ids_teamb_set)


# In[90]:


teamb_player_pass_locations


# In[91]:


teama_player_pass_locations


# In[26]:


teama_passes_starting, teamb_passes_starting = calculate_passes(starting_eleven_passes, player_ids_teama_set, player_ids_teamb_set)


# In[85]:


teamb_passes_starting


# In[32]:


def calculate_pass_pairs(df, player_ids):
    passes_df = df.loc[
        df['passedPlayerId'].isin(player_ids) &
        df['receivedPlayerId'].isin(player_ids)
    ].copy()
    passes_df.loc[:, 'pairs'] = passes_df['passedPlayerId'].astype(str) + '_' + passes_df['receivedPlayerId'].astype(str)
    result = passes_df.groupby('pairs').size().reset_index(name='num_of_passes')
    return result


# In[33]:


teama_pass_pairs = calculate_pass_pairs(starting_eleven_passes, player_ids_teama_set)
teamb_pass_pairs = calculate_pass_pairs(starting_eleven_passes, player_ids_teamb_set)


# In[35]:


def calculate_average_xT_gained(df, player_ids):
    passes_df = df[(df['passedPlayerId'].isin(player_ids)) & (df['isSucceeded'] == 1.0)].copy()
    result = passes_df.groupby('passedPlayerId')['receivedPlayer_xT_gained'].mean().reset_index(name='average_pass_value')
    return result


# In[36]:


teama_player_passes_value = calculate_average_xT_gained(starting_eleven_passes, player_ids_teama_set)
teamb_player_passes_value = calculate_average_xT_gained(starting_eleven_passes, player_ids_teamb_set)


# In[45]:


teama_name_csv = pd.read_excel("/Users/umutevren/Downloads/The data report of Go Ahead Eagles MA 15-1620220426.xls", sheet_name=1)
teamb_name_csv = pd.read_excel("/Users/umutevren/Downloads/The data report of Go Ahead Eagles MB 15-1620220426.xls", sheet_name=1)


# In[47]:


teama_name_filtered = teama_name_csv[teama_name_csv['id'].isin(player_ids_teama_set)]
teamb_name_filtered = teamb_name_csv[teamb_name_csv['id'].isin(player_ids_teamb_set)]

teama_players_id_name = teama_name_filtered[['id', 'Name']].rename(columns={'id': 'player_id', 'Name': 'player_name'})
teamb_players_id_name = teamb_name_filtered[['id', 'Name']].rename(columns={'id': 'player_id', 'Name': 'player_name'})


# In[ ]:





# In[50]:


teama_passes = starting_eleven_passes[starting_eleven_passes['passedPlayerId'].isin(player_ids_teama) & starting_eleven_passes['receivedPlayerId'].isin(player_ids_teama)]
teamb_passes = starting_eleven_passes[starting_eleven_passes['passedPlayerId'].isin(player_ids_teamb) & starting_eleven_passes['receivedPlayerId'].isin(player_ids_teamb)]


# In[52]:


# Create an adjacency matrix for each team
def create_adjacency_matrix(passes, player_ids):
    matrix = pd.DataFrame(0, index=player_ids, columns=player_ids)
    for _, row in passes.iterrows():
        matrix.at[row['passedPlayerId'], row['receivedPlayerId']] += 1
    return matrix


# In[72]:


teama_matrix = create_adjacency_matrix(teama_passes, player_ids_teama)
teamb_matrix = create_adjacency_matrix(teamb_passes, player_ids_teamb)

teama_total_suc_pass = teama_passes_starting.set_index('passedPlayerId')['suc_pass']
teama_total_num_of_pass = teama_passes_starting.set_index('passedPlayerId')['num_of_pass']
teamb_total_suc_pass = teamb_passes_starting.set_index('passedPlayerId')['suc_pass']
teamb_total_num_of_pass = teamb_passes_starting.set_index('passedPlayerId')['num_of_pass']


# In[73]:


teama_matrix['total_suc_pass'] = teama_total_suc_pass
teama_matrix['total_num_of_pass'] = teama_total_num_of_pass
teamb_matrix['total_suc_pass'] = teamb_total_suc_pass
teamb_matrix['total_num_of_pass'] = teamb_total_num_of_pass


# In[55]:


teama_matrix


# In[76]:


teamb_matrix


# In[77]:


teamb_matrix.index


# In[75]:


player_id_to_name_a = teama_players_id_name.set_index('player_id')['player_name'].to_dict()
teama_matrix['Player'] = teama_matrix.index.map(player_id_to_name_a)
teama_matrix = teama_matrix[['Player'] + [col for col in teama_matrix.columns if col != 'Player']]

player_id_to_name_b = teamb_players_id_name.set_index('player_id')['player_name'].to_dict()
teamb_matrix['Player'] = teamb_matrix.index.map(player_id_to_name_b)
teamb_matrix = teamb_matrix[['Player'] + [col for col in teamb_matrix.columns if col != 'Player']]





# In[ ]:


player_id_to_name = player_info.set_index('player_id')['player_name'].to_dict()

# Add the player names to the matrix dataframe
teamb_matrix['Player'] = teamb_matrix.index.map(player_id_to_name)

# Reorder the columns to place 'Player' as the first column
teamb_matrix = teamb_matrix[['Player'] + [col for col in teamb_matrix.columns if col != 'Player']]


# In[67]:


teamb_matrix.to_csv('/Users/umutevren/Downloads/teamb_matrix.csv', index=True)

teama_matrix.to_csv('/Users/umutevren/Downloads/teama_matrix.csv', index=True)

