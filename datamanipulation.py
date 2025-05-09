#!/usr/bin/env python
# coding: utf-8

"""
Pass Analysis Module for Football Match Data

This module analyzes passing patterns and creates adjacency matrices for two teams
from football match data. It processes player positions, pass success rates, and
creates visualizations of passing networks.
"""

import pandas as pd
import numpy as np
from typing import List, Set, Tuple, Dict
from pathlib import Path


class PassAnalyzer:
    """Class for analyzing passing patterns in football match data."""

    def __init__(self, data_dir: str):
        """
        Initialize the PassAnalyzer with data directory.

        Args:
            data_dir (str): Directory containing the match data files
        """
        self.data_dir = Path(data_dir)
        self.adj_pass = None
        self.overall_stats = None
        self.match_dribbles = None
        self.starting_eleven_passes = None
        self.player_ids_teama_set = set()
        self.player_ids_teamb_set = set()

    def load_data(self) -> None:
        """Load all required data files."""
        self.adj_pass = pd.read_csv(self.data_dir / "1484684C-0AFC-4383-B792-55FB5FBDFAFE_adjusted_passes_inverted.csv")
        self.overall_stats = pd.read_csv(self.data_dir / "1484684C-0AFC-4383-B792-55FB5FBDFAFE_adjusted_OverallStats.csv")
        self.match_dribbles = pd.read_csv(self.data_dir / "1484684C-0AFC-4383-B792-55FB5FBDFAFE_adjusted_dribbles.csv")

    @staticmethod
    def contains_zero(sub_on_list: str) -> bool:
        """
        Check if a substitution list contains zero.

        Args:
            sub_on_list (str): String representation of substitution list

        Returns:
            bool: True if zero is in the list, False otherwise
        """
        return 0 in eval(sub_on_list)

    def get_starting_players(self) -> None:
        """Identify and store starting players for both teams."""
        self.overall_stats['Sub on'] = self.overall_stats['Sub on'].astype(str)
        overall_stats_zero_data = self.overall_stats[self.overall_stats['Sub on'].apply(self.contains_zero)]
        overall_bool = overall_stats_zero_data[overall_stats_zero_data['Sub on'].apply(self.contains_zero).astype(bool)]

        starting11_teama = overall_bool[overall_bool['Match Team'] == 'Go Ahead Eagles MA 15-16']
        starting11_teamb = overall_bool[overall_bool['Match Team'] == 'Go Ahead Eagles MB 15-16']

        self.player_ids_teama_set = set(starting11_teama['id'].tolist())
        self.player_ids_teamb_set = set(starting11_teamb['id'].tolist())
        all_player_ids = self.player_ids_teama_set.union(self.player_ids_teamb_set)

        self.starting_eleven_passes = self.adj_pass[self.adj_pass['passedPlayerId'].isin(all_player_ids)]
        self.starting_eleven_passes['isSucceeded'] = self.starting_eleven_passes['isSucceeded'].astype(str)

    def calculate_passes(self, player_ids_set: Set[int]) -> pd.DataFrame:
        """
        Calculate pass statistics for a set of players.

        Args:
            player_ids_set (Set[int]): Set of player IDs to analyze

        Returns:
            pd.DataFrame: DataFrame containing pass statistics
        """
        passes_df = self.starting_eleven_passes[self.starting_eleven_passes['passedPlayerId'].isin(player_ids_set)]
        total_passes = passes_df.groupby('passedPlayerId').size().reset_index(name='num_of_pass')
        successful_passes = passes_df[passes_df['isSucceeded'].isin(['1.0', 'True'])].groupby('passedPlayerId').size().reset_index(name='suc_pass')
        result = pd.merge(total_passes, successful_passes, on='passedPlayerId', how='left').fillna(0)
        result['suc_pass'] = result['suc_pass'].astype(int)
        return result

    def calculate_median_positions(self, player_ids: Set[int]) -> pd.DataFrame:
        """
        Calculate median positions for players based on their passes.

        Args:
            player_ids (Set[int]): Set of player IDs to analyze

        Returns:
            pd.DataFrame: DataFrame containing median positions
        """
        passes_df = self.starting_eleven_passes[self.starting_eleven_passes['passedPlayerId'].isin(player_ids)].copy()
        return passes_df.groupby('passedPlayerId').agg(
            x=('passedPlayerPosX', 'median'),
            y=('passedPlayerPosY', 'median')
        ).reset_index()

    def calculate_pass_pairs(self, player_ids: Set[int]) -> pd.DataFrame:
        """
        Calculate pass pairs between players.

        Args:
            player_ids (Set[int]): Set of player IDs to analyze

        Returns:
            pd.DataFrame: DataFrame containing pass pair statistics
        """
        passes_df = self.starting_eleven_passes.loc[
            self.starting_eleven_passes['passedPlayerId'].isin(player_ids) &
            self.starting_eleven_passes['receivedPlayerId'].isin(player_ids)
        ].copy()
        passes_df.loc[:, 'pairs'] = passes_df['passedPlayerId'].astype(str) + '_' + passes_df['receivedPlayerId'].astype(str)
        return passes_df.groupby('pairs').size().reset_index(name='num_of_passes')

    def calculate_average_xT_gained(self, player_ids: Set[int]) -> pd.DataFrame:
        """
        Calculate average expected threat (xT) gained from passes.

        Args:
            player_ids (Set[int]): Set of player IDs to analyze

        Returns:
            pd.DataFrame: DataFrame containing average xT gained
        """
        passes_df = self.starting_eleven_passes[
            (self.starting_eleven_passes['passedPlayerId'].isin(player_ids)) &
            (self.starting_eleven_passes['isSucceeded'] == 1.0)
        ].copy()
        return passes_df.groupby('passedPlayerId')['receivedPlayer_xT_gained'].mean().reset_index(name='average_pass_value')

    def create_adjacency_matrix(self, passes: pd.DataFrame, player_ids: List[int]) -> pd.DataFrame:
        """
        Create an adjacency matrix for passes between players.

        Args:
            passes (pd.DataFrame): DataFrame containing pass data
            player_ids (List[int]): List of player IDs

        Returns:
            pd.DataFrame: Adjacency matrix of passes
        """
        matrix = pd.DataFrame(0, index=player_ids, columns=player_ids)
        for _, row in passes.iterrows():
            matrix.at[row['passedPlayerId'], row['receivedPlayerId']] += 1
        return matrix

    def process_team_data(self, team_name: str, player_ids: List[int], player_names_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and create final matrix for a team.

        Args:
            team_name (str): Name of the team
            player_ids (List[int]): List of player IDs
            player_names_df (pd.DataFrame): DataFrame containing player names

        Returns:
            pd.DataFrame: Processed team matrix
        """
        team_passes = self.starting_eleven_passes[
            self.starting_eleven_passes['passedPlayerId'].isin(player_ids) &
            self.starting_eleven_passes['receivedPlayerId'].isin(player_ids)
        ]
        
        team_matrix = self.create_adjacency_matrix(team_passes, player_ids)
        team_passes_stats = self.calculate_passes(set(player_ids))
        
        team_matrix['total_suc_pass'] = team_passes_stats.set_index('passedPlayerId')['suc_pass']
        team_matrix['total_num_of_pass'] = team_passes_stats.set_index('passedPlayerId')['num_of_pass']
        
        player_id_to_name = player_names_df.set_index('player_id')['player_name'].to_dict()
        team_matrix['Player'] = team_matrix.index.map(player_id_to_name)
        
        return team_matrix[['Player'] + [col for col in team_matrix.columns if col != 'Player']]

    def save_matrices(self, output_dir: str) -> None:
        """
        Save team matrices to CSV files.

        Args:
            output_dir (str): Directory to save the output files
        """
        output_path = Path(output_dir)
        teama_name_csv = pd.read_excel(output_path / "The data report of Go Ahead Eagles MA 15-1620220426.xls", sheet_name=1)
        teamb_name_csv = pd.read_excel(output_path / "The data report of Go Ahead Eagles MB 15-1620220426.xls", sheet_name=1)

        teama_players_id_name = teama_name_csv[teama_name_csv['id'].isin(self.player_ids_teama_set)][['id', 'Name']].rename(
            columns={'id': 'player_id', 'Name': 'player_name'}
        )
        teamb_players_id_name = teamb_name_csv[teamb_name_csv['id'].isin(self.player_ids_teamb_set)][['id', 'Name']].rename(
            columns={'id': 'player_id', 'Name': 'player_name'}
        )

        teama_matrix = self.process_team_data('Team A', list(self.player_ids_teama_set), teama_players_id_name)
        teamb_matrix = self.process_team_data('Team B', list(self.player_ids_teamb_set), teamb_players_id_name)

        teama_matrix.to_csv(output_path / 'teama_matrix.csv', index=True)
        teamb_matrix.to_csv(output_path / 'teamb_matrix.csv', index=True)


def main():
    """Main function to run the pass analysis."""
    data_dir = "/Users/umutevren/Downloads/14846"
    output_dir = "/Users/umutevren/Downloads"

    analyzer = PassAnalyzer(data_dir)
    analyzer.load_data()
    analyzer.get_starting_players()
    analyzer.save_matrices(output_dir)


if __name__ == "__main__":
    main() 