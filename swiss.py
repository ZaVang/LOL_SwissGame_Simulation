import random
import math
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse


class Team:
    def __init__(self, name, index, group, area, elo, wins=0, losses=0):
        self.name = name
        self.id = index
        self.group = group
        self.area = area
        self.elo = elo
        self.initial_elo = elo 
        self.wins = wins
        self.losses = losses
        self.opponents = []

    def win_game(self):
        self.wins += 1

    def lose_game(self):
        self.losses += 1

    def is_qualified(self):
        return self.wins == 3

    def is_eliminated(self):
        return self.losses == 3

    def reset(self):
        self.wins = 0
        self.losses = 0
        self.elo = self.initial_elo
        self.opponents.clear() 
    
class Results:
    def __init__(self):
        # Initializing all possible outcomes to 0
        self.outcome_counts = {
            "3-0": 0,
            "3-1": 0,
            "3-2": 0,
            "2-3": 0,
            "1-3": 0,
            "0-3": 0
        }
        self.total_qualifications = 0
        self.total_elo = 0
        self.simulations = 0

    def update(self, team):
        outcome = f"{team.wins}-{team.losses}"
        if outcome in self.outcome_counts:
            self.outcome_counts[outcome] += 1
        self.total_elo += team.elo
        if team.is_qualified():
            self.total_qualifications += 1
        self.simulations += 1

    def avg_elo(self):
        return self.total_elo / self.simulations
    
    def display_outcomes(self):
        for outcome, count in self.outcome_counts.items():
            print(f"Outcome {outcome}: {count} times")

def required_rounds(total_teams, qualified_teams):
    # Given that we qualify teams when they get 3 wins, the rounds needed is log2(total_teams)
    return int(math.log2(total_teams))

def calculate_win_probability(rating1, rating2):
    return 1.0 / (1.0 + 10**((rating2 - rating1) / 400))

def update_elo(rating1, rating2, K, outcome):
    """outcome: 1 if team1 wins, 0 if team1 loses"""
    expected_outcome = calculate_win_probability(rating1, rating2)
    return rating1 + K * (outcome - expected_outcome)

def get_k_value(elo_diff):
    if elo_diff > 400:  
        return 64
    elif 200 <= elo_diff <= 400:  
        return 48
    elif 100 <= elo_diff <= 200:
        return 32
    else:  
        return 24

def simulate_game(team1, team2, K, win_matrix, bo3=False):
    if bo3:
        team1_score, team2_score = 0, 0
        for _ in range(3):
            win_probability = calculate_win_probability(team1.elo, team2.elo)
            if random.random() < win_probability:
                team1_score += 1
            else:
                team2_score += 1
            if team1_score == 2 or team2_score == 2:
                break
        winner = team1 if team1_score > team2_score else team2
    else:
        win_probability = calculate_win_probability(team1.elo, team2.elo)
        winner = team1 if random.random() < win_probability else team2
    
    if winner == team1:
        team1.elo = update_elo(team1.elo, team2.elo, K, 1)
        team2.elo = update_elo(team2.elo, team1.elo, K, 0)
        team1.win_game()
        team2.lose_game()
    else:
        team1.elo = update_elo(team1.elo, team2.elo, K, 0)
        team2.elo = update_elo(team2.elo, team1.elo, K, 1)
        team1.lose_game()
        team2.win_game()
        
    team1.opponents.append(team2)
    team2.opponents.append(team1)
    if winner == team1:
        win_matrix[team1.id][team2.id] += 1
    else:
        win_matrix[team2.id][team1.id] += 1
    
def draw_teams(group):
    """Returns matchups for given group of teams."""
    if len(group) % 2 != 0:
        raise ValueError("Odd number of teams in the group.")
    
    random.shuffle(group)
    return [(group[i], group[i+1]) for i in range(0, len(group), 2)]

def generate_matchups(teams):
    """Generate matchups based on records."""

    record_to_group = {}  # { (wins, losses): [team1, team2, ...], ...}
    
    for team in teams:
        key = (team.wins, team.losses)
        if key not in record_to_group:
            record_to_group[key] = []
        record_to_group[key].append(team)
    
    matchups = []
    for record, group in record_to_group.items():
        matchups.extend(draw_teams(group))
    
    return matchups

def simulate_swiss_round(teams, win_matrix, starting_round=1):
    if starting_round <= 1:
        # First round special draw
        first_round_teams = teams.copy()
        
        # Sort teams by their group attribute
        first_round_teams.sort(key=lambda x: x.group)

        # Divide teams into their respective groups
        groups = [first_round_teams[i:i+4] for i in range(0, len(first_round_teams), 4)]

        # Pairing 1st group with 4th and 2nd group with 3rd
        matchups = []
        for i, j in [(0, 3), (1, 2)]:
            while groups[i] and groups[j]:  # Ensure there are teams left to pair
                team_from_group_i = random.choice(groups[i])
                valid_opponents = [team for team in groups[j] if team.area != team_from_group_i.area]  # Filter out teams from the same area
                if not valid_opponents and matchups:
                    last_matchup = matchups.pop()
                    valid_opponents = [last_matchup[1]] if last_matchup[1].area != last_matchup[0].area else []
                    if not valid_opponents:
                        raise ValueError("Couldn't resolve valid opponents after attempting to swap.")
                    groups[i].append(last_matchup[0])
                    groups[j].append(last_matchup[1])

                team_from_group_j = random.choice(valid_opponents)
                groups[i].remove(team_from_group_i)
                groups[j].remove(team_from_group_j)
                matchups.append((team_from_group_i, team_from_group_j))
    else:
        # If we are starting from a later round, just generate the matchups based on current standings
        teams_in_play = [t for t in teams if not t.is_qualified() and not t.is_eliminated()]
        matchups = generate_matchups(teams_in_play)

    round_num = starting_round
    # Simulate the rounds
    while any(not t.is_qualified() and not t.is_eliminated() for t in teams):
        # print(f"Round {round_num} matchups:")
        for team1, team2 in matchups:
            # print(f"{team1.name} {team1.wins}-{team1.losses} vs {team2.name} {team2.wins}-{team2.losses}")
            bo3 = team1.wins == 2 or team1.losses == 2 or team2.wins == 2 or team2.losses == 2
            K = get_k_value(abs(team1.elo - team2.elo))
            simulate_game(team1, team2, K, win_matrix, bo3)
        
        # Prepare matchups for next round based on the Swiss system
        teams_in_play = [t for t in teams if not t.is_qualified() and not t.is_eliminated()]
        matchups = generate_matchups(teams_in_play)
        round_num += 1
    
    qualified_teams = [t for t in teams if t.is_qualified()]
    return qualified_teams

def win_matrix_to_rate(win_matrix):
    # 初始化胜率矩阵，大小与胜利矩阵相同
    rate_matrix = [[0 for _ in range(len(win_matrix))] for _ in range(len(win_matrix))]
    
    for i in range(len(win_matrix)):
        for j in range(len(win_matrix)):
            total_matches = win_matrix[i][j] + win_matrix[j][i]
            
            # 避免除数为零的情况
            if total_matches == 0:
                rate_matrix[i][j] = 0
            else:
                rate_matrix[i][j] = win_matrix[i][j] / total_matches

    return rate_matrix


def save_win_rate_heatmap(win_rate_matrix, teams, save_path='figs/win_rate_heatmap.png'):
    team_names = [team.name for team in teams]
    plt.figure(figsize=(10, 8))
    sns.heatmap(win_rate_matrix, annot=True, cmap='YlGnBu', xticklabels=team_names, yticklabels=team_names)
    plt.title('Win Rate Heatmap')
    plt.savefig(save_path)
    plt.close()
    
    
def save_elo_distribution(results_objects, teams, save_path='figs/elo_distribution.png'):
    team_names = [team.name for team in teams]
    elos = [res.avg_elo() for res in results_objects]
    plt.figure(figsize=(10, 8))
    sns.boxplot(x=team_names, y=elos)
    plt.title('Elo Distribution')
    plt.ylabel('Elo Rating')
    plt.xticks(rotation=45)
    plt.savefig(save_path)
    plt.close()


def save_histogram(results_objects, teams, save_path='figs/histogram.png'):
    team_names = [team.name for team in teams]
    wins = [res.total_qualifications for res in results_objects]
    plt.figure(figsize=(10, 8))
    sns.barplot(x=team_names, y=wins)
    plt.title('Total Wins Histogram')
    plt.ylabel('Wins')
    plt.xticks(rotation=45)
    plt.savefig(save_path)
    plt.close()


def save_pie_chart(results_objects, teams, save_path='figs/pie_chart.png'):
    team_names = [team.name for team in teams]
    qualifications = [res.total_qualifications for res in results_objects]
    plt.figure(figsize=(10, 8))
    plt.pie(qualifications, labels=team_names, autopct='%1.1f%%')
    plt.title('Qualification Distribution')
    plt.savefig(save_path)
    plt.close()
    
    
def save_stacked_barplot(results_objects, teams, save_path='figs/stacked_barplot.png'):
    # Extract team names
    team_names = [team.name for team in teams]

    # Prepare data
    outcomes = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]
    data = {outcome: [] for outcome in outcomes}
    
    for res in results_objects:
        for outcome in outcomes:
            data[outcome].append(res.outcome_counts[outcome])

    # Plot
    plt.figure(figsize=(10, 8))
    x_positions = range(len(team_names))  # Use range for x positions
    bottom_data = [0] * len(team_names)
    for outcome in outcomes:
        plt.bar(x_positions, data[outcome], label=outcome, bottom=bottom_data)
        bottom_data = [bottom_data[i] + data[outcome][i] for i in range(len(team_names))]
    
    plt.title('Outcome Stacked Barplot')
    plt.ylabel('Frequency')
    plt.xticks(x_positions, team_names, rotation=45)  # Set team names as xtick labels
    plt.legend()
    plt.tight_layout()  # Adjust layout to ensure everything fits properly
    plt.savefig(save_path)
    plt.close()


def save_win_loss_heatmap(results_objects, teams, save_path='figs/win_loss_heatmap.png'):
    team_names = [team.name for team in teams]
    outcomes = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]
    data = []
    
    for res in results_objects:
        row = [res.outcome_counts[outcome]/res.simulations for outcome in outcomes]
        data.append(row)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, xticklabels=outcomes, yticklabels=team_names, annot=True, cmap="YlGnBu")
    plt.title('Outcome Heatmap')
    plt.ylabel('Teams')
    plt.xlabel('Outcome')
    plt.savefig(save_path)
    plt.close()
    

def save_win_loss_pie_charts(results_objects, teams, save_path='figs/win_loss_pie_charts.png'):
    outcomes = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.ravel()

    for i, res in enumerate(results_objects):
        sizes = [res.outcome_counts[outcome] for outcome in outcomes]
        axes[i].pie(sizes, labels=outcomes, autopct='%1.1f%%')
        axes[i].set_title(teams[i].name)

    # Hide the remaining unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Outcome Pie Charts for Each Team', y=1.05)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    

def main(initial_teams, num_simulations, starting_round=1):    
    team_results = [Results() for _ in range(16)]
    win_matrix = [[0 for _ in range(16)] for _ in range(16)]

    for _ in range(num_simulations):
        teams = copy.deepcopy(initial_teams)
        qualified_teams = simulate_swiss_round(teams, win_matrix, starting_round)
        for idx, team in enumerate(teams):
            team_results[idx].update(team)
        
        # for team in teams:
        #     team.reset()

    rate_matrix = win_matrix_to_rate(win_matrix)

    save_win_rate_heatmap(rate_matrix, initial_teams)
    save_elo_distribution(team_results, initial_teams)
    save_histogram(team_results, initial_teams)
    save_pie_chart(team_results, initial_teams)
    save_stacked_barplot(team_results, initial_teams)
    save_win_loss_heatmap(team_results, initial_teams)
    save_win_loss_pie_charts(team_results, initial_teams)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swiss Round Simulation")
    # parser.add_argument("-k", "--K", type=int, default=32, help="K-factor for Elo calculations")
    parser.add_argument("-s", "--simulations", type=int, default=1000000, help="Number of simulations to run")
    parser.add_argument("-r", "--starting_round", type=int, default=1, help="Starting round for the simulation")
    args = parser.parse_args()
    
    ## set teams
    jdg = Team("JDG", 0, 1, 'lpl', 2000, 3, 0)
    gen = Team("GEN", 1, 1, 'lck', 1950, 3, 0)
    g2 = Team("G2", 2, 1, 'lec', 1900, 2, 1)
    nrg = Team("NRG", 3, 1, 'lcs', 1700, 2, 1)
    blg = Team("BLG", 4, 2, 'lpl', 1800, 2, 1)
    t1 = Team("T1", 5, 2, 'lck', 1850, 2, 1)
    fnc = Team("FNC", 6, 2, 'lec', 1700, 1, 2)
    c9 = Team("C9", 7, 2, 'lcs', 1700, 1, 2)
    lng = Team("LNG", 8, 3, 'lpl', 1850, 2, 1)
    kt = Team("KT", 9, 3, 'lck', 1800, 2, 1)
    mad = Team("MAD", 10, 3, 'lec', 1600, 1, 2)
    tl = Team("TL", 11, 3, 'lcs', 1600, 1, 2)
    wbg = Team("WBG", 12, 4, 'lpl', 1750, 1, 2)
    dk = Team("DK", 13, 4, 'lck', 1750, 1, 2)
    bds = Team("BDS", 14, 4, 'lec', 1500, 0, 3)
    gam = Team("GAM", 15, 4, 'vcs', 1500, 0 ,3)
    
    teams = [jdg, gen, g2, nrg,
             blg, t1, fnc, c9,
             lng, kt, mad, tl,
             wbg, dk, bds, gam]
    
    main(teams, args.simulations, args.starting_round)

