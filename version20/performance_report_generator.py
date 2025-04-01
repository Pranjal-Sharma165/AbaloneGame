import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from datetime import datetime


def load_battle_results(directory="battle_results"):
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found")
        return []

    file_pattern = os.path.join(directory, "*.json")
    result_files = glob.glob(file_pattern)

    if not result_files:
        print(f"No battle result files found in '{directory}'")
        return []

    print(f"Found {len(result_files)} battle result files")

    results = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

                data['file_path'] = file_path
                results.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    results.sort(key=lambda x: x.get('match_number', 0))

    return results


def extract_weights_and_performance(results):
    
    weight_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'games': 0, 'matches': []})

    for result in results:
        match_number = result.get('match_number', 0)

        champion_weights = result.get('champion', {}).get('weights', {})
        challenger_weights = result.get('challenger', {}).get('weights', {})

        stats = result.get('stats', {})
        champion_wins = stats.get('champion_wins', 0)
        challenger_wins = stats.get('challenger_wins', 0)

        champion_weights_str = json.dumps(champion_weights, sort_keys=True)
        challenger_weights_str = json.dumps(challenger_weights, sort_keys=True)

        weight_performance[champion_weights_str]['weights'] = champion_weights
        weight_performance[champion_weights_str]['wins'] += champion_wins
        weight_performance[champion_weights_str]['losses'] += challenger_wins
        weight_performance[champion_weights_str]['games'] += champion_wins + challenger_wins
        weight_performance[champion_weights_str]['matches'].append(match_number)

        weight_performance[challenger_weights_str]['weights'] = challenger_weights
        weight_performance[challenger_weights_str]['wins'] += challenger_wins
        weight_performance[challenger_weights_str]['losses'] += champion_wins
        weight_performance[challenger_weights_str]['games'] += champion_wins + challenger_wins
        weight_performance[challenger_weights_str]['matches'].append(match_number)

    for weights_str, stats in weight_performance.items():
        if stats['games'] > 0:
            stats['win_rate'] = stats['wins'] / stats['games']
        else:
            stats['win_rate'] = 0

    return weight_performance


def analyze_battle_results(results):
    

    weight_performance = extract_weights_and_performance(results)

    sorted_weights = sorted(
        weight_performance.items(),
        key=lambda x: (x[1]['win_rate'], x[1]['wins'], -x[1]['losses']),
        reverse=True
    )

    match_data = {}
    for result in results:
        match_number = result.get('match_number', 0)
        stats = result.get('stats', {})
        champion_weights = result.get('champion', {}).get('weights', {})
        challenger_weights = result.get('challenger', {}).get('weights', {})

        match_data[match_number] = {
            'champion_weights': champion_weights,
            'challenger_weights': challenger_weights,
            'champion_wins': stats.get('champion_wins', 0),
            'challenger_wins': stats.get('challenger_wins', 0),
            'ties': stats.get('ties', 0)
        }

    weight_configs = {}
    for weights_str, stats in weight_performance.items():
        weight_configs[weights_str] = stats['weights']

    return {
        'weight_performance': weight_performance,
        'weight_configs': weight_configs,
        'sorted_weights': sorted_weights,
        'match_data': match_data
    }


def generate_performance_report(analysis_results, output_file="weight_performance_report.json"):
    
    weight_performance = analysis_results['weight_performance']
    weight_configs = analysis_results['weight_configs']
    sorted_weights = analysis_results['sorted_weights']

    report = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_configs': len(weight_performance),
        'configs_ranked': []
    }

    for i, (weights_str, stats) in enumerate(sorted_weights):
        config = weight_configs[weights_str]

        config_report = {
            'rank': i + 1,
            'weights': config,
            'wins': stats['wins'],
            'losses': stats['losses'],
            'games': stats['games'],
            'win_rate': stats['win_rate'],
            'matches_used': sorted(stats['matches'])
        }

        report['configs_ranked'].append(config_report)

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Performance report saved to: {output_file}")

    return output_file


def generate_evolution_charts(analysis_results, output_dir="evolution_charts"):
    
    match_data = analysis_results['match_data']

    os.makedirs(output_dir, exist_ok=True)

    matches = sorted(match_data.keys())
    if not matches:
        print("No match data available for charting")
        return []

    champion_weights = {}
    match_wins = {'champion': [], 'challenger': [], 'ties': []}

    for match in matches:
        data = match_data[match]

        match_wins['champion'].append(data['champion_wins'])
        match_wins['challenger'].append(data['challenger_wins'])
        match_wins['ties'].append(data['ties'])

        for key, value in data['champion_weights'].items():
            if key not in champion_weights:
                champion_weights[key] = []

                for m in range(min(matches), match):
                    champion_weights[key].append(None)

            while len(champion_weights[key]) < len(matches) - matches.index(match) - 1:
                champion_weights[key].append(None)

            champion_weights[key].append(value)

    chart_files = []

    plt.figure(figsize=(12, 6))
    plt.plot(matches, match_wins['champion'], 'b-', label='Champion Wins')
    plt.plot(matches, match_wins['challenger'], 'r-', label='Challenger Wins')
    plt.plot(matches, match_wins['ties'], 'g-', label='Ties')

    plt.title('AI Performance Over Matches')
    plt.xlabel('Match Number')
    plt.ylabel('Wins')
    plt.legend()
    plt.grid(True)

    win_chart_file = os.path.join(output_dir, 'win_evolution.png')
    plt.savefig(win_chart_file)
    plt.close()
    chart_files.append(win_chart_file)

    for key in champion_weights:
        try:

            if all(v is None for v in champion_weights[key]):
                continue

            valid_matches = []
            valid_values = []
            for i, (match, value) in enumerate(zip(matches, champion_weights[key])):
                if value is not None:
                    valid_matches.append(match)
                    valid_values.append(value)

            if valid_values:
                plt.figure(figsize=(12, 6))
                plt.plot(valid_matches, valid_values, 'b-', marker='o')

                plt.title(f'Evolution of {key} Weight')
                plt.xlabel('Match Number')
                plt.ylabel('Weight Value')
                plt.grid(True)

                weight_chart_file = os.path.join(output_dir, f'{key}_evolution.png')
                plt.savefig(weight_chart_file)
                plt.close()
                chart_files.append(weight_chart_file)
        except Exception as e:
            print(f"Error creating chart for {key}: {e}")

    try:
        plt.figure(figsize=(14, 8))

        for key in champion_weights:

            valid_matches = []
            valid_values = []
            for i, (match, value) in enumerate(zip(matches, champion_weights[key])):
                if value is not None:
                    valid_matches.append(match)
                    valid_values.append(value)

            if valid_values:
                min_val = min(valid_values)
                max_val = max(valid_values)
                range_val = max_val - min_val if max_val > min_val else 1

                normalized = [(v - min_val) / range_val for v in valid_values]
                plt.plot(valid_matches, normalized, marker='.', label=key)

        plt.title('Normalized Weight Evolution Over Matches')
        plt.xlabel('Match Number')
        plt.ylabel('Normalized Weight Value')
        plt.legend()
        plt.grid(True)

        combined_chart_file = os.path.join(output_dir, 'combined_weight_evolution.png')
        plt.savefig(combined_chart_file)
        plt.close()
        chart_files.append(combined_chart_file)
    except Exception as e:
        print(f"Error creating combined chart: {e}")

    print(f"Evolution charts saved to: {output_dir}")
    return chart_files


def create_top_performers_table(analysis_results, top_n=10):
    
    sorted_weights = analysis_results['sorted_weights']
    weight_configs = analysis_results['weight_configs']

    data = []
    for i, (weights_str, stats) in enumerate(sorted_weights[:top_n]):
        config = weight_configs[weights_str]

        row = {
            'Rank': i + 1,
            'Wins': stats['wins'],
            'Losses': stats['losses'],
            'Games': stats['games'],
            'Win Rate': f"{stats['win_rate']:.2%}",
        }

        for key, value in config.items():
            row[key] = round(value, 4)

        data.append(row)

    if data:
        df = pd.DataFrame(data)
        print(f"\nTop {top_n} Weight Configurations:")
        print(df)
        return df
    else:
        print("No weight configuration data available")
        return pd.DataFrame()


def find_best_weights(analysis_results):
    
    sorted_weights = analysis_results['sorted_weights']
    weight_configs = analysis_results['weight_configs']

    if sorted_weights:
        best_weights_str, best_stats = sorted_weights[0]
        best_weights = weight_configs[best_weights_str]

        print("\nBest Weight Configuration:")
        print(f"Wins: {best_stats['wins']}, Losses: {best_stats['losses']}, Win Rate: {best_stats['win_rate']:.2%}")
        print("Weights:", best_weights)

        return best_weights
    else:
        print("No weight configurations found")
        return {}


def analyze_weight_correlations(analysis_results):
    
    weight_performance = analysis_results['weight_performance']

    if not weight_performance:
        print("No weight data available for correlation analysis")
        return None

    data = []

    for weights_str, stats in weight_performance.items():
        if stats['games'] > 0:  # Only include configs that have been played
            row = {'win_rate': stats['win_rate']}
            row.update(stats['weights'])

            data.append(row)

    if not data:
        print("No valid weight data for correlation analysis")
        return None

    df = pd.DataFrame(data)

    try:
        corr = df.corr(numeric_only=True)

        print("\nWeight-Win Rate Correlations:")
        if 'win_rate' in corr:
            print(corr['win_rate'].sort_values(ascending=False))
        else:
            print("No correlations found")

        return corr
    except Exception as e:
        print(f"Error calculating correlations: {e}")
        return None


def main():
    
    print("Abalone AI Battle Results Analyzer")
    print("=" * 40)

    default_dir = "battle_results"
    directory = input(f"Enter battle results directory [default: {default_dir}]: ") or default_dir

    results = load_battle_results(directory)

    if not results:
        print("No results to analyze. Exiting.")
        return

    print(f"Analyzing {len(results)} battle results...")

    analysis_results = analyze_battle_results(results)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"analysis_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    report_file = os.path.join(output_dir, "weight_performance_report.json")
    generate_performance_report(analysis_results, report_file)

    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)  # Ensure charts directory exists
    generate_evolution_charts(analysis_results, charts_dir)

    top_n = int(input("How many top performers to display? [default: 10]: ") or 10)
    df = create_top_performers_table(analysis_results, top_n)

    if not df.empty:
        table_file = os.path.join(output_dir, "top_performers.csv")

        os.makedirs(os.path.dirname(table_file), exist_ok=True)
        df.to_csv(table_file, index=False)
        print(f"Top performers table saved to: {table_file}")

    best_weights = find_best_weights(analysis_results)
    if best_weights:
        best_weights_file = os.path.join(output_dir, "best_weights.json")
        with open(best_weights_file, 'w') as f:
            json.dump(best_weights, f, indent=2)
        print(f"Best weights saved to: {best_weights_file}")

    corr = analyze_weight_correlations(analysis_results)
    if corr is not None and not corr.empty:
        try:

            plt.figure(figsize=(10, 8))
            plt.title('Weight Value Correlations')
            plt.imshow(corr, cmap='coolwarm', interpolation='none', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
            plt.yticks(range(len(corr.columns)), corr.columns)

            corr_file = os.path.join(charts_dir, "weight_correlations.png")
            plt.tight_layout()
            plt.savefig(corr_file)
            plt.close()

            corr_data_file = os.path.join(output_dir, "weight_correlations.csv")
            corr.to_csv(corr_data_file)
            print(f"Correlation analysis saved to: {corr_data_file} and {corr_file}")
        except Exception as e:
            print(f"Error creating correlation visualization: {e}")

    print(f"\nAnalysis complete. All results saved to: {output_dir}")


if __name__ == "__main__":
    main()