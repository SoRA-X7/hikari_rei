#!/usr/bin/env python3
"""Combine paired PR matches and print a GitHub Actions summary."""

import argparse
import json
from pathlib import Path

from versus import save, summarize_games


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pr-sha', required=True)
    parser.add_argument('--main-sha', required=True)
    parser.add_argument('--first-seed', type=int, required=True)
    parser.add_argument('--seed-count', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('summaries', type=Path, nargs='+')
    args = parser.parse_args()
    if args.seed_count < 1:
        parser.error('--seed-count must be positive')

    games = []
    for path in args.summaries:
        shard = json.loads(path.read_text())
        games.extend(dict(game, artifact=path.parent.name) for game in shard['results'])
    games.sort(key=lambda game: (game['seed'], game['order']))
    expected_pairs = [
        (seed, order)
        for seed in range(args.first_seed, args.first_seed + args.seed_count)
        for order in (['hikari', 'main'], ['main', 'hikari'])
    ]
    if [(game['seed'], game['order']) for game in games] != expected_pairs:
        parser.error('expected one game in each side order for every seed')

    result = summarize_games(games, 'main', 'main')
    result.update(pr_sha=args.pr_sha, main_sha=args.main_sha, seeds=args.seed_count)
    save(args.output, result)

    print(
        '## PR versus main\n\n'
        f"PR `{result['pr_sha']}` vs. main `{result['main_sha']}` across {result['seeds']} paired seeds.\n\n"
        '| Games | PR wins | Main wins | Draws | PR win rate |\n'
        '| ---: | ---: | ---: | ---: | ---: |\n'
        f"| {result['games']} | {result['hikari_wins']} | {result['main_wins']} | "
        f"{result['draws']} | {result['hikari_win_rate']:.1%} |\n\n"
        '<details>\n<summary>Per-game results</summary>\n\n'
        '| Game | Seed | Result | Decision | PR APM | Main APM | PR sent APM | Main sent APM | PR PPS | Main PPS |\n'
        '| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |'
    )
    winners = {'hikari': 'PR win', 'main': 'Main win', None: 'Draw'}
    decisions = {'top_out': 'Top out', 'time_limit': 'Time limit', 'apm': 'APM', 'sent_apm': 'Sent APM'}
    for number, game in enumerate(games, start=1):
        players = {player['name']: player for player in game['players']}
        pr, main = players['hikari'], players['main']
        decision = decisions[game['tiebreak'] or game['reason']]
        print(
            f"| {number} | {game['seed']} | {winners[game['winner']]} | {decision} | "
            f"{pr['apm']:.2f} | {main['apm']:.2f} | {pr['sent_apm']:.2f} | {main['sent_apm']:.2f} | "
            f"{pr['pps']:.2f} | {main['pps']:.2f} |"
        )
    print('\n</details>')


if __name__ == '__main__':
    main()
