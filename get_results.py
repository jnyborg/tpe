import os
from os.path import join
from glob import glob
import json
import numpy as np
from collections import defaultdict



exp_path = 'outputs'
experiments = os.listdir(exp_path)
experiments = sorted(experiments)


name_map = {
    'austria_33UVP_2017': 'AT1',
    'france_30TXT_2017': 'FR1',
    'france_31TCJ_2017': 'FR2',
    'denmark_32VNH_2017': 'DK17',
    'denmark_32VNH_2018': 'DK18',
    'denmark_32VNH_2019': 'DK19',
    'denmark_32VNH_2020': 'DK20',
}

data = defaultdict(dict)
for experiment in experiments:
    results = glob(f'{exp_path}/{experiment}/overall_*.json')
    results = sorted(results)
    print(experiment)

    parts = experiment.split('_')
    model, train_tiles, name = parts[0], parts[1], '_'.join(parts[2:])
    if name == '':
        name = model

    data_results = {}
    for result in results:
        metrics = json.load(open(result))
        tile_name = result.split('/')[-1].replace('overall_', '').replace('.json', '')

        is_over_time = True
        if 'AT1' in train_tiles or 'FR1' in train_tiles or 'FR2' in train_tiles:
            is_over_time = False


        tile_name = '+'.join([name_map[x] for x in tile_name.split(',')])
        if not is_over_time:
            tile_name = tile_name.replace('DK17', 'DK1')
        else:
            continue  # TODO

        print('\t' + tile_name)

        tiles_result = {}
        for metric, scores in metrics.items():
            if metric not in ['accuracy', 'macro_f1']:
                continue
            mean = np.mean(scores) * 100
            std = np.std(scores) * 100
            tiles_result[metric] = [mean, std]
            print(f'\t\t{metric}: {mean:.1f}$\\pm${std:.1f}')
        data_results[tile_name] = tiles_result
    data[name].update(data_results)
print(data)

print('target results')
for model_name, tile_results in data.items():
    print(model_name)

    src_str = f'{model_name}'
    trg_str = f'{model_name}'
    latex_str = []

    avg_f1, avg_acc = [], []
    for tile, res in sorted(tile_results.items(), key=lambda x: x[0]):
        f = lambda x: f"{x['macro_f1'][0]:.1f} & {x['accuracy'][0]:.1f}"

        if '+' in tile:
            continue
        print(tile, f(res))
        avg_f1.append(res['macro_f1'][0])
        avg_acc.append(res['accuracy'][0])
        latex_str.append(f(res))
    latex_str.append(f"{np.mean(avg_f1):.1f} & {np.mean(avg_acc):.1f}")
    print(' && '.join(latex_str) + ' \\\\')

print()
print('source results')
for model_name, tile_results in data.items():
    print(model_name)

    src_str = f'{model_name}'
    trg_str = f'{model_name}'
    latex_str = []

    avg_f1, avg_acc = [], []
    for tile, res in sorted(tile_results.items(), key=lambda x: x[0]):
        f = lambda x: f"{x['macro_f1'][0]:.1f} & {x['accuracy'][0]:.1f}"

        if '+' not in tile:
            continue
        print(tile, f(res))
        avg_f1.append(res['macro_f1'][0])
        avg_acc.append(res['accuracy'][0])
        latex_str.append(f(res))
    latex_str.append(f"{np.mean(avg_f1):.1f} & {np.mean(avg_acc):.1f}")
    print(' && '.join(latex_str) + ' \\\\')
