# Bomberman reinforcement learning

Machine Learning Essentials project by **Leander Werner and Juan Campo**, built
on the Bomberman framework supplied with the course. We develop tabular
Q-learning and SARSA agents, from coin collection to opponent-aware bombing
and tactical blocking.

## Repository layout

- `agent_code/final_agent/`: self-contained submitted agent and trained model.
- `agent_code/q_learning_agent/`, `agent_code/sarsa_agent/`: research agents;
  feature, reward, exploration and update versions are selected by configuration.
- Other `agent_code/` folders: framework agents, opponents and templates.
- `evaluation/task1/`: coin collection; `task2/`: crates and bombs without
  opponents; `task3/`: opponents; `task4/`: joint Q-learning/SARSA training.
  Each contains experiment configurations, runners and analysis/plot modules.
- `main.py`, `environment.py`, `agents.py`, `items.py`, `settings.py`: game engine.
- `results/`: local experiment outputs and checkpoints, excluded from Git.

## Installation and play

Use Python 3.11 and Git. From the repository root:

```bash
conda create -n bomberman_rl python=3.11
conda activate bomberman_rl
python -m pip install -r requirements.txt
python main.py play --my-agent final_agent --n-rounds 10 --no-gui
```

Omit `--no-gui` to display the game. The submitted agent loads
`agent_code/final_agent/my-saved-model.pt`: a 30,000-round five-step Q-learning
model with V12 features, V5 rewards and softmax exploration during training.
Evaluation is greedy. The original course Dockerfile is retained.

## Training and evaluation

Run commands from a clean, committed checkout. Configurations specify training
budgets, seeds, versions and evaluation opponents. Training starts from a fresh
Q-table; `evaluate_checkpoint` configurations instead require the named local
model. Existing experiment directories are never overwritten.

A short complete example:

```bash
python evaluation/task2/run_experiment.py evaluation/task2/configs/task2_infrastructure_smoke.json --validate-only
python evaluation/task2/run_experiment.py evaluation/task2/configs/task2_infrastructure_smoke.json
python evaluation/task2/analyze_experiment.py results/task2/experiments/task2_infrastructure_smoke/experiment.json
python evaluation/task2/plot_experiment.py results/task2/experiments/task2_infrastructure_smoke/experiment.json
```

Use the corresponding task directory and config for other experiments.
Validation does not run games. On macOS, `caffeinate -i` can precede a runner
command to prevent idle sleep.

Outputs live under `results/<task>/experiments/<experiment_name>/`, where the
name comes from the JSON config. `experiment.json` records configuration,
commit, seeds and model hashes; `models/` contains checkpoints. Analysis writes
`analysis/summary.json` and `analysis/plots/experiment_plots.pdf`.
Report configs generate these automatically. Retained intermediate evaluations
have their own indices under `checkpoints/`. Task 4 trains both learners in
shared games, then evaluates and analyses each separately.

## Main experiments

Links identify entry configurations; all historical configs remain available
in the respective `configs/` directories.

| Question | Configurations |
| --- | --- |
| Task 1 navigation and learning rate | [BFS](evaluation/task1/configs/task1_bfs_mask_long.json), [visit decay](evaluation/task1/configs/task1_visit_decay_long.json), [sparse coins](evaluation/task1/configs/task1_sparse9_bfs_multiseed.json) |
| Task 2 final performance | [25k rounds, three training seeds](evaluation/task2/configs/task2_parallel_masks_v6_coin_route_bomb_quality50_reward_v3_8_td5_night_25000_multiseed.json) |
| Extra target classes | [25k class variant](evaluation/task2/configs/Report_Task2_ConflictBuckets_25000.json); compare with the final Task 2 model trained on world/agent seeds 11/10011 |
| D4 and five-step updates | [Reference](evaluation/task2/configs/Report_Task2_Reference_15000.json), [no D4](evaluation/task2/configs/Report_Task2_NoD4_10000.json), [one-step](evaluation/task2/configs/Report_Task2_TD1_10000.json); use the reference's 10k snapshot |
| Task 2 softmax | [Q-learning 15k](evaluation/task2/configs/task2_parallel_masks_v6_coin_route_bomb_quality50_reward_v3_8_td5_long_15000_qlearning_softmax.json), [Q-learning 25k](evaluation/task2/configs/task2_parallel_masks_v6_coin_route_bomb_quality50_reward_v3_8_td5_night_25000_multiseed_qlearning_softmax.json), corresponding `_sarsa_softmax` configs |
| Pressure parameters | [100-round sweep](evaluation/task3/configs/task3_v12_pressure_sweep_100.json), [500-round confirmation](evaluation/task3/configs/task3_v12_pressure_confirmation_500.json), each budget per setting |
| Opponent safety and HOLD pooling | [15k reference](evaluation/task3/configs/Report_Task3_Reference_15000.json), [nominal safety](evaluation/task3/configs/Report_Task3_NominalOpponentSafety_15000.json), [no pooling](evaluation/task3/configs/Report_Task3_NoHoldPooling_15000.json) |
| Task 3 algorithms and exploration | 30k [Q epsilon](evaluation/task3/configs/task3_rule_based_3_v12_reward_v5_30000_night.json), [Q softmax](evaluation/task3/configs/task3_rule_based_3_qlearning_softmax_v12_reward_v5_30000_night.json), [SARSA epsilon](evaluation/task3/configs/task3_rule_based_3_sarsa_epsilon_v12_reward_v5_30000_night.json), [SARSA softmax](evaluation/task3/configs/task3_rule_based_3_sarsa_softmax_v12_reward_v5_30000_night.json) |
| Joint training | [30k epsilon](evaluation/task4/configs/task4_q_sarsa_v12_reward_v5_30000.json), [30k softmax](evaluation/task4/configs/task4_q_sarsa_softmax_v12_reward_v5_30000.json) |

Historical checkpoint evaluations need their original model files; these are
not interchangeable with the submitted softmax model. The pressure sweep,
confirmation and fallback configs require the separately supplied original
epsilon-trained checkpoint at
`results/reference_models/task3_v12_epsilon_30000.pt`, with SHA-256
`ae358ded45741031b38bef91da2214daf7c8212b5465297542f2089e83548102`.
Other checkpoint-only configs declare their prerequisites in `checkpoint.path`.

Evaluation seeds measure variability across games, not independent training
runs. Compare models at matching training budgets and exploration schedules;
check recorded opponent seeds before treating comparisons as paired.
