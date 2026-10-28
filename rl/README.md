# Learning Retention Strategies

Preventing subscription customer churn requires balancing proactive interventions against their financial costs over a lifecycle. In this directory, a tabular reinforcement learning approach is implemented to train an agent to maximize customer lifetime value.

## Environments

The simulation models customer satisfaction (from 0 to 3) and a hypothetical churn probability across a 12-month horizon:
* **Basic Environment (`BasicEnv`):** Standard simulation modeling satisfaction drift, action costs, and probabilistic churn.
* **Real Environment (`RealEnv`):** Extended simulation incorporating customer types, and fatigue effects from repeated interventions.

## Agents

* **`QLAgent`:** Simple tabular Q-learning agent utilizing epsilon-greedy exploration and sample-average incremental updates.
* **`DumbAgent`:** Fixed-policy baseline agent executing a single constant action (`wait`, `call`, or `discount`).
* **`RandomAgent`:** Uniform random action baseline.

## Usage

To run the benchmarking script and evaluate policies:

```bash
git clone https://github.com/SepehrAkbari/stop-the-churn
cd stop-the-churn/rl/
chmod +x run.sh
./run.sh
```

To run individual evaluations:

```bash
python -m rl.src.main --help
python -m rl.src.main --env basic --agent ql --show-policy
```

## Contributing

To contribute to this project, you can fork this repository and create pull requests. You can also open an issue if you find a bug or wish to make a suggestion.

## License

This project is licensed under the [GNU General Public License (GPL)](/LICENSE).