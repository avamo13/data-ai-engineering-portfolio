# Autonomous ML Experiment Investigator

A stateful LangGraph agent that investigates supervised-learning datasets, gathers evidence through constrained tools, proposes a baseline methodology, requests human approval, executes deterministic scikit-learn experiments, and handles critique or execution failures through bounded replanning.

## Architecture

```text
Task + dataset
      ↓
Diagnostic agent
      ↓
read-only tools
      ↓
structured ExperimentPlan
      ↓
human approval
      ↓
deterministic plan validation
      ↓
experiment execution
      ├── success
      ├── invalid_plan
      └── execution_error
              ↓
        methodology critic
          ├── accept → report
          └── revise → approval → retry
```

## Why this is agentic

The model adaptively decides:

- which diagnostics to request;
- when enough evidence has been collected;
- which validation strategy and metric to propose;
- whether critic feedback requires replanning.

The graph and deterministic Python enforce:

- allowed transitions;
- human approval;
- experiment execution;
- bounded retries;
- plan compatibility;
- metric computation.

## Diagnostic tools

The agent can inspect:

- schema and dtypes;
- target distribution;
- missingness;
- duplicates;
- candidate timestamps;
- repeated entity identifiers;
- feature-target associations;
- read-only DuckDB queries.

It cannot execute arbitrary Python or shell commands.

## Benchmark

Five synthetic tasks contain controlled methodological requirements:

| Dataset | Intended issue |
|---|---|
| Churn | severe class imbalance |
| Demand | chronological generalization |
| Machine failure | grouped validation |
| Loan default | post-outcome leakage |
| IID regression | ordinary IID validation |

### Tool-using agent

This execution reaches a methodologically acceptable final plan on **5 / 5** tasks.

| Dataset | Final split | Final metric | Tool calls | Experiments |
|---|---|---|---:|---:|
| Churn | stratified random | Average Precision | 9 | 1 |
| Demand | chronological | MAE | 8 | 3 |
| Machine failure | grouped | Average Precision | 9 | 1 |
| Loan default | stratified random | Average Precision | 7 | 3 |
| IID regression | random | RMSE | 6 | 1 |

Average:

- **7.8 tool calls / task**
- **1.8 experiment attempts / task**

## Schema-only baseline

The baseline sees only:

- task objective;
- target;
- problem type;
- column names;
- dtypes.

It also reaches **5 / 5 methodology success** in this run.

That result is deliberately retained rather than forcing a favorable comparison. The tasks are interpretable enough that a strong LLM can infer many constraints directly from the schema and objective.

The demonstrated value of the agent architecture is therefore **evidence, auditability, control, and recovery**, not superior headline accuracy on this benchmark.

## Execution-error recovery

A particularly useful interactive failure occurs when the planner treats `application_id` as a possible chronological proxy.

The experiment attempts to parse values such as:

```text
APP_00000
```

as datetimes and raises `DateParseError`.

The workflow catches the exception and stores:

```text
status = execution_error
error_type = DateParseError
error_message = ...
```

instead of terminating the graph.

The error becomes evidence for the critic / replanning path.

## Final experiment outcomes

| Dataset | Split | Metric | Result |
|---|---|---|---:|
| Churn | stratified random | Average Precision | 0.2733 |
| Demand | chronological | MAE | 37.7074 |
| Machine failure | grouped | Average Precision | 0.5294 |
| Loan default | stratified random | Average Precision | 0.3646 |
| IID regression | random | RMSE | 17.4004 |

The scores are not compared across tasks.

## Main findings

- tool use can turn assumptions into explicit observations;
- structured state makes agent decisions auditable;
- deterministic execution prevents arbitrary model-generated code from owning ML methodology;
- runtime exceptions can become recoverable observations;
- human approval provides an external safety boundary;
- the schema-only LLM is already very strong on interpretable methodology tasks;
- the critic can become overly cautious and trigger unnecessary reruns;
- benchmark design and scoring are themselves part of agent evaluation.

## Limitations

- only five synthetic tasks;
- informative schemas;
- single stochastic run;
- overactive critic on some cases;
- hand-authored methodology expectations;
- no benchmark of human approval;
- limited action/tool surface;
- no latency or API-cost optimization.

## Reproducibility

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example`:

```text
OPENAI_API_KEY=...
```

Then run:

```text
autonomous_ml_experiment_investigator_case_study.ipynb
```

The benchmark datasets are generated locally.
