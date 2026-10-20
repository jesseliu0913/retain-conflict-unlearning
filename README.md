# Conflict-Aware Unlearning

Implementation for **When Retain Constraints Conflict: Mitigating Forget-Retain Interference in Tabular Data**.

**Accepted to Findings of EMNLP 2026.**

CAU reduces interference between forget and retain constraints before unlearning. Sample selection removes retain records within the forget distribution's normalized-distance boundary. Feature selection masks retain columns ranked by their absolute standardized mean difference. The training objective combines answer-token activation steering on forget examples with language-model supervision on the selected retain examples.

## Install

Python 3.11 or newer is required. Install a PyTorch build suitable for your GPU, then:

```bash
pip install -e .
```

For optional clustering-based split preparation:

```bash
pip install -e '.[prepare]'
```

## Data

The pipeline accepts JSON arrays or JSONL records with `instruction`, `input`, and `output` fields:

```json
{"instruction": "Predict the outcome: 0 or 1.", "input": "Patient information: Age is 45, Glucose is 120.", "output": "0"}
```

Public tabular datasets are available on Hugging Face. These links provide data; their preprocessing and splits may differ from a particular experiment.

| Dataset | Hugging Face |
| --- | --- |
| Breast Cancer Wisconsin | [Dataset](https://huggingface.co/datasets/scikit-learn/breast-cancer-wisconsin) |
| Pima Indians Diabetes | [Dataset](https://huggingface.co/datasets/khoaguin/pima-indians-diabetes-database) |
| Diabetic Retinopathy Debrecen | [Dataset](https://huggingface.co/datasets/BenchmarkDatasets/Diabetic_Retinopathy_Debrecen) |
| Stroke Prediction | [Dataset](https://huggingface.co/datasets/MLLab-TS/healthcare_dataset_stroke) |
| Adult Income | [Dataset](https://huggingface.co/datasets/scikit-learn/adult-census-income) |

To prepare a sample-level forget request from a training CSV:

```bash
python -m cau.prepare --input data/train.csv --target Outcome \
  --instruction 'Predict the diabetes outcome. Respond with 0 or 1.' \
  --mode sample --forget-clusters 0 2 --output-dir data/split
```

Cluster IDs depend on the input ordering and encoder. Choose the clusters for your forget request. For a feature request, use `--mode feature --forget-columns SkinThickness Insulin`; the retain copy masks those columns. Keep held-out evaluation rows outside split preparation.

## Select and train

```bash
python -m cau.select --forget data/split/forget.json --retain data/split/retain.json \
  --mode sample --distance-percentile 90 --output data/retain_selected.json

python -m cau.train --model path/to/finetuned-model \
  --forget-file data/split/forget.json --retain-file data/retain_selected.json \
  --output-dir outputs/cau --epochs 1 --batch-size 4 \
  --gradient-accumulation 4 --learning-rate 1e-5
```

For supplied row-aligned embeddings, `python -m cau.select_embeddings --train data/train.json --forget data/split/forget.json --retain data/split/retain.json --embeddings data/embeddings.npy --near-percentile 10 --output data/retain_selected.json` removes the retain examples nearest the forget centroid. The embeddings must have one row per record in `--train`.

For feature selection, use `--mode feature --top-k 1` with `cau.select`. Numeric features must use the `column is value` format shown above. Feature selection preserves labels and masks the chosen input values as `unknown`.

Training starts from a checkpoint already fine-tuned on the original training set. The defaults steer layer 7 and update the down-projection weights in layers 5–7 of a Llama/Qwen-style decoder. Set `--module-regex` and `--trainable-regex` for other architectures. `--steering`, `--forget-weight`, and `--retain-weight` control the objective.

## Evaluate

```bash
python -m cau.evaluate --model outputs/cau --data data/test.json \
  --output outputs/evaluation.json
```

This reports exact-match accuracy and mean answer-token loss. Apply it separately to the forget, retain, and held-out splits.

## Code

- `cau/prepare.py`: tabular serialization and forget requests.
- `cau/select.py`: sample selection and feature masking.
- `cau/objective.py`: activation steering and retain supervision.
- `cau/train.py`, `cau/evaluate.py`: training and evaluation.

The activation-steering component follows [WMDP's representation-steering formulation](https://github.com/centerforaisafety/wmdp).

## Tests

```bash
python -m unittest discover -s tests -v
```
