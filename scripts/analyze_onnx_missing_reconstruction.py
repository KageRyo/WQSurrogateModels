"""Post-hoc stage isolation for saved reconstruction ONNX results; no retraining."""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from edge_optimization import FEATURES, categories, sha, write_json
from edge_optimization_paths import artifact_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--archived-run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    out = artifact_path(args.run / "stage-analysis")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    frame = pd.read_csv(args.data_root / "data/dataV1.csv")
    summary = []
    with threadpool_limits(limits=1):
        for condition in ["missing_bod", "missing_nh3n", "missing_bod_nh3n"]:
            key = "s0-rf50-" + condition
            source = (
                args.archived_run / "jobs" / key / "indicator_reconstruction.joblib"
            )
            bundle = joblib.load(source, mmap_mode="r")
            pipeline = bundle["wqi_model"]
            saved = (
                args.run
                / "x86"
                / (key + "-indicator_reconstruction")
                / "external_10714.npz"
            )
            data = np.load(saved, allow_pickle=False)
            native = frame.iloc[data["row_id"]][FEATURES].copy()
            for c in bundle["missing_indicators"]:
                native[c] = data["native_reconstructed_" + c]
            restored = pd.DataFrame(
                data["onnx_reconstructed_features"],
                columns=FEATURES,
                index=native.index,
            )
            counterfactual = pipeline.predict(restored)
            stage_delta = np.abs(data["native64"] - counterfactual)
            final_delta = np.abs(counterfactual - data["onnx32"])
            transformed_native = pipeline[0].transform(native)
            transformed_onnx = pipeline[0].transform(restored)
            forest = pipeline[-1]
            leaf_native = forest.apply(transformed_native)
            leaf_onnx = forest.apply(transformed_onnx)
            changed_leaves = np.sum(leaf_native != leaf_onnx, axis=1)
            rows = []
            for i in np.flatnonzero(np.abs(data["native64"] - data["onnx32"]) > 1e-3):
                row = dict(
                    row_id=int(data["row_id"][i]),
                    native64=float(data["native64"][i]),
                    native_on_onnx_reconstruction=float(counterfactual[i]),
                    onnx32=float(data["onnx32"][i]),
                    changed_native_tree_leaves=int(changed_leaves[i]),
                    native_category=str(categories([data["native64"][i]])[0]),
                    onnx_category=str(categories([data["onnx32"][i]])[0]),
                    first_divergent_branches=[],
                )
                for t in np.flatnonzero(leaf_native[i] != leaf_onnx[i]):
                    tree = forest.estimators_[t].tree_
                    node = 0
                    left = transformed_native[i].astype(np.float32)
                    right = transformed_onnx[i].astype(np.float32)
                    while tree.children_left[node] != tree.children_right[node]:
                        feature = tree.feature[node]
                        threshold = tree.threshold[node]
                        native_left = bool(left[feature] <= threshold)
                        onnx_left = bool(right[feature] <= threshold)
                        if native_left != onnx_left:
                            row["first_divergent_branches"].append(
                                dict(
                                    tree=int(t),
                                    node=int(node),
                                    feature=FEATURES[feature],
                                    threshold=float(threshold),
                                    native_input_float32=float(left[feature]),
                                    onnx_reconstructed_input_float32=float(
                                        right[feature]
                                    ),
                                    native_goes_left=native_left,
                                    onnx_reconstruction_goes_left=onnx_left,
                                )
                            )
                            break
                        node = (
                            tree.children_left[node]
                            if native_left
                            else tree.children_right[node]
                        )
                rows.append(row)
            np.savez_compressed(
                out / (key + ".npz"),
                row_id=data["row_id"],
                native_on_onnx_reconstruction=counterfactual,
                changed_native_tree_leaves=changed_leaves,
            )
            write_json(out / (key + "-large-differences.json"), rows)
            summary.append(
                dict(
                    condition=condition,
                    source_sha256=sha(source),
                    prediction_sha256=sha(saved),
                    post_hoc=True,
                    rows=len(stage_delta),
                    native_reconstruction_to_onnx_reconstruction=dict(
                        max_abs_score_difference=float(stage_delta.max()),
                        rows_above_1e_3=int((stage_delta > 1e-3).sum()),
                        category_changes=int(
                            (
                                categories(data["native64"])
                                != categories(counterfactual)
                            ).sum()
                        ),
                    ),
                    same_onnx_reconstruction_native_to_onnx_final_stage=dict(
                        max_abs_difference=float(final_delta.max()),
                        rows_above_1e_3=int((final_delta > 1e-3).sum()),
                    ),
                    rows_with_changed_native_tree_leaves=int(
                        (changed_leaves > 0).sum()
                    ),
                    large_difference_rows=len(rows),
                    large_difference_rows_with_branch_evidence=sum(
                        bool(r["first_divergent_branches"]) for r in rows
                    ),
                )
            )
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
