"""Dataset-split helpers shared by preprocessing and training entry points."""
import hashlib
import json
import os


def load_split_id_list(path):
    """Return ids in file order from a caption-record list or id-keyed JSON object."""
    if not path:
        raise ValueError("a split JSON path is required")
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        ids = [str(row["id"]) for row in data]
    elif isinstance(data, dict):
        ids = [str(k) for k in data]
    else:
        raise ValueError(f"{path}: expected a list of records or an id-keyed object")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate object ids")
    return ids


def load_split_ids(path):
    return set(load_split_id_list(path))


def split_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def training_and_validation_ids(train_split, test_split, size=100, output_dir=None):
    """Reserve the first `size` training records for validation; preserve test intact."""
    population = load_split_id_list(train_split)
    if not 0 <= size < len(population):
        raise ValueError(
            f"validation size must be between 0 and {len(population) - 1}, got {size}")
    chosen = population[:size]
    training = population[size:]
    test = load_split_id_list(test_split)
    assert_disjoint(("training", set(training)), ("validation", set(chosen)),
                    ("test", set(test)))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        record = {
            "train_source": os.path.abspath(train_split),
            "train_source_sha256": split_digest(train_split),
            "test_source": os.path.abspath(test_split),
            "test_source_sha256": split_digest(test_split),
            "selection": f"first {size} records of train_source",
            "training_ids": training,
            "validation_ids": chosen,
            "test_ids": test,
        }
        dst = os.path.join(output_dir, "validation_test_split.json")
        tmp = f"{dst}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(record, f, indent=2)
        os.replace(tmp, dst)
    return set(training), set(chosen)


def assert_disjoint(*named_sets):
    for i, (name_a, ids_a) in enumerate(named_sets):
        for name_b, ids_b in named_sets[i + 1:]:
            overlap = ids_a & ids_b
            if overlap:
                raise ValueError(f"{name_a}/{name_b} overlap: {sorted(overlap)[:5]}")
