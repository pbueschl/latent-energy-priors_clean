from lep.shapenet import create_split, discover_meshes


def test_create_split_shuffles_before_limit_and_is_deterministic():
    shape_ids = [f"id_{i:02d}" for i in range(30)]
    split_a = create_split(shape_ids, seed=7, limit=8)
    split_b = create_split(shape_ids, seed=7, limit=8)
    selected = set(split_a["train"] + split_a["val"] + split_a["test"])
    lexicographic_first = set(sorted(shape_ids)[:8])

    assert split_a == split_b
    assert len(selected) == 8
    assert selected != lexicographic_first


def test_discover_meshes_class_subdirs_uses_file_stems(tmp_path):
    skull_dir = tmp_path / "skull_mri"
    brain_dir = tmp_path / "brain"
    skull_dir.mkdir()
    brain_dir.mkdir()
    (skull_dir / "001_skull.stl").write_text("solid skull\nendsolid skull\n", encoding="utf-8")
    (brain_dir / "001_brain.stl").write_text("solid brain\nendsolid brain\n", encoding="utf-8")

    records = discover_meshes(str(tmp_path), synset="medshape", layout="class_subdirs")
    by_id = {record.shape_id: record for record in records}

    assert set(by_id) == {"001_brain", "001_skull"}
    assert by_id["001_skull"].mesh_path.endswith("skull_mri/001_skull.stl")
