import tempfile

import numpy as np
import pytest

from smoe.utils.visualization.visualize import (
    visualize_expert_load_barv,
    visualize_expert_load_heatmap,
)


def test_visualization_expert_load_heatmap():
    load_sum = np.random.rand(16)
    visualize_expert_load_heatmap(
        load_sum,
        layer_idx=0,
        dataset_name="test",
        shape=(4, 4),
        save_dir=tempfile.mktemp(),
    )
    load_sum = np.random.randint(0, 5, size=(16,))
    visualize_expert_load_heatmap(
        load_sum,
        layer_idx=0,
        dataset_name="test",
        shape=(4, 4),
        save_dir=tempfile.mktemp(),
    )
    with pytest.raises(ValueError):
        visualize_expert_load_heatmap(
            load_sum,
            layer_idx=0,
            dataset_name="test",
            shape=(4, 4),
            save_dir=".gitignore",
        )


def test_visualization_expert_load_barv():
    load_sum = np.random.rand(16)
    visualize_expert_load_barv(
        load_sum,
        layer_idx=0,
        dataset_name="test",
        y_max=10,
        x_label="experts",
        save_dir=tempfile.mktemp(),
    )
    with pytest.raises(ValueError):
        visualize_expert_load_barv(
            load_sum,
            layer_idx=0,
            dataset_name="test",
            y_max=10,
            x_label="experts",
            save_dir=".gitignore",
        )
