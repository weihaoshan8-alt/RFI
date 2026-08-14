"""V13 开集 + V20 闭集四分类的软件调用最小示例。"""

from pathlib import Path

import numpy as np

from recognition_core import TwoStageV20RecognitionService


ROOT = Path(__file__).resolve().parent
service = TwoStageV20RecognitionService(
    ROOT / "model" / "v13_open_set_model.npz",
    ROOT / "model" / "v20_closed_set_long_window_model.joblib",
)

with np.load(ROOT / "test_data" / "recognition_test_v13_formal_4000.npz", allow_pickle=False) as source:
    iq = source["iq"]

result = service.predict_array(iq, cluster_unknown=True)
print("V13 accepted blocks:", result["stage1"]["known_blocks"])
print("V13 rejected blocks:", result["stage1"]["unknown_blocks"])
print("V20 groups:", len(result["stage2"]["records"]))
print("First V20 group:", result["stage2"]["records"][0])
