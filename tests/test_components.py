import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from field.anonymizer import FaceAnonymizer
from field.gait_estimator import GaitEstimator
from field.object_mapper import ObjectMapper
from field.queue_writer import QueueWriter


class FakeInterpreter:
    def __init__(self, outputs):
        self.outputs = outputs
        self.input = {"index": 0, "shape": np.array([1, 4, 4, 3]),
                      "dtype": np.float32}

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [self.input]

    def get_output_details(self):
        return [{"index": i} for i in range(len(self.outputs))]

    def set_tensor(self, index, tensor):
        self.last_tensor = tensor

    def invoke(self):
        pass

    def get_tensor(self, index):
        return self.outputs[index]


class ComponentTests(unittest.TestCase):
    def test_missing_anonymizer_is_rejected(self):
        with self.assertRaises(ValueError):
            FaceAnonymizer("does-not-exist.xml")

    def test_object_mapper_decodes_ssd_outputs(self):
        mapper = ObjectMapper("missing.tflite")
        mapper.model = FakeInterpreter([
            np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32),
            np.array([[1.0]], dtype=np.float32),
            np.array([[0.9]], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        ])
        result = mapper.tag_frame(np.zeros((10, 20, 3), dtype=np.uint8))
        self.assertEqual(result[0].real_label, "person")
        self.assertEqual(result[0].fantasy_label, "traveler")

    def test_gait_descriptor_uses_pose_output(self):
        estimator = GaitEstimator("missing.tflite")
        keypoints = np.zeros((1, 1, 17, 3), dtype=np.float32)
        for index in (5, 6, 11, 12, 15, 16):
            keypoints[0, 0, index] = [0.5, 0.5, 0.9]
        estimator.model = FakeInterpreter([keypoints])
        descriptor = estimator.describe_burst([np.zeros((10, 10, 3), dtype=np.uint8)] * 3)
        self.assertIsInstance(descriptor, str)

    def test_queue_writes_unique_complete_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = QueueWriter(tmp, str(Path(tmp) / "not-mounted"))
            path = writer.write_record(
                np.zeros((8, 8, 3), dtype=np.uint8), None, None, [], None
            )
            image_path = Path(path).with_suffix(".jpg")
            self.assertTrue(Path(path).exists())
            self.assertTrue(image_path.exists())
            self.assertGreater(image_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
