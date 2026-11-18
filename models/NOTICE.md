# Model provenance

| file | source | license |
|---|---|---|
| `inat_birds.onnx`, `inat_bird_labels.txt` | Google Coral `mobilenet_v2_1.0_224_inat_bird_quant.tflite` (github.com/google-coral/test_data), converted by `tools/export_cpp_models.py` (dequantized, cut at the softmax) | Apache-2.0 |
| `ssd_birds.onnx`, `ssd_birds.anchors`, `coco_labels.txt` | Google Coral `ssdlite_mobiledet_coco_qat_postprocess.tflite`, converted (dequantized, post-processing removed → done in C++) | Apache-2.0 |
