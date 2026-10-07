# Chassis research candidate

This sealed checkpoint is exported for integration and review. It is unassessed for physical skills and hardware deployment.

Use policy.onnx with the frozen 35D command ABI in deployment/interface.json and compare abi_test_vectors.json. For flat motion the RMCS profile needs this model identity and its intended command limits; the ordinary context is [1,0,0,0,0,0,0]. Jump requests require the frozen command-clock context and are not qualified by this export.

Validate package files with sha256sum -c SHA256SUMS.
