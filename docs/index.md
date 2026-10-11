# Arena Benchmark

Arena uses Docker, ROS 2 Jazzy, Isaac Sim and HuNav. External model repositories submit native Nav2 FollowPath goals; Arena tracks them with RPP.

Arena owns the `arena` and `isaac` services. Model repositories are responsible for model installation and client processes. The InternNav, LightNav and NaVILA benchmark names remain aliases of one common runner.

[Model Integration and Evaluation](Model_Evaluation.md) is the central guide to the architecture, model repositories, NaVILA, InternNav and LightNav setup, shared interface, and evaluation workflow. All three model clients have completed integration; see the guide for validation boundaries.

- [Installation](installation.md)
- [Model Integration and Evaluation](Model_Evaluation.md)
- [Run Current Benchmarks](Model_Evaluation.md#benchmark-commands)
- [Artifacts and Validation](benchmark/outputs-and-metrics.md)
- [Troubleshooting](Model_Evaluation.md#troubleshooting)
