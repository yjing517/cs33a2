# 未同步文件及原因

下载链路约10~20KB/s；为尽快交付，仅同步源码、报告、全部逐轮JSON、日志、10张PNG、kernel/算子摘要和CSV。以下原始/可再生文件完整保留在远程，不删除：

- reports/gpu_study/artifacts/medium-train_step-fp32-w5-n256-nsys.nsys-rep: 1.67 MiB。
- reports/gpu_study/artifacts/medium-train_step-fp32-w5-n256-nsys.sqlite: 6.54 MiB。
- reports/gpu_study/artifacts/small-forward-bf16-n128.memory.html: 1.16 MiB。
- reports/gpu_study/artifacts/small-forward-bf16-n128.pickle: 0.87 MiB。
- reports/gpu_study/artifacts/small-forward-bf16-n2048.memory.html: 1.16 MiB。
- reports/gpu_study/artifacts/small-forward-bf16-n2048.pickle: 0.87 MiB。
- reports/gpu_study/artifacts/small-forward-fp32-n128.memory.html: 0.88 MiB。
- reports/gpu_study/artifacts/small-forward-fp32-n128.pickle: 0.66 MiB。
- reports/gpu_study/artifacts/small-forward-fp32-n2048.memory.html: 0.89 MiB。
- reports/gpu_study/artifacts/small-forward-fp32-n2048.pickle: 0.66 MiB。
- reports/gpu_study/artifacts/small-forward-torch.trace.json: 13.24 MiB。
- reports/gpu_study/artifacts/small-train_step-bf16-n128.memory.html: 2.53 MiB。
- reports/gpu_study/artifacts/small-train_step-bf16-n128.pickle: 1.89 MiB。
- reports/gpu_study/artifacts/small-train_step-fp32-n128.memory.html: 2.17 MiB。
- reports/gpu_study/artifacts/small-train_step-fp32-n128.pickle: 1.63 MiB。
- reports/gpu_study/artifacts/small-train_step-fp32-w5-n256-nsys.nsys-rep: 0.89 MiB。
- reports/gpu_study/artifacts/small-train_step-fp32-w5-n256-nsys.sqlite: 3.60 MiB。
- reports/gpu_study/artifacts/small-train_step-fp32-w5-n512-nsys.nsys-rep: 0.89 MiB。
- reports/gpu_study/artifacts/small-train_step-fp32-w5-n512-nsys.sqlite: 3.60 MiB。
- reports/gpu_study/artifacts/small-train_step-torch.trace.json: 37.25 MiB。

包含6个快照、3个Nsight原报告、6个memory_viz HTML、2个Chrome trace和3个SQLite。需要进一步GUI查看时可单独下载。
综合终端日志采用交付时快照；原始活跃日志和pid文件留在远程。
显存诊断仅测1次，std=0不代表真实方差为零。residual_probe及checkpoint/compile GPU对照未执行。
