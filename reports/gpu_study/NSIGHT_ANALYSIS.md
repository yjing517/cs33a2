# Nsight Systems 实测分析

所有统计来自真实 `.nsys-rep` 导出的 SQLite。warmup 被 NVTX capture range 排除。GPU kernel 通过 correlationId 关联到发起它的 CUDA API，再映射到 NVTX CPU 范围；没有把 GPU 执行时间直接与 CPU 范围求交。kernel sum 是 GPU 工作时间之和，不等于端到端墙钟时间。GEMM 比例按 kernel 名称识别，属于可复核的分类估计。


## medium-train_step-fp32-w5-n256-nsys

| 阶段 | NVTX CPU 范围 ms | GPU kernel sum ms | GEMM 比例 |
|---|---:|---:|---:|
| forward | 114.206 | 62.723 | 74.1% |
| loss | 0.219 | 0.123 | 0.0% |
| backward | 200.484 | 126.015 | 66.3% |
| optimizer | 98.731 | 72.663 | 0.0% |
| attention_scores_matmul | 6.983 | 2.173 | 53.4% |
| attention_mask_softmax | 7.492 | 5.108 | 0.0% |
| attention_output_matmul | 7.290 | 1.821 | 83.7% |

forward 最耗时的三个 kernel（完整名称与所有计数见 JSON）：

- `ampere_sgemm_128x64_tn`：169 次，43.766 ms
- `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>,`：48 次，2.994 ms
- `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::nativ`：290 次，2.967 ms

backward 最耗时的三个 kernel（完整名称与所有计数见 JSON）：

- `ampere_sgemm_128x64_nn`：120 次，20.035 ms
- `void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(T1::Params)`：48 次，19.131 ms
- `void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nt_align1>(T1::Params)`：49 次，18.882 ms

optimizer 最耗时的三个 kernel（完整名称与所有计数见 JSON）：

- `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>,`：1314 次，25.530 ms
- `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)`：876 次，24.212 ms
- `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>,`：219 次，6.038 ms

Attention 的 mask+Softmax GPU kernel sum=5.108 ms；QK/PV 两个矩阵乘相关范围合计=3.994 ms。Softmax FLOPs 较少，但包含多次读写和归约，FLOPs 比例不能预测相同的耗时比例。


## small-train_step-fp32-w5-n256-nsys

| 阶段 | NVTX CPU 范围 ms | GPU kernel sum ms | GEMM 比例 |
|---|---:|---:|---:|
| forward | 56.879 | 19.890 | 69.7% |
| loss | 0.418 | 0.124 | 0.0% |
| backward | 110.751 | 41.808 | 62.2% |
| optimizer | 53.166 | 22.449 | 0.0% |
| attention_scores_matmul | 3.311 | 0.794 | 52.3% |
| attention_mask_softmax | 3.706 | 1.949 | 0.0% |
| attention_output_matmul | 3.451 | 0.855 | 86.2% |

forward 最耗时的三个 kernel（完整名称与所有计数见 JSON）：

- `ampere_sgemm_128x64_tn`：85 次，12.710 ms
- `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>,`：24 次，1.139 ms
- `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::nativ`：146 次，0.995 ms

backward 最耗时的三个 kernel（完整名称与所有计数见 JSON）：

- `ampere_sgemm_128x64_nn`：84 次，11.766 ms
- `void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nt_align1>(T1::Params)`：25 次，4.976 ms
- `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>,`：172 次，4.219 ms

optimizer 最耗时的三个 kernel（完整名称与所有计数见 JSON）：

- `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>,`：666 次，8.063 ms
- `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)`：444 次，7.296 ms
- `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>,`：111 次，1.891 ms

Attention 的 mask+Softmax GPU kernel sum=1.949 ms；QK/PV 两个矩阵乘相关范围合计=1.649 ms。Softmax FLOPs 较少，但包含多次读写和归约，FLOPs 比例不能预测相同的耗时比例。


## small-train_step-fp32-w5-n512-nsys

| 阶段 | NVTX CPU 范围 ms | GPU kernel sum ms | GEMM 比例 |
|---|---:|---:|---:|
| forward | 69.701 | 44.734 | 63.2% |
| loss | 0.268 | 0.223 | 0.0% |
| backward | 100.677 | 94.488 | 58.3% |
| optimizer | 52.645 | 22.650 | 0.0% |
| attention_scores_matmul | 4.214 | 2.795 | 47.0% |
| attention_mask_softmax | 4.580 | 7.180 | 0.0% |
| attention_output_matmul | 4.472 | 2.286 | 90.3% |

forward 最耗时的三个 kernel（完整名称与所有计数见 JSON）：

- `ampere_sgemm_128x64_tn`：85 次，24.892 ms
- `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>,`：24 次，2.230 ms
- `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::nativ`：146 次，2.182 ms

backward 最耗时的三个 kernel（完整名称与所有计数见 JSON）：

- `ampere_sgemm_128x64_nn`：84 次，23.474 ms
- `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nt_align1>(T1::Params)`：60 次，11.797 ms
- `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>,`：172 次，10.672 ms

optimizer 最耗时的三个 kernel（完整名称与所有计数见 JSON）：

- `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>,`：666 次，8.124 ms
- `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)`：444 次，7.350 ms
- `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>,`：111 次，1.902 ms

Attention 的 mask+Softmax GPU kernel sum=7.180 ms；QK/PV 两个矩阵乘相关范围合计=5.081 ms。Softmax FLOPs 较少，但包含多次读写和归约，FLOPs 比例不能预测相同的耗时比例。


这些柱状图是从采集数据生成的分析图，不是 Nsight GUI 截图。完整版报告可由用户在 Nsight Systems GUI 打开 `.nsys-rep` 检查。
