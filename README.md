# vllm FP8 Latency and Throughput benchmarks on AMD MI300x

Documentation for vLLM Inferencing on AMD Instinct platforms. 

## Overview

vLLM is a toolkit and library for large language model (LLM) inference and serving. It deploys the PagedAttention algorithm, which reduces memory consumption and increases throughput by leveraging dynamic key and value allocation in GPU memory. vLLM also incorporates many recent LLM acceleration and quantization algorithms, such as fp8 GeMM, fp8 KV cache, continuous batching, flash attention, hip graph, tensor parallel, GPTQ, AWQ, and token speculation. In addition, AMD implements high-performance custom kernels and modules in vLLM to enhance performance further.

This documentation shows some reference performance numbers and the steps to reproduce it for the popular Llama 3.1 series models from Meta with a pre-built AMD vLLM docker optimized for an AMD Instinct™ MI300X accelerator.

It includes:

   · ROCm™ 6.2

   · vLLM 0.6.1

   · PyTorch 2.5dev (nightly)

## System configuration

The performance data below was measured on a server with MI300X accelerators with the following system configuration. The performance might vary with different system configurations.

| System  | MI300X with 8 GPUs  |
|---|---|
| BKC | 24.11 |
| ROCm | version ROCm 6.2.0 |
| amdgpu | build 2009461 |
| OS | Ubuntu 22.04 |
| Linux Kernel | 5.15.0-117-generic |
| BMCVersion | C2789.BC.0809.00 |
| BiosVersion | C2789.5.BS.1C11.AG.1 |
| CpldVersion | 02.02.00 |
| DCSCMCpldVersion | 02.02.00 |
| CX7 | FW 28.40.1000 |
| RAM | 1 TB |
| Host CPU | Intel(R) Xeon(R) Platinum 8480C |
| Cores | 224 |
| VRAM | 192 GB |
| Power cap | 750 W |
| SCLK/MCLK | 2100 Mhz / 1300 Mhz |

## Pull latest Docker

You can pull the image with `docker pull ghcr.io/saienduri/vllm:20241004`

### What is New

   · GEMMs tuning charts
   
## Reproducing benchmark results

### Use pre-quantized models

To make it easier to run fp8 Llama 3.1 models on MI300X, the quantized checkpoints are available on AMD Huggingface space as follows 

- https://huggingface.co/amd/Meta-Llama-3.1-8B-Instruct-FP8-KV 
- https://huggingface.co/amd/Meta-Llama-3.1-70B-Instruct-FP8-KV 
- https://huggingface.co/amd/Meta-Llama-3.1-405B-Instruct-FP8-KV
- https://huggingface.co/amd/grok-1-FP8-KV

Currently these models are private. Please join https://huggingface.co/amd to access. 

Download the model you want to run.  

These FP8 quantized checkpoints were generated with AMD’s Quark Quantizer. For more information about Quark, please refer to https://quark.docs.amd.com/latest/quark_example_torch_llm_gen.html

### Quantize your own models
This step is optional for you to use quantized models on your own. Take Llama 3.1 405B as an example. 

Download the Model View the Meta-Llama-3.1-405B model at https://huggingface.co/meta-llama/Meta-Llama-3.1-405B. Ensure that you have been granted access, and apply for it if you do not have access.

If you do not already have a HuggingFace token, open your user profile (https://huggingface.co/settings/profile), select "Access Tokens", press "+ Create New Token", and create a new Read token.

Install the `huggingface-cli` (if not already available on your system) and log in with the token you created earlier and download the model. The instructions in this document assume that the model will be stored under `/data/llama-3.1`. You can store the model in a different location, but then you'll need to update other commands accordingly. The model is quite large and will take some time to download; it is recommended to use tmux or screen to keep your session running without getting disconnected.

    sudo pip install -U "huggingface_hub[cli]"
    
    huggingface-cli login

Enter the token you created earlier; you do NOT need to save it as a git credential

Create the directory for Llama 3.1 models (if it doesn't already exist)

    sudo mkdir -p /data/llama-3.1
    
    sudo chmod -R a+w /data/llama-3.1

Download the model

    huggingface-cli download meta-llama/Meta-Llama-3.1-405B-Instruct --exclude "original/*" --local-dir /data/llama-3.1/Meta-Llama-3.1-405B-Instruct

Similarly, you can download Meta-Llama-3.1-70B and Meta-Llama-3.1-8B.

[Download and install Quark](https://quark.docs.amd.com/latest/install.html)

Run the quantization script in the example folder using the following command line:
export MODEL_DIR = [local model checkpoint folder] or meta-llama/Meta-Llama-3.1-405B-Instruct
#### single GPU
    python3 quantize_quark.py \
        --model_dir $MODEL_DIR \
        --output_dir Meta-Llama-3.1-405B-Instruct-FP8-KV \
        --quant_scheme w_fp8_a_fp8 \
        --kv_cache_dtype fp8 \
        --num_calib_data 128 \
        --model_export quark_safetensors \
        --no_weight_matrix_merge

#### If model size is too large for single GPU, please use multi GPU instead.
    python3 quantize_quark.py \
        --model_dir $MODEL_DIR \
        --output_dir Meta-Llama-3.1-405B-Instruct-FP8-KV \
        --quant_scheme w_fp8_a_fp8 \
        --kv_cache_dtype fp8 \
        --num_calib_data 128 \
        --model_export quark_safetensors \
        --no_weight_matrix_merge \
        --multi_gpu


### Launch AMD vLLM Docker

Download and launch the docker,

    docker run -it --rm --ipc=host --network=host --group-add render \
    --privileged --security-opt seccomp=unconfined \
    --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
    --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    -v /data/llama-3.1:/data/llm \
    docker pull powderluv/vllm_dev_channel


### Benchmark with AMD vLLM Docker

There are some system settings to be configured for optimum performance on MI300X. 

#### NUMA balancing setting

To optimize performance, disable automatic NUMA balancing. Otherwise, the GPU might hang until the periodic balancing is finalized. For further details, refer to the AMD Instinct MI300X system optimization guide.

Disable automatic NUMA balancing

    sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

Check if NUMA balancing is disabled (returns 0 if disabled)

    cat /proc/sys/kernel/numa_balancing
    0

#### LLM performance settings

Some environment variables enhance the performance of the vLLM kernels and PyTorch's tunableOp on the MI300X accelerator. The settings below are already preconfigured in the Docker image. See the AMD Instinct MI300X workload optimization guide for more information.

##### vLLM performance settings

    export PYTORCH_TUNABLEOP_ENABLED=1
    export PYTORCH_TUNABLEOP_ROCBLAS_ENABLED=0
    export PYTORCH_TUNABLEOP_TUNING=1
    export HIP_FORCE_DEV_KERNARG=1
    export VLLM_USE_ROCM_CUSTOM_PAGED_ATTN=1
    export VLLM_USE_TRITON_FLASH_ATTN=0
    export VLLM_USE_TRITON_FLASH_ATTN=0
    export VLLM_INSTALL_PUNICA_KERNELS=1
    export TOKENIZERS_PARALLELISM=false
    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export NCCL_MIN_NCHANNELS=112
    export VLLM_FP8_PADDING=1
    export VLLM_FP8_ACT_PADDING=1
    export VLLM_FP8_WEIGHT_PADDING=1
    export VLLM_FP8_REDUCE_CONV=1
    export VLLM_SCHED_PREFILL_KVC_FREEPCT=31.0

You can set both PYTORCH_TUNABLEOP_ENABLED and PYTORCH_TUNABLEOP_TUNING to 1 to performance GEMM tuning for the 1st benchmark run. 
It will take some time to complete the tuning during the benchmark. After tuning, it will generate several csv files as the performance lookup database. For the subsequent benchmark runs, you can keep PYTORCH_TUNABLEOP_ENABLED as 1 and set 
PYTORCH_TUNABLEOP_TUNING to 0 to use the selected kernels. 


##### Latency Benchmark

Benchmark Meta-Llama-3.1-405B FP8 with input 128 tokens, output 128 tokens, batch size 32 and tensor parallelism 8 as an example,

    python /app/vllm/benchmarks/benchmark_latency.py \
    --model /data/llm/Meta-Llama-3.1-405B-Instruct-FP8-KV \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --dtype half \
    --gpu-memory-utilization 0.99 \
    --distributed-executor-backend mp \
    --tensor-parallel-size 8 \
    --batch size 32 \
    --input-len 128 \
    --output-len 128

If you want to run Meta-Llama-3.1-405B FP16, please run

    python /app/vllm/benchmarks/benchmark_latency.py \
    --model /data/llm/Meta-Llama-3.1-405B-Instruct \
    --dtype float16 \
    --gpu-memory-utilization 0.99 \
    --distributed-executor-backend mp \
    --tensor-parallel-size 8 \
    --batch size 32 \
    --input-len 128 \
    --output-len 128

You can change various input-len, output-len, batch size and run the benchmark as well. When output-len is 1, it measures prefill latency (TTFT). 
Decoding latency (TPOT) can be calculated based on the measured latency. 

For more information about the parameters, please run

    /app/vllm/benchmarks/benchmark_latency.py -h

##### Throughput Benchmark

Benchmark Meta-Llama-3.1-405B FP8 with input 128 tokens, output 128 tokens and tensor parallelism 8 as an example,

    python /app/vllm/benchmarks/benchmark_throughput.py \
    --model /data/llm/Meta-Llama-3.1-405B-Instruct-FP8-KV \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --dtype half \
    --gpu-memory-utilization 0.99 \
    --num-prompts 2000 \
    --distributed-executor-backend mp \
    --num-scheduler-steps 10 \
    --tensor-parallel-size 8 \
    --input-len 128 \
    --output-len 128

If you want to run Meta-Llama-3.1-405B FP16, please run

    python /app/vllm/benchmarks/benchmark_throughput.py \
    --model /data/llm/Meta-Llama-3.1-405B-Instruct \
    --dtype float16 \
    --gpu-memory-utilization 0.99 \
    --num-prompts 2000 \
    --distributed-executor-backend mp \
    --num-scheduler-steps 10 \
    --tensor-parallel-size 8 \
    --input-len 128 \
    --output-len 128

You can change various input-len, output-len, num-prompts and run the benchmark as well.
Please note num-scheduler-step is a new feature added in vLLM 0.6.0. It can improve the decoding latency and throughput, however, it may increase the prefill latency.

For more information about the parameters, please run

    /app/vllm/benchmarks/benchmark_throughput.py -h

Tensor parallism (TP) parameters depends on the model size. For Llama 3.1 70B and 8B model, TP 1 can be used as well for MI300X. In general, TP 8 and 1 is recommended to achieve the optimum performance. 

##### Online Server Benchmark
 
Make the following changes if required
 
/app/vllm/benchmarks/backend_request_func.py
 
line 242 + "ignore_eos": True,
 
/app/vllm/benchmarks/benchmark_serving.py
line 245 -         interval = np.random.exponential(1.0 / request_rate)
line 245 +         ## interval = np.random.exponential(1.0 / request_rate)
line 246 +         interval = 1.0 / request_rate
 
Benchmark Meta-Llama-3.1-70B with input 4096 tokens, output 512 tokens and tensor parallelism 8 as an example,
 
    vllm serve /data/llm/Meta-Llama-3.1-70B-Instruct-FP8-KV \
    --swap-space 16 \
    --disable-log-requests \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --dtype float16 \
    --max-model-len 8192 \
    --tensor-parallel-size 8
 
Change port (for example --port 8005) if port=8000 is currently being used by other processes.
 
run client in a separate terminal. Use port_id from previous step else port-id=8000.
 
    python /app/vllm/benchmarks/benchmark_serving.py \
    --port 8000 \
    --model /data/llm/Meta-Llama-3.1-70B-Instruct-FP8-KV \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 512 \
    --request-rate 1 \
    --num-prompts 500 \
    --percentile-metrics ttft,tpot,itl,e2el
 
Once all prompts are processed, terminate the server gracefully (ctrl+c).
 
##### CPX mode
 
Currently only CPX-NPS1 mode is supported. So ONLY tp=1 is supported in CPX mode.
But multiple instances can be started simultaneously (if needed) in CPX-NPS1 mode.
 
Set GPUs in CPX mode
 
    rocm-smi --setcomputepartition cpx
 
Example of running Llama3.1-8B on 1 CPX-NPS1 GPU with input 4096 and output 512. As mentioned above, tp=1.

    HIP_VISIBLE_DEVICES=0 \
    python3 /app/vllm/benchmarks/benchmark_throughput.py \
    --max-model-len 4608 \
    --num-scheduler-steps 10 \
    --num-prompts 100 \
    --model /data/llm/Meta-Llama-3.1-70B-Instruct-FP8-KV \
    --input-len 4096 \
    --output-len 512 \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --output-json <path/to/output.json> \
    --quantization fp8 \
    --gpu-memory-utilization 0.99
 
Set GPU to SPX mode.

    rocm-smi --setcomputepartition spx

### MMLU_PRO_Biology Accuracy Eval
 
### fp16
vllm (pretrained=models--meta-llama--Meta-Llama-3.1-405B-Instruct/snapshots/069992c75aed59df00ec06c17177e76c63296a26,dtype=float16,tensor_parallel_size=8), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 64
 
| Tasks |Version|    Filter    |n-shot|  Metric   |   |Value |   |Stderr|
|-------|------:|--------------|-----:|-----------|---|-----:|---|-----:|
|biology|      0|custom-extract|     5|exact_match|↑  |0.8466|±  |0.0135|
 
### fp8
vllm (pretrained=models--meta-llama--Meta-Llama-3.1-405B-Instruct/snapshots/069992c75aed59df00ec06c17177e76c63296a26,dtype=float16,quantization=fp8,quantized_weights_path=/llama.safetensors,tensor_parallel_size=8), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 32
 
| Tasks |Version|    Filter    |n-shot|  Metric   |   |Value|   |Stderr|
|-------|------:|--------------|-----:|-----------|---|----:|---|-----:|
|biology|      0|custom-extract|     5|exact_match|↑  |0.848|±  |0.0134|


## Performance

### LLaMA2/3 *MLPerf* 70B

Please refer to the MLPerf instructions for recreating the MLPerf numbers.

## Version

### Release Notes
20240906a: Legacy quantization formats required `--quantization fp8_rocm` as a flag instead of `--quantization fp8`

Updated:

torchvision: https://github.com/ROCm/vision/commit/fab848869c0f88802297bad43c0ad80f33ecabb4/

vLLM: https://github.com/ROCm/vllm/commit/6f35c77845068dcc90c222fdfd1b56c3db149ad1


### Docker Manifest
ROCm6.2 GA
tip-of-tree (hipBLASLT, rocBLAS, Flash-attention, CK, Triton, MIOpen, RCCL, Apex)
Python 3.9
Ubuntu 22
PyTorch 2.4 Release

 

| Component | Commit/Link |
| -- | -- |
| Base Docker | rocm/pytorch:rocm6.2_ubuntu22.04_py3.9_pytorch_release_2.2.1 |
| PyTorch Commit |[https://github.com/ROCm/pytorch/commit/c4d355377af3b1c48b37e05f81293c65f25689aa](https://github.com/ROCm/pytorch/commit/c4d355377af3b1c48b37e05f81293c65f25689aa "https://github.com/rocm/pytorch/commit/c4d355377af3b1c48b37e05f81293c65f25689aa")
| PyTorch wheels | [http://rocm-ci.amd.com/view/Release-6.2/job/pytorch-pipeline-manylinux-wheel-builder_rel-6.2/315/execution/node/169/ws/final_pkgs/](http://rocm-ci.amd.com/view/Release-6.2/job/pytorch-pipeline-manylinux-wheel-builder_rel-6.2/315/execution/node/169/ws/final_pkgs/ "http://rocm-ci.amd.com/view/release-6.2/job/pytorch-pipeline-manylinux-wheel-builder_rel-6.2/315/execution/node/169/ws/final_pkgs/")
| apex | [https://github.com/ROCm/apex/commit/ac13eaffb8a3dd8d574979263aa24bce2a5966a4](https://github.com/ROCm/apex/commit/ac13eaffb8a3dd8d574979263aa24bce2a5966a4 "https://github.com/rocm/apex/commit/ac13eaffb8a3dd8d574979263aa24bce2a5966a4")
| torchvision |[https://github.com/pytorch/vision/commit/48b1edffdc6f34b766e2b4bbf23b78bd4df94181](https://github.com/pytorch/vision/commit/48b1edffdc6f34b766e2b4bbf23b78bd4df94181 "https://github.com/pytorch/vision/commit/48b1edffdc6f34b766e2b4bbf23b78bd4df94181")
| torchdata |[https://github.com/pytorch/data/commit/5e6f7b7dc5f8c8409a6a140f520a045da8700451](https://github.com/pytorch/data/commit/5e6f7b7dc5f8c8409a6a140f520a045da8700451 "https://github.com/pytorch/data/commit/5e6f7b7dc5f8c8409a6a140f520a045da8700451")
| hipblaslt |[https://github.com/ROCm/hipBLASLt/commit/3f6a167cc3e2aa3f6be7a48b53c67e482628c910](https://github.com/ROCm/hipBLASLt/commit/3f6a167cc3e2aa3f6be7a48b53c67e482628c910 "https://github.com/rocm/hipblaslt/commit/3f6a167cc3e2aa3f6be7a48b53c67e482628c910")
|RocBLAS |[https://github.com/ROCm/rocBLAS/commit/9b1bd5ab663b2cd9669e90eda1a2bc9382a8c72d](https://github.com/ROCm/rocBLAS/commit/9b1bd5ab663b2cd9669e90eda1a2bc9382a8c72d "https://github.com/rocm/rocblas/commit/9b1bd5ab663b2cd9669e90eda1a2bc9382a8c72d")
| CK |[https://github.com/ROCm/composable_kernel/commit/c8b6b64240e840a7decf76dfaa13c37da5294c4a](https://github.com/ROCm/composable_kernel/commit/c8b6b64240e840a7decf76dfaa13c37da5294c4a "https://github.com/rocm/composable_kernel/commit/c8b6b64240e840a7decf76dfaa13c37da5294c4a")
| RCCL |[https://github.com/ROCm/rccl/commit/d3171b51b7a5808bd5b984ddbed3a43ffabdc2fe](https://github.com/ROCm/rccl/commit/d3171b51b7a5808bd5b984ddbed3a43ffabdc2fe "https://github.com/rocm/rccl/commit/d3171b51b7a5808bd5b984ddbed3a43ffabdc2fe")|
MIOpen |[https://github.com/ROCm/MIOpen/commit/4be2a0339f5aeddec46f938c99e02e5ee885b99f](https://github.com/ROCm/MIOpen/commit/4be2a0339f5aeddec46f938c99e02e5ee885b99f "https://github.com/rocm/miopen/commit/4be2a0339f5aeddec46f938c99e02e5ee885b99f")
| triton |[https://github.com/triton-lang/triton/commit/0e9267202532ed1709dcc12c636220cf239dc377](https://github.com/triton-lang/triton/commit/0e9267202532ed1709dcc12c636220cf239dc377 "https://github.com/triton-lang/triton/commit/0e9267202532ed1709dcc12c636220cf239dc377")
| Flash-attention |[https://github.com/ROCm/flash-attention/commit/28e7f4ddbd6924c0533bc0cb151f9485e94846a4](https://github.com/ROCm/flash-attention/commit/28e7f4ddbd6924c0533bc0cb151f9485e94846a4 "https://github.com/rocm/flash-attention/commit/28e7f4ddbd6924c0533bc0cb151f9485e94846a4")
| vllm |[https://github.com/ROCm/vllm/commit/7c5fd50478803e12a0a4ba6050dc4ed63188a651](https://github.com/ROCm/vllm/commit/7c5fd50478803e12a0a4ba6050dc4ed63188a651 "https://github.com/rocm/vllm/commit/7c5fd50478803e12a0a4ba6050dc4ed63188a651")
|rccl-tests |[https://github.com/ROCm/rccl-tests/commit/52aee698fa255c1eb081d1f33368cca1a82b1b67](https://github.com/ROCm/rccl-tests/commit/52aee698fa255c1eb081d1f33368cca1a82b1b67 "https://github.com/rocm/rccl-tests/commit/52aee698fa255c1eb081d1f33368cca1a82b1b67") |
