
## compile
- cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="ivcore11" -DLLAMA_CURL=OFF
- cmake --build build --config Release -v

## run
- cd build/bin
- ./llama-cli --model /share/fshare/common/models/Qwen/Qwen3-GGUF/Qwen3-32B-Q2_K.gguf --cache-type-k q8_0 --threads 1 --prompt '<｜User｜>你是谁呢？<｜Assistant｜>'  -no-cnv --n-gpu-layers 100 --temp 0.01

