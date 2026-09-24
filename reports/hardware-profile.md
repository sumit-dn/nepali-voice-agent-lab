# Hardware profile

## This machine

```
{
  "os": "Linux-7.0.0-31-generic-x86_64-with-glibc2.43",
  "python": "3.11.15",
  "cpu": "Intel(R) Core(TM) i5-7400 CPU @ 3.00GHz",
  "cpu_cores_physical": 4,
  "cpu_threads": 4,
  "ram_total_gb": 16.1,
  "ram_available_gb": 7.1,
  "disk_free_gb": 12.7,
  "gpus": [],
  "cuda": null,
  "packages": {
    "torch": "2.14.0+cpu",
    "torchaudio": "2.11.0+cpu",
    "transformers": "5.17.0",
    "huggingface_hub": "1.32.0",
    "onnxruntime": "1.30.0",
    "piper-tts": "1.8.0",
    "silero-vad": "6.2.1"
  }
}
```

## Per-model load time and peak resources (latest run of each kind)

| Kind | Model | Variant | Load (s) | Peak proc RAM (MB) | Peak sys RAM used (MB) | Proc CPU avg % | Peak VRAM (MB) | GPU util % | Documented requirement |
|---|---|---|---|---|---|---|---|---|---|
| asr | indicconformer-600m-rnnt | original | 6.596 | 5333 | 13158 | 342.800 | – | – | as indicconformer-600m |
| asr | indicconformer-600m | original | 8.109 | 3095 | 11452 | 307.200 | – | – | Not stated on card. ONNX Runtime; CPU should work (untested here), GPU via onnxruntime-gpu |
| llm | llama3.1-8b | default | 0.024 | 308 | 12043 | 0.200 | – | – | Q4 ~6-7 GB RAM; CPU feasible but slow |
| tts | piper-ne-google-x-low | default | – | – | – | – | – | – | CPU real-time (onnxruntime) |
| tts | xtts-ne-oshara-e20 | default | 172.557 | 2968 | 15772 | 12.400 | – | – | Separate env (make install-xtts). CPU works but slow (RTF ~3.7, load ~32 s on i5-7400); GPU recommended |

For Ollama/vLLM models the weights live in the server process: use the system RAM column and the server's own reporting (`ollama ps`).
