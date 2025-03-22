Setup:
- Clone submodules to get HarmBench
- Download https://huggingface.co/meta-llama/Llama-Guard-3-8B
- pip install pandas torch transformers accelerate csvkit

To filter results:
    csvcut -c normal_safe,base64_safe,rot13_safe,hex_safe llamaguard3_harmbench_results.csv > safe.csv
    csvgrep -c 'base64_safe' -m 'False' safe.csv
