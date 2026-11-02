import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.serving_artifacts import ROOT, numerical_experiment


def main():
    parser = argparse.ArgumentParser(description='Create simulated manifests, never weights.')
    parser.add_argument('--method', choices=['gptq', 'awq', 'gguf'], required=True)
    parser.add_argument('--bits', type=int, choices=[4, 8], required=True)
    args = parser.parse_args()
    directory = ROOT / 'tmp/quantization'
    directory.mkdir(parents=True, exist_ok=True)
    baseline = {'format': 'fp16', 'bits': 16, 'simulated': True,
                'base_checkpoint': 'offline_placeholder',
                'estimated_weight_bytes': 2200000000,
                'fixture_scenario': 'clean'}
    candidate = dict(baseline, format=args.method, bits=args.bits,
                     estimated_weight_bytes=1100000000 * args.bits // 8,
                     fixture_scenario='refund_regression' if args.method == 'awq' else 'clean')
    for name, manifest in [('baseline', baseline), (args.method, candidate)]:
        (directory / (name + '.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({
        'baseline': 'tmp/quantization/baseline.json',
        'candidate': 'tmp/quantization/' + args.method + '.json',
        'result': 'SIMULATED MANIFEST ONLY: no weights downloaded or quantized',
        'compatible_runtime': 'llama_cpp_cpu' if args.method == 'gguf' else 'vllm_gpu',
        'size_basis': 'simulated_payload_estimate_not_file_measurement',
        'numerical_experiment': numerical_experiment(args.bits)}, indent=2))


if __name__ == '__main__':
    main()
