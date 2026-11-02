import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.audio import AudioClient, generated_call, transcribe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/tmp/anchor-audio-call.wav')
    parser.add_argument('--fail-chunk', type=int)
    args = parser.parse_args()
    raw = generated_call()
    Path(args.output).write_bytes(raw)
    client = AudioClient(fail_chunks=() if args.fail_chunk is None else {args.fail_chunk})
    result = transcribe(raw, client=client)
    print(json.dumps({'recording': args.output, 'ledger': result['ledger'],
                      'segments': result['segments'], 'gaps': result['gaps'],
                      'partial': result['partial'],
                      'silence_rejections': result['silence_rejections']}, indent=2))
    print('Offline smoke: speech recognition, acoustic diarization, real provider transport, '
          'and production infrastructure are NOT verified.')
    print('Anchor audio preprocessing: bounded chunks, redacted timestamped evidence, '
          'unknown business roles, and explicit missing intervals.')


if __name__ == '__main__':
    main()
