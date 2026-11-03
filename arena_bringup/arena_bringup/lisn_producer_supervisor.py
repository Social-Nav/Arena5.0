"""Short-lived owner process for one bounded VLM request.

The parent starts this process with ``start_new_session=True`` and supervises
the whole process group against an absolute monotonic deadline.  This process
reads a JSON request from stdin (config + base64 image), performs the HTTP
request and the response JSON decode, and writes a JSON result to stdout.

The credential travels only through the stdin pipe; it is never in argv, a
file, or stdout.
"""
import base64
import json
import sys

from arena_bringup.lisn_slow_loop_ros import (
    HttpVlmProducer,
    ProducerConfig,
    ProducerUnavailableError,
    safe_error_code,
)

MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_RESULT_BYTES = 4 * 1024 * 1024


def main() -> int:
    producer = None
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        result = {'ok': False, 'code': 'oversize_request'}
    else:
        try:
            payload = json.loads(raw.decode('utf-8'))
            config = ProducerConfig(**payload['config'])
            request = dict(payload['request'])
            request['image_bytes'] = base64.b64decode(request.pop('image_bytes'))
            producer = HttpVlmProducer(config=config)
            proposal = producer.produce(**request)
            result = {
                'ok': True,
                'proposal': proposal,
                'meta': dict(producer.last_response_meta),
                'request_count': int(producer.request_count),
                'auth_latched': bool(producer.auth_latched),
            }
        except ProducerUnavailableError as exc:
            result = {
                'ok': False,
                'code': exc.code,
                'status': exc.status,
                'request_count': int(getattr(producer, 'request_count', 0)),
                'auth_latched': bool(getattr(producer, 'auth_latched', False)),
            }
        except BaseException as exc:
            result = {
                'ok': False,
                'code': safe_error_code(exc),
                'request_count': int(getattr(producer, 'request_count', 0)),
                'auth_latched': bool(getattr(producer, 'auth_latched', False)),
            }
    out = json.dumps(result, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    if len(out) > MAX_RESULT_BYTES:
        out = json.dumps({'ok': False, 'code': 'oversize_result'},
                         separators=(',', ':')).encode('utf-8')
    sys.stdout.buffer.write(out)
    sys.stdout.buffer.flush()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
