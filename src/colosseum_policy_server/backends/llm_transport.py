"""Per-session Bearer authentication for Anthropic-compatible gateways."""
import httpx


class BearerTransport(httpx.BaseTransport):
    def __init__(self, api_key, inner=None):
        self._api_key = api_key
        self._inner = inner if inner is not None else httpx.HTTPTransport()

    def handle_request(self, request):
        request.headers.pop('x-api-key', None)
        request.headers['Authorization'] = 'Bearer ' + self._api_key
        return self._inner.handle_request(request)

    def close(self):
        self._api_key = ''
        self._inner.close()
