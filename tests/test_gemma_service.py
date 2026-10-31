import io
import json
from urllib.error import HTTPError
from unittest.mock import Mock

import gemma_service


def test_generation_and_evaluation_request(monkeypatch):
    monkeypatch.setenv('GEMMA_MODEL_ID', 'gemma-test')
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    monkeypatch.setenv('GEMMA_MODEL_ID', 'gemma-test')
    evaluation = {'score': 88, 'feedback': 'Good work.'}
    payload = {'candidates': [{'content': {'parts': [{'text': json.dumps(evaluation)}]}}]}
    transport = Mock(return_value=io.BytesIO(json.dumps(payload).encode()))
    monkeypatch.setattr(gemma_service, 'urlopen', transport)
    assert gemma_service.generate('IELTS', 'We mitigate the damage.', ['mitigate']) == evaluation
    request = transport.call_args.args[0]
    assert request.full_url.endswith('/models/gemma-test:generateContent')
    assert request.get_header('X-goog-api-key') == 'test-key'
    body = json.loads(request.data)
    assert body['generationConfig']['temperature'] == 0.2
    assert 'mitigate' in body['contents'][0]['parts'][0]['text']
    assert 'test-key' not in request.full_url


def test_generation_schema_and_invalid_response(monkeypatch):
    monkeypatch.setenv('GEMMA_MODEL_ID', 'gemma-test')
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    word = dict.fromkeys(('word', 'pronunciation', 'definition', 'nuance', 'spectrum_comparison', 'challenge'), 'value')
    result = {'words': [word, word, word], 'mini_story': 'Story'}
    payload = {'candidates': [{'content': {'parts': [{'text': '```json\n' + json.dumps(result) + '\n```'}]}}]}
    transport = Mock(return_value=io.BytesIO(json.dumps(payload).encode()))
    monkeypatch.setattr(gemma_service, 'urlopen', transport)
    assert gemma_service.generate('IELTS') == result
    assert json.loads(transport.call_args.args[0].data)['generationConfig']['temperature'] == 0.4
    transport.return_value = io.BytesIO(b'{"candidates": []}')
    assert 'error' in gemma_service.generate('IELTS')


def test_no_key_and_upstream_auth_error(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    assert 'GEMINI_API_KEY' in gemma_service.generate('IELTS')['error']
    monkeypatch.setenv('GEMINI_API_KEY', 'secret')
    def fail(*args, **kwargs):
        raise HTTPError('https://example.invalid', 403, 'secret', {}, None)
    monkeypatch.setattr(gemma_service, 'urlopen', fail)
    assert 'secret' not in gemma_service.generate('IELTS')['error']


def test_upstream_status_is_actionable_without_exposing_response(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'secret')
    for code in (400, 404, 500):
        def fail(*args, code=code, **kwargs):
            raise HTTPError('https://example.invalid', code, 'secret detail', {}, None)
        monkeypatch.setattr(gemma_service, 'urlopen', fail)
        message = gemma_service.generate('IELTS')['error']
        assert f'HTTP {code}' in message
        assert 'secret' not in message


def test_discovers_available_gemma_after_stale_model_404(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    monkeypatch.setenv('GEMMA_MODEL_ID', 'gemma-stale')
    result = {'score': 90, 'feedback': 'Good work.'}
    transport = Mock(side_effect=[
        HTTPError('https://example.invalid', 404, 'not found', {}, None),
        io.BytesIO(json.dumps({'models': [
            {'name': 'models/gemini-example', 'supportedGenerationMethods': ['generateContent']},
            {'name': 'models/gemma-available', 'supportedGenerationMethods': ['generateContent']}
        ]}).encode()),
        io.BytesIO(json.dumps({'candidates': [{'content': {'parts': [{'text': json.dumps(result)}]}}]}).encode())
    ])
    monkeypatch.setattr(gemma_service, 'urlopen', transport)
    assert gemma_service.generate('IELTS', 'sentence', ['word']) == result
    assert '/gemma-available:' in transport.call_args.args[0].full_url


def test_no_available_gemma_is_explicit_not_mocked(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    monkeypatch.delenv('GEMMA_MODEL_ID', raising=False)
    transport = Mock(return_value=io.BytesIO(b'{"models": []}'))
    monkeypatch.setattr(gemma_service, 'urlopen', transport)
    assert 'no Gemma model' in gemma_service.generate('IELTS')['error']
    assert transport.call_count == 1


def test_repairs_malformed_output_once(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    monkeypatch.setenv('GEMMA_MODEL_ID', 'gemma-test')
    def reply(text, reason='STOP'):
        return io.BytesIO(json.dumps({'candidates': [{'finishReason': reason, 'content': {'parts': [{'text': text}]}}]}).encode())
    expected = {'score': 85, 'feedback': 'Good work.'}
    transport = Mock(side_effect=[reply('{"score":', 'MAX_TOKENS'), reply(json.dumps(expected))])
    monkeypatch.setattr(gemma_service, 'urlopen', transport)
    assert gemma_service.generate('IELTS', 'sentence', ['word']) == expected
    assert transport.call_count == 2
    transport.side_effect = [reply('prose'), reply('still prose')]
    assert 'even after recovery' in gemma_service.generate('IELTS', 'sentence', ['word'])['error']


def test_timeout_and_blocked_output_are_distinct(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    monkeypatch.setenv('GEMMA_MODEL_ID', 'gemma-test')
    transport = Mock(side_effect=TimeoutError())
    monkeypatch.setattr(gemma_service, 'urlopen', transport)
    assert 'timed out during Gemma generation' in gemma_service.generate('IELTS')['error']
    transport.side_effect = None
    transport.return_value = io.BytesIO(b'{"promptFeedback":{"blockReason":"SAFETY"}}')
    assert 'blocked' in gemma_service.generate('IELTS')['error']


def test_discovery_prefers_smaller_advertised_instruction_model(monkeypatch):
    models = [{'name': 'models/' + name, 'supportedGenerationMethods': ['generateContent']}
              for name in ('gemma-3-27b-it', 'gemma-3-4b-it', 'gemma-3-12b-it')]
    monkeypatch.setattr(gemma_service, 'urlopen', Mock(return_value=io.BytesIO(json.dumps({'models': models}).encode())))
    assert gemma_service.discover_model('fake-key') == 'gemma-3-4b-it'


def test_transient_500_recovers_without_user_retry(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    monkeypatch.setenv('GEMMA_MODEL_ID', 'gemma-test')
    monkeypatch.setattr(gemma_service.time, 'sleep', lambda _: None)
    result = {'score': 90, 'feedback': 'Good work.'}
    transport = Mock(side_effect=[HTTPError('https://example.invalid', 500, 'private', {}, None),
        io.BytesIO(json.dumps({'candidates': [{'content': {'parts': [{'text': json.dumps(result)}]}}]}).encode())])
    monkeypatch.setattr(gemma_service, 'urlopen', transport)
    assert gemma_service.generate('IELTS','sentence',['word']) == result
    assert transport.call_count == 2


def test_persistent_failure_is_bounded_and_uses_alternative(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    monkeypatch.setenv('GEMMA_MODEL_ID', 'gemma-test')
    monkeypatch.setattr(gemma_service.time, 'sleep', lambda _: None)
    monkeypatch.setattr(gemma_service,'discover_model',lambda *args,**kwargs:'gemma-alternative')
    transport=Mock(side_effect=HTTPError('https://example.invalid',503,'private',{},None))
    monkeypatch.setattr(gemma_service,'urlopen',transport)
    result=gemma_service.generate('IELTS')
    assert 'after automatic recovery' in result['error']
    assert transport.call_count==3
    assert '/gemma-alternative:' in transport.call_args.args[0].full_url


def test_gemma4_selection_uses_smallest_advertised_model(monkeypatch):
    models = [{'name': 'models/' + name, 'supportedGenerationMethods': ['generateContent']}
              for name in ('gemma-4-26b-a4b-it', 'gemma-4-31b-it', 'gemma-4-e4b-it')]
    monkeypatch.setattr(gemma_service, 'urlopen', Mock(return_value=io.BytesIO(json.dumps({'models': models}).encode())))
    assert gemma_service.discover_model('fake-key') == 'gemma-4-e4b-it'


def test_provider_logs_do_not_expose_key_or_arbitrary_error_text(monkeypatch, caplog):
    monkeypatch.setattr(gemma_service.time,'sleep',lambda _:None)
    monkeypatch.setattr(gemma_service,'discover_model',lambda *a,**kw:None)
    def fail(*args,**kwargs):
        raise HTTPError('https://example.invalid',500,'private detail',{},io.BytesIO(b'{"error":{"status":"INTERNAL","message":"private secret-key"}}'))
    monkeypatch.setattr(gemma_service,'urlopen',fail)
    import time
    try:
        gemma_service.complete('secret-key','gemma-test',{},time.monotonic()+30)
    except HTTPError:
        pass
    assert 'provider_status=INTERNAL' in caplog.text
    assert 'secret-key' not in caplog.text and 'private' not in caplog.text
