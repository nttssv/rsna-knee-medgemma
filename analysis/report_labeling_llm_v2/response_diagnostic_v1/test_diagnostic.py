"""Synthetic guard tests; no report data, model weights or inference."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from inspect_responses import strict_suffix, hypothetical_response, token_structure


@pytest.mark.parametrize('raw', [
    'prose<unused94>thought<unused95>{}',
    ' <unused94>thought<unused95>{}',
    '<unused94>thought{}',
    '<unused95><unused94>thought{}',
    '<unused94>outer<unused94>inner<unused95>{}',
    '<unused94>thought<unused95>{}<unused95>',
    '<unused94>thought<unused95>{}<unused94>more',
    '<unused94><unused95>{}',
    '<unused94>thought<unused95>  ',
])
def test_malformed_envelope_is_not_rescued(raw):
    with pytest.raises(ValueError):
        strict_suffix(raw)


def test_suffix_bytes_unchanged():
    suffix = '\r\n```json\n{"text":"ñ😀"}\n```  '
    assert strict_suffix('<unused94>thought\nnotes<unused95>'+suffix) == suffix


def test_truncated_generation_never_rescued():
    response = hypothetical_response('<unused94>thought<unused95>{}', '', 'generation_truncated')
    assert len(response['rows']) == 12
    assert all(r['status'] == 'generation_truncated' and r['label'] is None for r in response['rows'])


@pytest.mark.parametrize('ids,eos_count,answer', [([100,5,6,101,7,8,106],1,2),([100,5,6,101,7,8],0,2)])
def test_token_partition_accounts_for_every_token(ids,eos_count,answer):
    result = token_structure(ids,100,101,[1,106])
    assert result['thought_tokens'] == 2
    assert result['answer_tokens'] == answer
    assert result['eos_tokens'] == eos_count
    assert sum(result[k] for k in ('thought_tokens','answer_tokens','eos_tokens','delimiter_tokens')) == len(ids)


@pytest.mark.parametrize('ids', [[5,100,6,101,106],[100,5,101,6,101,106],[100,5,106,101,6],[100,101,5,106]])
def test_anomalous_token_structure_rejected(ids):
    with pytest.raises(ValueError):
        token_structure(ids,100,101,[1,106])
