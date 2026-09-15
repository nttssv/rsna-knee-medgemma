"""No network, Transformers or weights required by these unit tests."""
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import v2_tokenizer_preflight as preflight


def test_inventory_requires_vocabulary(tmp_path):
    (tmp_path/'tokenizer_config.json').write_text('{}')
    with pytest.raises(ValueError,match='vocabulary'):
        preflight.inventory(tmp_path)


def test_inventory_never_reads_weights(tmp_path):
    (tmp_path/'tokenizer.json').write_text('{}')
    (tmp_path/'model.safetensors').symlink_to(tmp_path/'missing')
    rows=preflight.inventory(tmp_path)
    assert [r['file'] for r in rows]==['tokenizer.json']


@pytest.mark.parametrize('algorithm',['sha1','sha256'])
def test_hub_blob_hash_detects_tampering(tmp_path,algorithm):
    data=b'{"synthetic":true}'
    blob_id=(hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
             if algorithm=='sha1' else hashlib.sha256(data).hexdigest())
    blob=tmp_path/blob_id;blob.write_bytes(data)
    (tmp_path/'tokenizer.json').symlink_to(blob)
    assert preflight.inventory(tmp_path,True)[0]['hub_blob']==blob_id
    blob.write_bytes(b'changed')
    with pytest.raises(ValueError,match='content-addressed'):
        preflight.inventory(tmp_path,True)


def test_regular_export_cannot_claim_hub_blob(tmp_path):
    (tmp_path/'tokenizer.json').write_text('{}')
    with pytest.raises(ValueError,match='content-addressed'):
        preflight.inventory(tmp_path,True)


def test_missing_context_remains_unknown():
    result=preflight.summarize([SimpleNamespace(input_tokens=100)],'medgemma',None)
    assert result['config_context_budget_passed'] is None
    assert result['input_budget_passed'] and result['model_calls']==0
    assert result['loaded_weight_context_verified'] is False


@pytest.mark.parametrize('tokens,context',[(8193,50000),(7000,8000)])
def test_input_and_output_reserve_both_enforced(tokens,context):
    with pytest.raises(ValueError,match='budget'):
        preflight.summarize([SimpleNamespace(input_tokens=tokens)],'qwen',context)
