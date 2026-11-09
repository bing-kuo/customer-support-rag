"""Key-only diagnostic redaction regression."""
import json
from scripts.cloud_setup import diagnose
from tests.test_key_only import fixture_policy


def test_hostile_private_values_never_echo(tmp_path):
    p=tmp_path/'env';p.write_text('OPENAI_API_KEY=hostile private secret\n')
    text=json.dumps(diagnose(p,fixture_policy(tmp_path)))
    assert 'hostile private secret' not in text and 'invalid_format' in text


def test_missing_only_key_not_fifteen_policy_fields(tmp_path):
    p=tmp_path/'env';p.write_text('')
    assert diagnose(p,fixture_policy(tmp_path))=={'ready':False,'issues':[{'category':'missing_or_blank','field':'OPENAI_API_KEY'}]}


def test_unavailable_and_symlink_safely_rejected(tmp_path):
    p=tmp_path/'missing';assert not diagnose(p)['ready']
    target=tmp_path/'env';target.write_text('OPENAI_API_KEY=synthetic-mock');p.symlink_to(target)
    assert not diagnose(p)['ready']
