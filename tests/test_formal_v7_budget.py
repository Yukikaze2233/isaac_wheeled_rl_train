from datetime import timedelta
import json
import pytest
from support_training_budget import make_review, NOW
from wheeled_tasks.chassis.training_budget import BudgetOwner, BudgetLimit, digest_file


def test_elapsed_formal_budget_retains_failed_update_when_attempt_is_appended(tmp_path):
    path, review = make_review(tmp_path, pilot_cap=0, roles={'main':'main'})
    review['gpu_meter_owner']={'kind':'campaign_elapsed_wall_v1','id':'formal','origin_utc':NOW.isoformat()}
    path.write_text(json.dumps(review))
    identity={k:v for k,v in review['attempts']['main'].items() if k!='role'}
    owner=BudgetOwner.authenticate(path,digest_file(path),review['ledger_path'],'main',identity,now=lambda:NOW)
    token=owner.reserve_update(phase='actor')
    owner.fail_update(token,'interrupted')
    owner.close()
    review['attempts']['repair']={**review['attempts']['main'],'training_source_sha256':'c'*64}
    path.write_text(json.dumps(review))
    identity={k:v for k,v in review['attempts']['repair'].items() if k!='role'}
    owner=BudgetOwner.authenticate(path,digest_file(path),review['ledger_path'],'repair',identity,now=lambda:NOW+timedelta(seconds=20))
    assert owner.summary()['paid_updates_global']==1
    assert 'not_gpu_utilization' in owner.summary()['gpu_meter_scope']
    owner.now=lambda:NOW+timedelta(seconds=100)
    with pytest.raises(BudgetLimit):owner.check_runtime()
    owner.close()


def test_repair_resume_requires_registered_exact_parent_without_refund(tmp_path):
    from support_training_budget import TestMeter, publish
    path, review = make_review(tmp_path, roles={'main':'main'})
    identity={k:v for k,v in review['attempts']['main'].items() if k!='role'}
    owner=BudgetOwner.authenticate(path,digest_file(path),review['ledger_path'],'main',identity,
        gpu_meter=TestMeter(),allow_test_meter=True,now=lambda:NOW)
    token=owner.reserve_update(phase='actor');owner.complete_update(token,actor_optimized=True)
    checkpoint=tmp_path/'registered.pt';publish(owner,checkpoint);owner.close()
    review['attempts']['repair']={**review['attempts']['main'],
        'training_source_sha256':'c'*64,'parent_checkpoint_sha256':digest_file(checkpoint)}
    path.write_text(json.dumps(review));identity={k:v for k,v in review['attempts']['repair'].items() if k!='role'}
    owner=BudgetOwner.authenticate(path,digest_file(path),review['ledger_path'],'repair',identity,
        resume_checkpoint=checkpoint,gpu_meter=TestMeter(),allow_test_meter=True,now=lambda:NOW)
    assert owner.verify_resume(checkpoint)['attempt_id']=='main'
    assert owner.summary()['paid_updates_global']==1
    owner.close()
