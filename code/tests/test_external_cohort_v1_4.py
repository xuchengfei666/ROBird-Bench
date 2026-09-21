from robird.external_cohort_v1_3 import SHARED_HASH
CLUSTER={690953732,690959646,690959649,690960036,690959649};OBS={377824923,377824924}
def accept(a,b,source): return source=='within_new_cohort' and int(a['photo_id']) in CLUSTER and int(b['photo_id']) in CLUSTER and {int(a['observation_id']),int(b['observation_id'])}==OBS and a['sha256']==b['sha256']
def test_reviewed_cluster_only():
    a={'photo_id':690959646,'observation_id':377824923,'sha256':SHARED_HASH};b={'photo_id':690960036,'observation_id':377824924,'sha256':SHARED_HASH}
    assert accept(a,b,'within_new_cohort');assert not accept(a,b,'history');assert not accept(dict(a,photo_id=123),b,'within_new_cohort')
