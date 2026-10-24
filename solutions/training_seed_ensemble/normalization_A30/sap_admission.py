"""Launch TTL is separate from the Sap driver runtime180+30 clock."""
import time
def admit(plan,stage,now=None):
 a=plan['launch_admission'];now=time.time() if now is None else now
 assert a['schema']=='training-seed-Sap-admission-v1'
 assert 0<a['launch_not_after_unix']-a['bound_at_unix']<=300
 assert a['bound_at_unix']<=now<=a['launch_not_after_unix'],stage+': launch validity expired; preserve original intent'
 return a
