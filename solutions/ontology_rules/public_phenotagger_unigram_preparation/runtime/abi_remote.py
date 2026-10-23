import json,sys,sysconfig,platform
print(json.dumps(dict(executable=sys.executable,version=sys.version,implementation=sys.implementation.name,SOABI=sysconfig.get_config_var('SOABI'),MULTIARCH=sysconfig.get_config_var('MULTIARCH'),glibc=platform.libc_ver(),machine=platform.machine())))
