"""Use the verified private transport in a fresh full64-only namespace.

No training source, payload, deadline, or previous server is changed. The old
source_bridge module is preserved byte for byte as the mechanism reference.
"""
import source_bridge

source_bridge.MA = '69b991ee-183f-4add-af1a-579af92479c9'
source_bridge.REL = 'patient-response-code/full64-a800-source-bridge-v1'
source_bridge.ROOT = '/home/zhangyuheng/projects/CHIP2026_PatientPheX/' + source_bridge.REL

if __name__ == '__main__':
    source_bridge.main()
