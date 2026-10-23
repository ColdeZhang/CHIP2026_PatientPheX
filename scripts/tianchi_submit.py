#!/usr/bin/env python3
"""Submit one frozen day-one artifact; record acceptance, never collect scores.

Requires an already authenticated Chromium CDP session and Playwright.
The entry JSON supplies file, filename, sha256, bytes, season_id and provenance.
An existing receipt blocks retries, including ambiguous earlier attempts.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--cdp', default='http://127.0.0.1:9222')
    args = parser.parse_args()
    if args.receipt.exists():
        raise RuntimeError('Existing receipt: inspect it and platform history before any retry')
    entry = json.loads(args.entry.read_text())
    artifact = Path(entry['file'])
    assert artifact.name == entry['filename']
    assert artifact.suffix == '.jsonl' and 0 < entry['bytes'] < 100 * 1024 * 1024
    assert artifact.stat().st_size == entry['bytes']
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == entry['sha256']
    assert entry['season_id'] == 1862
    receipt = dict(entry, status='preflight', official_score_collected=False)

    def save():
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')

    from playwright.sync_api import sync_playwright
    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(args.cdp)
        page = next(p for p in browser.contexts[0].pages
                    if '/competition/entrance/532512/submission' in p.url)
        auth = page.evaluate('''async () => {
            const s = await (await fetch('/home/user/getSession', {signal:AbortSignal.timeout(15000)})).json();
            const r = await (await fetch('/v3/proxy/competition/api/race/getDetail?raceId=532512', {signal:AbortSignal.timeout(15000)})).json();
            return {login:s.data?.login, signed:r.data?.signed,
                    showSubmit:r.data?.showSubmit, seasonId:r.data?.currentSeasonId};
        }''')
        assert auth == dict(login=True, signed=True, showSubmit=True, seasonId=1862)
        assert set(page.locator('th:visible').all_inner_texts()) == {'文件名称', '提交时间', '状态'}
        before = int(re.search(r'剩余提交次数\s*(\d+)\s*次', page.inner_text('body')).group(1))
        assert before > 0
        row = page.locator('tr').filter(has_text=entry['filename'])
        assert row.count() == 0, 'Filename already in history; do not upload again'
        receipt.update(status='upload_attempt_started', attempt_started_utc=now(), quota_before=before)
        save()  # Durable intent before the only mutation; ambiguity must never auto-retry.
        page.locator('input[type=file]').set_input_files(str(artifact))
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            page.wait_for_timeout(1000)
            match = re.search(r'剩余提交次数\s*(\d+)\s*次', page.inner_text('body'))
            after = int(match.group(1)) if match else before
            if row.count() and after == before - 1:
                assert row.locator('td').count() == 3
                cells = row.locator('td').all_inner_texts()
                # Current table has filename/time/status only; never read a score table.
                receipt.update(status='accepted', quota_after=after, observed_at_utc=now(),
                               platform_filename=cells[0], platform_time=cells[1],
                               platform_status=cells[2])
                save()
                print(json.dumps({k: receipt[k] for k in
                                  ('status', 'filename', 'quota_before', 'quota_after',
                                   'platform_time', 'platform_status')}, ensure_ascii=False))
                return
        receipt.update(status='acceptance_requires_inspection', observed_at_utc=now())
        save()
        raise RuntimeError('Acceptance ambiguous; inspect history before retry')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Browser exception messages can contain authenticated request details.
        print(json.dumps({'error_type': type(error).__name__,
                          'action': 'inspect receipt and platform history; do not blindly retry'}))
        raise SystemExit(2)
