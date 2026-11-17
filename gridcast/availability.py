"""External, read-only availability checks. No credentials or cloud dependencies."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def fetch(url):
    # curl supplies normal TLS verification, redirects, HTTP status and deadlines.
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / 'body'
        result = subprocess.run(['curl', '--silent', '--show-error', '--location',
            '--max-time', '25', '--connect-timeout', '10', '--output', str(output),
            '--write-out', '%{http_code}', url], capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f'Connection/TLS failure: curl {result.returncode}: {result.stderr.strip()[:200]}')
        if result.stdout.strip() != '200':
            raise ValueError(f'HTTP {result.stdout.strip()}')
        return output.read_text(encoding='utf-8')


def check(name, url, validate):
    errors = []
    for attempt in range(3):
        try:
            body = fetch(url)
            validate(body)
            return {'name':name,'url':url,'ok':True,'attempts':attempt+1,'recovered_errors':errors}
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc))
            if attempt < 2:
                time.sleep(15)
    return {'name':name,'url':url,'ok':False,'attempts':3,'errors':errors}


def contains(marker):
    def validate(body):
        if marker not in body:
            raise ValueError('Expected page content missing; possible error/challenge page')
    return validate


def fresh_health(body):
    data = json.loads(body)
    if data.get('status') != 'ok' or not 0 <= data['age_seconds'] <= 5400:
        raise ValueError('Latest saved forecast is delayed or invalid')


def archive_health(body):
    data = json.loads(body)
    age = (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(data['updated_at'])).total_seconds()
    if not data['dates'] or data['forecast_issues'] < 1 or not 0 <= age <= 6*3600:
        raise ValueError('Forecast archive is empty, invalid or over six hours old')


def main():
    base = 'https://gridcast-read-api.tp-duarte.workers.dev'
    checks = [
        ('Homepage','https://tpduarte.com/',contains('Turning analysis')),
        ('WWW homepage','https://www.tpduarte.com/',contains('Turning analysis')),
        ('Dashboard page','https://tpduarte.com/dashboard',contains('Forecast Added Value')),
        ('Homepage stylesheet','https://tpduarte.com/assets/portfolio.css',contains('body')),
        ('Latest forecast',base+'/health',fresh_health),
        ('Saved archive',base+'/v1/archive',archive_health),
    ]
    results = [check(*spec) for spec in checks]
    report = {'checked_at':dt.datetime.now(dt.timezone.utc).isoformat(),'checks':results}
    Path('availability-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary,'a',encoding='utf-8') as stream:
            stream.write('## External availability checks\n\n'+report['checked_at']+' UTC\n\n')
            for item in results:
                stream.write(f"- {'PASS' if item['ok'] else 'FAIL'}: {item['name']} ({item['attempts']} attempts)\n")
    raise SystemExit(0 if all(item['ok'] for item in results) else 1)


if __name__ == '__main__':
    main()

