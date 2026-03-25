#!/usr/bin/env python3
import os, subprocess, json, requests

token = os.environ.get('GITHUB_TOKEN')
if not token:
    try:
        result = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True, check=True)
        token = result.stdout.strip()
    except:
        pass

print('Token found:', bool(token))

if token:
    url = 'https://models.inference.ai.azure.com/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    payload = {
        'model': 'gpt-4o-mini',
        'messages': [{'role': 'user', 'content': 'Translate to English: "Vilka brott rapporterades?"'}],
        'temperature': 0.1,
        'max_tokens': 100
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        print(f'HTTP Status: {r.status_code}')
        if r.ok:
            text = r.json()['choices'][0]['message']['content']
            print(f'Translation: {text}')
        else:
            print(f'Error: {r.text[:300]}')
    except Exception as e:
        print(f'Request failed: {e}')
else:
    print('No token available')
