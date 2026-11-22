import glob

for f in glob.glob('web/*.js') + glob.glob('web/*.html'):
    with open(f, 'r', encoding='utf-8') as fp:
        for lno, line in enumerate(fp, 1):
            if '"Z"' in line or "'Z'" in line:
                print(f"{f}:{lno}: {line.strip()[:100]}")
