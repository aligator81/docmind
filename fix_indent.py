import re

with open('backend/app/services/document_processor.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix the except line
for i in range(len(lines)):
    if re.match(r'^        except Exception as e:', lines[i]):
        lines[i] = re.sub(r'^        ', r'    ', lines[i])

# Fix the try block and except block
for i in range(len(lines)):
    if re.match(r'^            ', lines[i]):
        lines[i] = re.sub(r'^            ', r'        ', lines[i])

# Fix the function body for wrapper
for i in range(83, 113):
    if lines[i].startswith('        '):
        lines[i] = '            ' + lines[i].lstrip()

# Fix the with block
for i in range(127, 131):
    if lines[i].startswith('        '):
        lines[i] = '            ' + lines[i].lstrip()

with open('backend/app/services/document_processor.py', 'w', encoding='utf-8') as f:
    f.write(''.join(lines))

print('File fixed.')