import json
import sys

notebook_path = 'gemini_pro_1127_2.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
except Exception as e:
    print(f"Error reading notebook: {e}")
    sys.exit(1)

code_cells = [cell['source'] for cell in nb['cells'] if cell['cell_type'] == 'code']
full_code = ""

for cell_source in code_cells:
    # cell_source is a list of strings
    cell_code = "".join(cell_source)
    full_code += cell_code + "\n\n"

print("Extracted code from notebook. Executing...")

# Print code with line numbers for debugging
lines = full_code.split('\n')
start_line = max(0, 50)
end_line = min(len(lines), 70)
for i in range(start_line, end_line):
    print(f"{i+1}: {lines[i]}")

try:
    exec(full_code)
    print("\nSUCCESS: All code cells executed without error.")
except Exception as e:
    print(f"\nFAILURE: Error executing notebook code: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
