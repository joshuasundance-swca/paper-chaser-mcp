import ast, sys
tree = ast.parse(open('paper_chaser_mcp/dispatch/_core.py', encoding='utf-8').read())
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        kind = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        end = node.end_lineno or node.lineno
        print(f"{node.lineno:5d} {end:5d} {end - node.lineno + 1:5d} {kind}{node.name}")
