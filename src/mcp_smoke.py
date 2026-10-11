#!/usr/bin/env python3
"""Agent-side smoke test: start the MCP server over stdio, list tools, make one valid and one refused call."""
import asyncio, json, sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
ROOT = Path(__file__).resolve().parents[1]


async def main():
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / 'src/mcp_server.py')])
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        tools = await s.list_tools(); print('tools:', [t.name for t in tools.tools])
        out = {}
        for args in [{'claim_id': 'Ej3y_single-trace', 'as_of': '2018-12-31', 'stage': 'publication'},
                     {'claim_id': 'Ej3y_single-trace', 'as_of': '2018-12-31', 'stage': 'launch'},
                     {'claim_id': 'Ej3y_single-trace', 'as_of': '2023-01-01', 'stage': 'publication'}]:
            res = await s.call_tool('verify_claim', args)
            body = json.loads(res.content[0].text)
            out[f"{args['stage']}@{args['as_of']}"] = body
            print(json.dumps(args), '->', json.dumps(body)[:400])
        (ROOT / 'results/mcp_smoke.json').write_text(json.dumps(out, indent=2))

asyncio.run(main())
