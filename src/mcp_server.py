#!/usr/bin/env python3
"""MCP server exposing the contract-enforcing verifier as an agent tool (stdio transport)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcp.server.mcpserver import MCPServer
from verifier import ReliabilityVerifier, ContractViolation

mcp = MCPServer('scireligraph-verifier')
_v = None


@mcp.tool()
def verify_claim(claim_id: str, as_of: str, stage: str = 'publication') -> dict:
    """Probability that a SCORE claim replicates, using only evidence dated <= as_of.

    stage='publication' uses original-study metadata + bibliographic evidence graph;
    stage='launch' adds the planned replication design. Calls on/after the replication launch are refused.
    The result includes base_rate, cross-validated skill, a selective_ok flag and provenance; treat
    p_replicate as a weak signal unless selective_ok is true.
    """
    global _v
    _v = _v or ReliabilityVerifier()
    try:
        return _v.verify_claim(claim_id, as_of, stage)
    except ContractViolation as e:
        return {'error': 'contract_violation', 'detail': str(e)}


if __name__ == '__main__':
    mcp.run()
