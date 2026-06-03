"""Sprint 302 — formal-invariant harness for §14 item 4.

Vision §14 item 4: "Formal verification on highest-value
contracts. Payment escrow and royalty distribution
contracts undergo formal-methods verification, not just
standard audit."

This sprint ships the SPEC LAYER + RUNTIME PROBE: pinned
invariants in code (the formal spec), a checker that
verifies them against on-chain state via an injected
backend, and a public read surface so anyone can audit
what PRSM has committed to. Actual symbolic-execution runs
(halmos, Certora) consume the same registry on a follow-on
sprint.

Highest-value target this sprint: RoyaltyDistributor v2.
Five pinned invariants cover anti-confiscation (network fee
fixed at 2%), ownership integrity, treasury immutability,
solvency (THE money invariant), and pause-state observability.
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.formal_invariants import (
    INVARIANT_REGISTRY,
    Invariant,
    InvariantChecker,
    InvariantKind,
    InvariantResult,
    InvariantSeverity,
    InvariantStatus,
    list_invariants_for_contract,
)


# ── Enums ────────────────────────────────────────────


def test_severity_values():
    assert InvariantSeverity.CRITICAL.value == "critical"
    assert InvariantSeverity.HIGH.value == "high"
    assert InvariantSeverity.MEDIUM.value == "medium"


def test_status_values():
    assert InvariantStatus.PASS.value == "pass"
    assert InvariantStatus.FAIL.value == "fail"
    assert InvariantStatus.SKIPPED.value == "skipped"


def test_kind_values():
    # Pinned check kinds — adding a new one requires updating
    # the dispatcher
    assert InvariantKind.UINT256_EQ.value == "uint256_eq"
    assert InvariantKind.UINT256_GTE.value == "uint256_gte"
    assert InvariantKind.ADDRESS_EQ.value == "address_eq"
    assert InvariantKind.BOOL_READ.value == "bool_read"
    assert (
        InvariantKind.BALANCE_GTE_CLAIMABLE.value
        == "balance_gte_claimable"
    )


# ── Pinned registry ──────────────────────────────────


def test_registry_has_royalty_distributor():
    assert "royalty_distributor" in INVARIANT_REGISTRY
    invariants = INVARIANT_REGISTRY["royalty_distributor"]
    assert len(invariants) >= 5


def test_registry_invariant_ids_unique():
    seen = set()
    for invs in INVARIANT_REGISTRY.values():
        for inv in invs:
            assert inv.id not in seen, (
                f"duplicate invariant id {inv.id}"
            )
            seen.add(inv.id)


def test_registry_has_network_fee_anti_tamper():
    invs = INVARIANT_REGISTRY["royalty_distributor"]
    ids = {i.id for i in invs}
    assert "INV-RD-1" in ids
    rd1 = next(i for i in invs if i.id == "INV-RD-1")
    assert rd1.kind == InvariantKind.UINT256_EQ
    assert rd1.severity == InvariantSeverity.CRITICAL
    assert rd1.expected == 200


def test_registry_has_solvency_invariant():
    """The single most important invariant —
    balance(this) >= totalClaimable. Failure = insolvency."""
    invs = INVARIANT_REGISTRY["royalty_distributor"]
    solvency = next(
        (i for i in invs
         if i.kind == InvariantKind.BALANCE_GTE_CLAIMABLE),
        None,
    )
    assert solvency is not None
    assert solvency.severity == InvariantSeverity.CRITICAL


def test_registry_has_owner_check():
    invs = INVARIANT_REGISTRY["royalty_distributor"]
    addr_eq = [
        i for i in invs
        if i.kind == InvariantKind.ADDRESS_EQ
    ]
    assert len(addr_eq) >= 2  # owner + networkTreasury


def test_list_invariants_for_unknown_contract_empty():
    assert list_invariants_for_contract("nonexistent") == []


# ── InvariantChecker — mock backend ──────────────────


class _MockBackend:
    """Returns scripted values per (addr, selector) tuple,
    or raises RuntimeError to simulate RPC failure."""

    def __init__(self):
        self.uint256: dict = {}
        self.address: dict = {}
        self.bool_v: dict = {}
        self.has_role: dict = {}
        self.raise_for: set = set()

    def call_uint256(
        self, addr: str, selector: str,
    ):
        key = (addr.lower(), selector.lower())
        if key in self.raise_for:
            raise RuntimeError("simulated RPC error")
        return self.uint256.get(key)

    def call_address(
        self, addr: str, selector: str,
    ):
        key = (addr.lower(), selector.lower())
        if key in self.raise_for:
            raise RuntimeError("simulated RPC error")
        return self.address.get(key)

    def call_bool(
        self, addr: str, selector: str,
    ):
        key = (addr.lower(), selector.lower())
        if key in self.raise_for:
            raise RuntimeError("simulated RPC error")
        return self.bool_v.get(key)

    def token_balance_of(
        self, token: str, holder: str,
    ):
        key = (token.lower(), holder.lower(), "balance")
        if key in self.raise_for:
            raise RuntimeError("simulated RPC error")
        return self.uint256.get(key)

    def call_has_role(
        self, addr: str, role_hash: str, account: str,
    ):
        key = (addr.lower(), role_hash.lower(), account.lower())
        if key in self.raise_for:
            raise RuntimeError("simulated RPC error")
        return self.has_role.get(key)


def _checker(backend) -> InvariantChecker:
    return InvariantChecker(backend=backend)


def test_check_uint256_eq_pass():
    inv = Invariant(
        id="X-1", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.HIGH,
        spec_text="x() == 42",
        kind=InvariantKind.UINT256_EQ,
        selector="0xdead", expected=42,
    )
    backend = _MockBackend()
    backend.uint256[("0xabc", "0xdead")] = 42
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.PASS
    assert result.value == 42


def test_check_uint256_eq_fail():
    inv = Invariant(
        id="X-2", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.HIGH,
        spec_text="x() == 42",
        kind=InvariantKind.UINT256_EQ,
        selector="0xdead", expected=42,
    )
    backend = _MockBackend()
    backend.uint256[("0xabc", "0xdead")] = 100
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.FAIL
    assert result.value == 100


def test_check_uint256_eq_skipped_on_rpc_error():
    inv = Invariant(
        id="X-3", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.HIGH,
        spec_text="x() == 42",
        kind=InvariantKind.UINT256_EQ,
        selector="0xdead", expected=42,
    )
    backend = _MockBackend()
    backend.raise_for.add(("0xabc", "0xdead"))
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.SKIPPED
    assert "rpc" in (result.error or "").lower()


def test_check_uint256_gte_pass():
    inv = Invariant(
        id="X-4", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.HIGH,
        spec_text="x() >= 100",
        kind=InvariantKind.UINT256_GTE,
        selector="0xdead", expected=100,
    )
    backend = _MockBackend()
    backend.uint256[("0xabc", "0xdead")] = 200
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.PASS


def test_check_uint256_gte_fail():
    inv = Invariant(
        id="X-5", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.HIGH,
        spec_text="x() >= 100",
        kind=InvariantKind.UINT256_GTE,
        selector="0xdead", expected=100,
    )
    backend = _MockBackend()
    backend.uint256[("0xabc", "0xdead")] = 50
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.FAIL


def test_check_address_eq_pass_case_insensitive():
    inv = Invariant(
        id="X-6", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.HIGH,
        spec_text="owner() == 0xab",
        kind=InvariantKind.ADDRESS_EQ,
        selector="0xowner",
        expected="0xABcdef0000000000000000000000000000000001",
    )
    backend = _MockBackend()
    backend.address[("0xabc", "0xowner")] = (
        "0xabcdef0000000000000000000000000000000001"
    )
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.PASS


def test_check_address_eq_fail():
    inv = Invariant(
        id="X-7", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.HIGH,
        spec_text="owner() == ...",
        kind=InvariantKind.ADDRESS_EQ,
        selector="0xowner",
        expected="0x" + "11" * 20,
    )
    backend = _MockBackend()
    backend.address[("0xabc", "0xowner")] = "0x" + "22" * 20
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.FAIL


def test_check_bool_read_observable():
    """BOOL_READ is observability — never fails on value
    itself, just surfaces. Used for paused() etc."""
    inv = Invariant(
        id="X-8", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.MEDIUM,
        spec_text="paused() — operator-observable",
        kind=InvariantKind.BOOL_READ,
        selector="0xpaused", expected=None,
    )
    backend = _MockBackend()
    backend.bool_v[("0xabc", "0xpaused")] = True
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.PASS
    assert result.value is True


def test_check_balance_gte_claimable_pass():
    """The solvency invariant — backend looks up
    balance(contract) and totalClaimable separately."""
    inv = Invariant(
        id="X-9", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="ftns.balanceOf(this) >= totalClaimable",
        kind=InvariantKind.BALANCE_GTE_CLAIMABLE,
        selector="0xtotalclaimable",
        # extra params for the ftns address (looked up via
        # a contract method too, or supplied in params)
        params={
            "ftns_selector": "0xftnsaddr",
            "totalclaimable_selector": "0xtotalclaimable",
        },
    )
    backend = _MockBackend()
    contract_addr = "0xc0ffee"
    ftns_addr = "0x" + "ff" * 20
    backend.address[(contract_addr, "0xftnsaddr")] = ftns_addr
    backend.uint256[
        (contract_addr, "0xtotalclaimable")
    ] = 1_000
    backend.uint256[
        (ftns_addr, contract_addr, "balance")
    ] = 1_500
    result = _checker(backend).check_one(inv, contract_addr)
    assert result.status == InvariantStatus.PASS
    assert "balance=1500" in (result.diagnostic or "")
    assert "totalClaimable=1000" in (result.diagnostic or "")


def test_check_balance_gte_claimable_fail():
    inv = Invariant(
        id="X-10", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="ftns.balanceOf(this) >= totalClaimable",
        kind=InvariantKind.BALANCE_GTE_CLAIMABLE,
        selector="0xtotalclaimable",
        params={
            "ftns_selector": "0xftnsaddr",
            "totalclaimable_selector": "0xtotalclaimable",
        },
    )
    backend = _MockBackend()
    contract_addr = "0xc0ffee"
    ftns_addr = "0x" + "ff" * 20
    backend.address[(contract_addr, "0xftnsaddr")] = ftns_addr
    backend.uint256[
        (contract_addr, "0xtotalclaimable")
    ] = 2_000
    backend.uint256[
        (ftns_addr, contract_addr, "balance")
    ] = 1_500
    result = _checker(backend).check_one(inv, contract_addr)
    assert result.status == InvariantStatus.FAIL


def test_check_balance_gte_claimable_skipped_on_rpc_fail():
    inv = Invariant(
        id="X-11", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="ftns.balanceOf(this) >= totalClaimable",
        kind=InvariantKind.BALANCE_GTE_CLAIMABLE,
        selector="0xtotalclaimable",
        params={
            "ftns_selector": "0xftnsaddr",
            "totalclaimable_selector": "0xtotalclaimable",
        },
    )
    backend = _MockBackend()
    contract_addr = "0xc0ffee"
    backend.raise_for.add((contract_addr, "0xftnsaddr"))
    result = _checker(backend).check_one(inv, contract_addr)
    assert result.status == InvariantStatus.SKIPPED


# ── check_contract aggregation ───────────────────────


def test_check_contract_returns_list():
    backend = _MockBackend()
    # No backend data — all RPC reads return None; checker
    # marks each None-return as SKIPPED (can't verify).
    results = _checker(backend).check_contract(
        "royalty_distributor", contract_address="0xabc",
    )
    assert (
        len(results)
        == len(INVARIANT_REGISTRY["royalty_distributor"])
    )
    # All skipped because backend returns None
    for r in results:
        assert r.status == InvariantStatus.SKIPPED


def test_check_contract_unknown_returns_empty():
    backend = _MockBackend()
    assert _checker(backend).check_contract(
        "nonexistent", contract_address="0xabc",
    ) == []


# ── Public surface ───────────────────────────────────


def test_invariant_to_dict_serializable():
    inv = INVARIANT_REGISTRY["royalty_distributor"][0]
    d = inv.to_dict()
    assert d["id"] == inv.id
    assert d["contract_name"] == inv.contract_name
    assert d["kind"] == inv.kind.value
    assert d["severity"] == inv.severity.value
    # Callable fields not present
    assert "check_fn" not in d


def test_result_to_dict_serializable():
    inv = Invariant(
        id="X-12", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.HIGH,
        spec_text="x",
        kind=InvariantKind.UINT256_EQ,
        selector="0xdead", expected=42,
    )
    backend = _MockBackend()
    backend.uint256[("0xabc", "0xdead")] = 42
    result = _checker(backend).check_one(inv, "0xabc")
    d = result.to_dict()
    assert d["status"] == "pass"
    assert d["invariant_id"] == "X-12"
    assert d["value"] == 42


# ── Sprint 356 — FTNSToken + EscrowPool extension ────
#
# Background: while researching the §14 item 4 extension we
# discovered the existing `_SEL_FTNS` selector was wrong
# (keccak256("ftns()") first 4 bytes = 0xefa21b41, not the
# 0x9b03f021 that ship sprint 302 had committed). Result: on
# real mainnet RPC, INV-RD-4 — explicitly called "THE money
# invariant" in the module docstring — would SKIP rather
# than catch solvency drift. The mocked-backend tests above
# all pass with any selector, so this stayed invisible until
# we tried to extend the harness to additional contracts.
#
# This block adds: the selector correctness pin (regression
# test on the discovered bug), the new UINT256_LTE kind for
# supply-cap-style invariants, FTNSToken registry entries
# (supply ceiling), and EscrowPool registry entries
# (solvency mirror of INV-RD-4 against totalEscrowedBalance).


def test_ftns_selector_pinned_to_canonical_keccak():
    """Regression pin: the `ftns()` getter selector MUST be
    the keccak256("ftns()") first-4-bytes value 0xefa21b41.
    The original sprint 302 commit had 0x9b03f021 which is
    NOT the correct selector and would silently SKIP INV-RD-4
    on real RPC. Catching this is exactly what this harness
    was built to do — but the harness itself had a typo.
    """
    from prsm.economy.web3 import formal_invariants as fi
    assert fi._SEL_FTNS == "0xefa21b41", (
        f"_SEL_FTNS was {fi._SEL_FTNS}; canonical keccak256("
        f"'ftns()')[:4] is 0xefa21b41"
    )


def test_uint256_lte_kind_value():
    assert InvariantKind.UINT256_LTE.value == "uint256_lte"


def test_check_uint256_lte_pass():
    inv = Invariant(
        id="X-LTE-1", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="totalSupply() <= MAX_SUPPLY",
        kind=InvariantKind.UINT256_LTE,
        selector="0xdead",
        expected=1_000_000_000 * 10**18,
    )
    backend = _MockBackend()
    backend.uint256[("0xabc", "0xdead")] = (
        100_000_000 * 10**18
    )
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.PASS
    assert result.value == 100_000_000 * 10**18


def test_check_uint256_lte_pass_at_boundary():
    inv = Invariant(
        id="X-LTE-2", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="totalSupply() <= MAX_SUPPLY",
        kind=InvariantKind.UINT256_LTE,
        selector="0xdead", expected=1000,
    )
    backend = _MockBackend()
    # Exactly at boundary — LTE means inclusive of equal
    backend.uint256[("0xabc", "0xdead")] = 1000
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.PASS


def test_check_uint256_lte_fail_supply_breach():
    """If totalSupply ever exceeds MAX_SUPPLY, the contract
    has been compromised (MINTER_ROLE was supposed to enforce
    this on every mint). Failure here = monetary base attack.
    """
    inv = Invariant(
        id="X-LTE-3", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="totalSupply() <= MAX_SUPPLY",
        kind=InvariantKind.UINT256_LTE,
        selector="0xdead",
        expected=1_000_000_000 * 10**18,
    )
    backend = _MockBackend()
    backend.uint256[("0xabc", "0xdead")] = (
        1_000_000_001 * 10**18
    )
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.FAIL
    assert "1000000001" in (result.diagnostic or "")


def test_check_uint256_lte_skipped_on_rpc_error():
    inv = Invariant(
        id="X-LTE-4", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="totalSupply() <= MAX_SUPPLY",
        kind=InvariantKind.UINT256_LTE,
        selector="0xdead", expected=1000,
    )
    backend = _MockBackend()
    backend.raise_for.add(("0xabc", "0xdead"))
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.SKIPPED


def test_check_uint256_lte_skipped_on_none():
    inv = Invariant(
        id="X-LTE-5", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="x", kind=InvariantKind.UINT256_LTE,
        selector="0xdead", expected=1000,
    )
    backend = _MockBackend()  # no value set → returns None
    result = _checker(backend).check_one(inv, "0xabc")
    assert result.status == InvariantStatus.SKIPPED


# ── FTNSToken registry ───────────────────────────────


def test_registry_has_ftns_token():
    assert "ftns_token" in INVARIANT_REGISTRY
    invs = INVARIANT_REGISTRY["ftns_token"]
    assert len(invs) >= 2


def test_ftns_max_supply_invariant_pinned_to_1B():
    invs = INVARIANT_REGISTRY["ftns_token"]
    max_inv = next(
        (i for i in invs if i.id == "INV-FT-1"), None,
    )
    assert max_inv is not None
    assert max_inv.severity == InvariantSeverity.CRITICAL
    assert max_inv.kind == InvariantKind.UINT256_EQ
    # 1B FTNS in wei
    assert max_inv.expected == 1_000_000_000 * 10**18


def test_ftns_total_supply_lte_max_supply_invariant():
    invs = INVARIANT_REGISTRY["ftns_token"]
    sup = next(
        (i for i in invs if i.id == "INV-FT-2"), None,
    )
    assert sup is not None
    assert sup.severity == InvariantSeverity.CRITICAL
    assert sup.kind == InvariantKind.UINT256_LTE
    assert sup.expected == 1_000_000_000 * 10**18


# ── EscrowPool registry ──────────────────────────────


def test_registry_has_escrow_pool():
    assert "escrow_pool" in INVARIANT_REGISTRY
    invs = INVARIANT_REGISTRY["escrow_pool"]
    assert len(invs) >= 1


def test_escrow_pool_solvency_invariant_critical():
    """Mirror of INV-RD-4 against totalEscrowedBalance.
    If ftns.balanceOf(EscrowPool) drops below the sum of
    requester escrow credits, some requester withdraw or
    batch-settlement will revert at the ERC-20 transfer
    boundary — operational impact is the same shape as
    RoyaltyDistributor insolvency.
    """
    invs = INVARIANT_REGISTRY["escrow_pool"]
    sol = next(
        (i for i in invs
         if i.kind == InvariantKind.BALANCE_GTE_CLAIMABLE),
        None,
    )
    assert sol is not None
    assert sol.severity == InvariantSeverity.CRITICAL
    # Reserve-label override should surface in the spec_text
    assert "totalEscrowedBalance" in sol.spec_text


def test_escrow_pool_solvency_diagnostic_uses_reserve_label():
    """When `reserve_label` param is set, the
    balance-gte-claimable handler MUST surface that label in
    the diagnostic. Without it, operators reading EscrowPool
    output would see 'totalClaimable=N' which is the wrong
    contract's variable name."""
    inv = Invariant(
        id="X-EP-DIAG", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="ftns.balanceOf(this) >= totalEscrowedBalance",
        kind=InvariantKind.BALANCE_GTE_CLAIMABLE,
        selector="0xtotalescrowed",
        params={
            "ftns_selector": "0xftnsaddr",
            "totalclaimable_selector": "0xtotalescrowed",
            "reserve_label": "totalEscrowedBalance",
        },
    )
    backend = _MockBackend()
    contract_addr = "0xpool"
    ftns_addr = "0x" + "aa" * 20
    backend.address[(contract_addr, "0xftnsaddr")] = ftns_addr
    backend.uint256[
        (contract_addr, "0xtotalescrowed")
    ] = 5_000
    backend.uint256[
        (ftns_addr, contract_addr, "balance")
    ] = 5_500
    result = _checker(backend).check_one(inv, contract_addr)
    assert result.status == InvariantStatus.PASS
    assert (
        "totalEscrowedBalance=5000"
        in (result.diagnostic or "")
    )
    # Old label MUST NOT appear when override is set
    assert "totalClaimable" not in (result.diagnostic or "")


_FOUNDATION_SAFE = (
    "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791"
)
_DISARMED_HOT_KEY = (
    "0x8eaA00FF741323bc8B0ab1290c544738D9b2f012"
)
_DEFAULT_ADMIN_ROLE = "0x" + "00" * 32
_MINTER_ROLE = (
    "0x9f2df0fed2c77648de5860a4cc508cd0818c85b8b8a1ab4ceeef8d981c8956a6"
)


def test_has_role_kind_value():
    assert InvariantKind.HAS_ROLE_EQ.value == "has_role_eq"


def test_check_has_role_pass_true_match():
    """Positive assertion: Foundation Safe HAS admin role.
    Invariant expects True; backend returns True; PASS."""
    inv = Invariant(
        id="X-HR-1", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text=(
            "hasRole(DEFAULT_ADMIN_ROLE, Foundation Safe)"
        ),
        kind=InvariantKind.HAS_ROLE_EQ,
        selector="",
        expected=True,
        params={
            "role_hash": _DEFAULT_ADMIN_ROLE,
            "account": _FOUNDATION_SAFE,
        },
    )
    backend = _MockBackend()
    key = (
        "0xtoken",
        _DEFAULT_ADMIN_ROLE,
        _FOUNDATION_SAFE.lower(),
    )
    backend.has_role[key] = True
    result = _checker(backend).check_one(inv, "0xtoken")
    assert result.status == InvariantStatus.PASS
    assert result.value is True


def test_check_has_role_pass_false_match():
    """Negative assertion: disarmed hot key MUST NOT hold
    MINTER_ROLE per CR-2026-05-06-3. Invariant expects
    False; backend returns False; PASS (the disarm
    actually held)."""
    inv = Invariant(
        id="X-HR-2", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text=(
            "hasRole(MINTER_ROLE, disarmed_hot_key) == false"
        ),
        kind=InvariantKind.HAS_ROLE_EQ,
        selector="",
        expected=False,
        params={
            "role_hash": _MINTER_ROLE,
            "account": _DISARMED_HOT_KEY,
        },
    )
    backend = _MockBackend()
    key = (
        "0xtoken", _MINTER_ROLE, _DISARMED_HOT_KEY.lower(),
    )
    backend.has_role[key] = False
    result = _checker(backend).check_one(inv, "0xtoken")
    assert result.status == InvariantStatus.PASS
    assert result.value is False


def test_check_has_role_fail_disarm_broken():
    """The high-leverage failure mode: disarmed hot key
    suddenly holds MINTER_ROLE again. Expected False;
    backend returns True; FAIL with audit-visible
    diagnostic."""
    inv = Invariant(
        id="X-HR-3", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text=(
            "hasRole(MINTER_ROLE, disarmed_hot_key) == false"
        ),
        kind=InvariantKind.HAS_ROLE_EQ,
        selector="",
        expected=False,
        params={
            "role_hash": _MINTER_ROLE,
            "account": _DISARMED_HOT_KEY,
        },
    )
    backend = _MockBackend()
    key = (
        "0xtoken", _MINTER_ROLE, _DISARMED_HOT_KEY.lower(),
    )
    backend.has_role[key] = True
    result = _checker(backend).check_one(inv, "0xtoken")
    assert result.status == InvariantStatus.FAIL
    assert result.value is True


def test_check_has_role_fail_admin_lost():
    """Other failure mode: Foundation Safe loses admin role
    via accidental grantRole/revokeRole sequence. Expected
    True; backend returns False; FAIL."""
    inv = Invariant(
        id="X-HR-4", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="hasRole(DEFAULT_ADMIN_ROLE, foundation)",
        kind=InvariantKind.HAS_ROLE_EQ,
        selector="",
        expected=True,
        params={
            "role_hash": _DEFAULT_ADMIN_ROLE,
            "account": _FOUNDATION_SAFE,
        },
    )
    backend = _MockBackend()
    key = (
        "0xtoken",
        _DEFAULT_ADMIN_ROLE,
        _FOUNDATION_SAFE.lower(),
    )
    backend.has_role[key] = False
    result = _checker(backend).check_one(inv, "0xtoken")
    assert result.status == InvariantStatus.FAIL


def test_check_has_role_skipped_on_none():
    inv = Invariant(
        id="X-HR-5", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="x", kind=InvariantKind.HAS_ROLE_EQ,
        selector="", expected=True,
        params={
            "role_hash": _MINTER_ROLE,
            "account": _DISARMED_HOT_KEY,
        },
    )
    backend = _MockBackend()  # no has_role set → None
    result = _checker(backend).check_one(inv, "0xtoken")
    assert result.status == InvariantStatus.SKIPPED


def test_check_has_role_skipped_on_rpc_error():
    inv = Invariant(
        id="X-HR-6", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="x", kind=InvariantKind.HAS_ROLE_EQ,
        selector="", expected=True,
        params={
            "role_hash": _MINTER_ROLE,
            "account": _DISARMED_HOT_KEY,
        },
    )
    backend = _MockBackend()
    key = (
        "0xtoken", _MINTER_ROLE, _DISARMED_HOT_KEY.lower(),
    )
    backend.raise_for.add(key)
    result = _checker(backend).check_one(inv, "0xtoken")
    assert result.status == InvariantStatus.SKIPPED


def test_check_has_role_skipped_on_missing_params():
    """If role_hash or account is missing from params,
    SKIPPED rather than crash — same pattern as
    BALANCE_GTE_CLAIMABLE's selector-missing path."""
    inv = Invariant(
        id="X-HR-7", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="x", kind=InvariantKind.HAS_ROLE_EQ,
        selector="", expected=True,
        params={},  # missing both
    )
    backend = _MockBackend()
    result = _checker(backend).check_one(inv, "0xtoken")
    assert result.status == InvariantStatus.SKIPPED


def test_ftns_has_admin_role_invariant_pinned():
    """INV-FT-3: Foundation Safe is the sole admin per
    CR-2026-05-06-3. Pinned invariant catches if admin
    grants drift."""
    invs = INVARIANT_REGISTRY["ftns_token"]
    inv = next((i for i in invs if i.id == "INV-FT-3"), None)
    assert inv is not None
    assert inv.kind == InvariantKind.HAS_ROLE_EQ
    assert inv.severity == InvariantSeverity.CRITICAL
    assert inv.expected is True
    assert (
        inv.params["role_hash"].lower()
        == _DEFAULT_ADMIN_ROLE.lower()
    )
    assert (
        inv.params["account"].lower()
        == _FOUNDATION_SAFE.lower()
    )


def test_ftns_minter_role_disarmed_invariant_pinned():
    """INV-FT-4: disarmed hot key MUST NOT hold MINTER_ROLE.
    The 900M-FTNS unilateral-mint attack surface that
    CR-2026-05-06-3 closed. Pinned invariant catches re-arm."""
    invs = INVARIANT_REGISTRY["ftns_token"]
    inv = next((i for i in invs if i.id == "INV-FT-4"), None)
    assert inv is not None
    assert inv.kind == InvariantKind.HAS_ROLE_EQ
    assert inv.severity == InvariantSeverity.CRITICAL
    assert inv.expected is False  # NEGATIVE assertion
    assert (
        inv.params["role_hash"].lower()
        == _MINTER_ROLE.lower()
    )
    assert (
        inv.params["account"].lower()
        == _DISARMED_HOT_KEY.lower()
    )


def test_ftns_admin_role_disarmed_invariant_pinned():
    """INV-FT-5: disarmed hot key MUST NOT hold admin role
    either. The full disarm-verification surface from
    CR-2026-05-06-3 covers both MINTER_ROLE and
    DEFAULT_ADMIN_ROLE."""
    invs = INVARIANT_REGISTRY["ftns_token"]
    inv = next((i for i in invs if i.id == "INV-FT-5"), None)
    assert inv is not None
    assert inv.kind == InvariantKind.HAS_ROLE_EQ
    assert inv.severity == InvariantSeverity.CRITICAL
    assert inv.expected is False
    assert (
        inv.params["role_hash"].lower()
        == _DEFAULT_ADMIN_ROLE.lower()
    )
    assert (
        inv.params["account"].lower()
        == _DISARMED_HOT_KEY.lower()
    )


# ── Sprint 358 — EmissionController halving-cycle pin ──


def test_registry_has_emission_controller():
    assert "emission_controller" in INVARIANT_REGISTRY
    invs = INVARIANT_REGISTRY["emission_controller"]
    assert len(invs) >= 2


def test_emission_epoch_duration_pinned_to_4_years():
    """INV-EC-1: mainnet EPOCH_DURATION_SECONDS == 4 years
    (126144000s). The halving cadence is the canonical
    monetary-policy parameter for Phase 8 emissions; drift
    would either dilute or constrict FTNS issuance.
    Enforced at construction by chainid-8453 check;
    runtime invariant guards against contract substitution.
    """
    invs = INVARIANT_REGISTRY["emission_controller"]
    inv = next((i for i in invs if i.id == "INV-EC-1"), None)
    assert inv is not None
    assert inv.kind == InvariantKind.UINT256_EQ
    assert inv.severity == InvariantSeverity.CRITICAL
    assert inv.expected == 4 * 365 * 86400


# ── Sprint 359 — CompensationDistributor / StorageSlashing
#                    / StakeBond extension ─────────────


def test_registry_has_compensation_distributor():
    assert "compensation_distributor" in INVARIANT_REGISTRY
    invs = INVARIANT_REGISTRY["compensation_distributor"]
    assert len(invs) >= 2


def test_compensation_min_weight_schedule_delay_pinned():
    invs = INVARIANT_REGISTRY["compensation_distributor"]
    inv = next((i for i in invs if i.id == "INV-CD-1"), None)
    assert inv is not None
    assert inv.kind == InvariantKind.UINT256_EQ
    assert inv.severity == InvariantSeverity.CRITICAL
    assert inv.expected == 90 * 86400


def test_compensation_owner_is_foundation():
    invs = INVARIANT_REGISTRY["compensation_distributor"]
    inv = next((i for i in invs if i.id == "INV-CD-2"), None)
    assert inv is not None
    assert inv.kind == InvariantKind.ADDRESS_EQ
    assert (
        inv.expected.lower() == _FOUNDATION_SAFE.lower()
    )


def test_registry_has_storage_slashing():
    assert "storage_slashing" in INVARIANT_REGISTRY
    invs = INVARIANT_REGISTRY["storage_slashing"]
    assert len(invs) >= 3


def test_storage_min_max_grace_pinned():
    invs = INVARIANT_REGISTRY["storage_slashing"]
    min_g = next((i for i in invs if i.id == "INV-SS-1"), None)
    max_g = next((i for i in invs if i.id == "INV-SS-2"), None)
    assert min_g is not None
    assert min_g.expected == 3600  # 1 hour
    assert max_g is not None
    assert max_g.expected == 30 * 86400  # 30 days


def test_storage_owner_is_foundation():
    invs = INVARIANT_REGISTRY["storage_slashing"]
    inv = next((i for i in invs if i.id == "INV-SS-3"), None)
    assert inv is not None
    assert inv.kind == InvariantKind.ADDRESS_EQ
    assert (
        inv.expected.lower() == _FOUNDATION_SAFE.lower()
    )


def test_registry_has_stake_bond():
    assert "stake_bond" in INVARIANT_REGISTRY
    invs = INVARIANT_REGISTRY["stake_bond"]
    assert len(invs) >= 4


def test_stake_bond_unbond_delay_bounds_pinned():
    invs = INVARIANT_REGISTRY["stake_bond"]
    min_d = next((i for i in invs if i.id == "INV-SB-1"), None)
    max_d = next((i for i in invs if i.id == "INV-SB-2"), None)
    assert min_d is not None
    assert min_d.expected == 86400  # 1 day
    assert max_d is not None
    assert max_d.expected == 30 * 86400  # 30 days


def test_stake_bond_challenger_bounty_pinned():
    """The anti-confiscation invariant: challenger bounty
    locked at 70% of slashed amount. Mirrors INV-RD-1's
    network-fee anti-tamper pattern."""
    invs = INVARIANT_REGISTRY["stake_bond"]
    inv = next((i for i in invs if i.id == "INV-SB-3"), None)
    assert inv is not None
    assert inv.kind == InvariantKind.UINT256_EQ
    assert inv.severity == InvariantSeverity.CRITICAL
    assert inv.expected == 7000  # 70% in bps


def test_stake_bond_owner_is_foundation():
    invs = INVARIANT_REGISTRY["stake_bond"]
    inv = next((i for i in invs if i.id == "INV-SB-4"), None)
    assert inv is not None
    assert inv.kind == InvariantKind.ADDRESS_EQ
    assert (
        inv.expected.lower() == _FOUNDATION_SAFE.lower()
    )


# ── sp984 — CreatorStakeRegistry runtime invariants (PENDING_COMMISSION) ──


def test_registry_has_creator_stake_registry():
    """The §14 money-custody contract joins the runtime invariant
    registry (PUBLIC pinned spec), completing the dual-lane coverage
    every other money contract has (symbolic sp983 + runtime here)."""
    assert "creator_stake_registry" in INVARIANT_REGISTRY
    invs = INVARIANT_REGISTRY["creator_stake_registry"]
    assert len(invs) >= 4


def test_creator_stake_registry_unbond_delay_bounds_pinned():
    invs = INVARIANT_REGISTRY["creator_stake_registry"]
    min_d = next((i for i in invs if i.id == "INV-CSR-1"), None)
    max_d = next((i for i in invs if i.id == "INV-CSR-2"), None)
    assert min_d is not None and min_d.expected == 86400  # 1 day
    assert max_d is not None and max_d.expected == 30 * 86400  # 30 days


def test_creator_stake_registry_owner_is_foundation():
    """The critical governance pin: post-ceremony the registry is
    sole-owned by the Foundation Safe (mirrors INV-SB-4)."""
    invs = INVARIANT_REGISTRY["creator_stake_registry"]
    inv = next((i for i in invs if i.id == "INV-CSR-3"), None)
    assert inv is not None
    assert inv.kind == InvariantKind.ADDRESS_EQ
    assert inv.severity == InvariantSeverity.CRITICAL
    assert inv.expected.lower() == _FOUNDATION_SAFE.lower()


def test_creator_stake_registry_paused_readable():
    invs = INVARIANT_REGISTRY["creator_stake_registry"]
    inv = next((i for i in invs if i.id == "INV-CSR-4"), None)
    assert inv is not None
    assert inv.kind == InvariantKind.BOOL_READ


# ── sp985 — BatchSettlementRegistry runtime invariants (LIVE on mainnet) ──


def test_registry_has_settlement_registry():
    """The deployed-on-mainnet settlement + consensus-slashing contract
    (BatchSettlementRegistry, settlement_registry in networks.py) joins the
    runtime invariant registry — it was previously in NEITHER formal lane
    despite being live and holding the slash path."""
    assert "settlement_registry" in INVARIANT_REGISTRY
    invs = INVARIANT_REGISTRY["settlement_registry"]
    assert len(invs) >= 6


def test_settlement_registry_owner_is_foundation():
    invs = INVARIANT_REGISTRY["settlement_registry"]
    inv = next((i for i in invs if i.id == "INV-BSR-1"), None)
    assert inv is not None
    assert inv.kind == InvariantKind.ADDRESS_EQ
    assert inv.severity == InvariantSeverity.CRITICAL
    assert inv.expected.lower() == _FOUNDATION_SAFE.lower()


def test_settlement_registry_challenge_window_floor_pinned():
    """The dispute-window FLOOR. Drift toward 0 would let a slash finalize
    before honest providers can dispute a CONSENSUS_MISMATCH challenge — a
    direct false-slash risk on the live slashing path."""
    invs = INVARIANT_REGISTRY["settlement_registry"]
    inv = next((i for i in invs if i.id == "INV-BSR-2"), None)
    assert inv is not None
    assert inv.kind == InvariantKind.UINT256_EQ
    assert inv.severity == InvariantSeverity.CRITICAL
    assert inv.expected == 3600  # 1 hour


def test_settlement_registry_lookback_and_gas_floors_pinned():
    invs = INVARIANT_REGISTRY["settlement_registry"]
    lookback = next((i for i in invs if i.id == "INV-BSR-4"), None)
    slash_gas = next((i for i in invs if i.id == "INV-BSR-6"), None)
    assert lookback is not None and lookback.expected == 86400  # 1 day
    # The MIN_SLASH_GAS floor — the L4 audit fix that stops a slash call being
    # griefed into an out-of-gas revert (which would silently spare a guilty
    # provider).
    assert slash_gas is not None and slash_gas.expected == 150000


# ── sp988 — Fleet governance sweep: last deployed contracts' owner/admin pins ──


def test_registry_has_publisher_key_anchor_and_key_distribution():
    """The last two deployed contracts with governance state join the registry,
    completing fleet-wide governance-capture monitoring."""
    assert "publisher_key_anchor" in INVARIANT_REGISTRY
    assert "key_distribution" in INVARIANT_REGISTRY


def test_publisher_key_anchor_admin_is_foundation():
    """PublisherKeyAnchor.admin() is immutable; pinning it == Foundation Safe
    confirms the deployed contract's admin is the Safe (detects a wrong-admin
    deployment, which is unrecoverable since admin is immutable)."""
    invs = INVARIANT_REGISTRY["publisher_key_anchor"]
    inv = next((i for i in invs if i.id == "INV-PKA-1"), None)
    assert inv is not None
    assert inv.kind == InvariantKind.ADDRESS_EQ
    assert inv.severity == InvariantSeverity.CRITICAL
    assert inv.expected.lower() == _FOUNDATION_SAFE.lower()


def test_key_distribution_owner_is_foundation():
    invs = INVARIANT_REGISTRY["key_distribution"]
    inv = next((i for i in invs if i.id == "INV-KD-1"), None)
    assert inv is not None
    assert inv.kind == InvariantKind.ADDRESS_EQ
    assert inv.severity == InvariantSeverity.CRITICAL
    assert inv.expected.lower() == _FOUNDATION_SAFE.lower()


# ── sp987 — Provenance registries: royalty-rate ceiling (both deployed) ──


def test_registry_has_provenance_registries():
    """Both deployed provenance registries (v1 + v2, read by the off-chain
    RoyaltyDistributor) join the runtime registry — neither was covered."""
    assert "provenance_registry_v2" in INVARIANT_REGISTRY
    assert "provenance_registry" in INVARIANT_REGISTRY


def test_provenance_max_royalty_rate_pinned():
    """The cross-contract over-allocation guard: MAX_ROYALTY_RATE_BPS (9800)
    must equal 10000 - RoyaltyDistributor.NETWORK_FEE_BPS (200, pinned by
    INV-RD-1), so a registered royalty rate + the network fee can NEVER exceed
    gross revenue. Pinned on BOTH deployed registries — a v2 redeploy or a
    network-fee change that broke the pairing would surface here."""
    for name, inv_id in [
        ("provenance_registry_v2", "INV-PRV2-1"),
        ("provenance_registry", "INV-PRV-1"),
    ]:
        invs = INVARIANT_REGISTRY[name]
        inv = next((i for i in invs if i.id == inv_id), None)
        assert inv is not None, name
        assert inv.kind == InvariantKind.UINT256_EQ
        assert inv.severity == InvariantSeverity.CRITICAL
        assert inv.expected == 9800


def test_emission_mainnet_chain_id_pinned():
    """INV-EC-2: BASE_MAINNET_CHAIN_ID() == 8453. The
    chainid pin that enforces the 4-year mainnet halving
    constraint at construction. Drift would suggest
    contract substitution to a non-Base deployment."""
    invs = INVARIANT_REGISTRY["emission_controller"]
    inv = next((i for i in invs if i.id == "INV-EC-2"), None)
    assert inv is not None
    assert inv.kind == InvariantKind.UINT256_EQ
    assert inv.severity == InvariantSeverity.CRITICAL
    assert inv.expected == 8453


def test_balance_gte_claimable_default_label_preserved():
    """Backward-compat — when `reserve_label` is NOT in
    params, the diagnostic still says 'totalClaimable' so
    INV-RD-4's output shape stays identical to sprint 302."""
    inv = Invariant(
        id="X-RD-DIAG", contract_name="x", title="t",
        description="d",
        severity=InvariantSeverity.CRITICAL,
        spec_text="ftns.balanceOf(this) >= totalClaimable",
        kind=InvariantKind.BALANCE_GTE_CLAIMABLE,
        selector="0xtc",
        params={
            "ftns_selector": "0xftnsaddr",
            "totalclaimable_selector": "0xtc",
            # NO reserve_label override
        },
    )
    backend = _MockBackend()
    contract_addr = "0xrd"
    ftns_addr = "0x" + "bb" * 20
    backend.address[(contract_addr, "0xftnsaddr")] = ftns_addr
    backend.uint256[(contract_addr, "0xtc")] = 100
    backend.uint256[
        (ftns_addr, contract_addr, "balance")
    ] = 200
    result = _checker(backend).check_one(inv, contract_addr)
    assert result.status == InvariantStatus.PASS
    assert "totalClaimable=100" in (result.diagnostic or "")
