# PRSM Security Configuration Audit

> **Historical audit snapshot — January 2026. Not a current-state document.**
>
> This audit was captured in January 2026 following the Technical Due Diligence (TDD) remediation work tracked in the [historical remediation plan](REMEDIATION_HARDENING_MASTER_PLAN.md). The control statuses and score-progression narrative reflect that point in time.
>
> **Framing note on "Series A investment readiness":** the original framing assumed PRSM would raise a conventional Series A at the protocol-entity level. That framing is superseded — per `PRSM_Tokenomics.md` §3 and `PRSM_Vision.md` §9, bootstrap capital now flows through **Prismatica equity (Reg D 506(c))**, not a PRSM-entity fundraise. The control inventory below remains directionally useful for any future diligence surface (Prismatica investor diligence, SOC 2 pursuit, enterprise customer security reviews), but the "Before Series A Close" deadlines no longer describe a real process.
>
> **Related docs in this suite:**
> - [`SECURITY_HARDENING.md`](SECURITY_HARDENING.md) — security policy and hardening guide
> - [`SECURITY_HARDENING_CHECKLIST.md`](SECURITY_HARDENING_CHECKLIST.md) — implementation-verification checklist
> - [`PENETRATION_TESTING_GUIDE.md`](PENETRATION_TESTING_GUIDE.md) — pen-test methodology
> - [`REMEDIATION_HARDENING_MASTER_PLAN.md`](REMEDIATION_HARDENING_MASTER_PLAN.md) — sprint-level plan (historical; see post-v1.6 status audit at top)
> - [`2026-04-10-audit-gap-roadmap.md`](2026-04-10-audit-gap-roadmap.md) Phase 6 + Phase 7 — current protocol-layer hardening scope
>
> When a fresh security audit is run (pre-Prismatica Reg D 506(c) raise, pre-enterprise customer engagement, etc.), this document should be re-generated against the then-current code rather than updated in place.

## Overview

This document provides a comprehensive audit of PRSM's security configuration following the Technical Due Diligence remediation. Captured January 2026 as a point-in-time inventory.

**Audit Date:** January 2026
**Score Progression (as reported January 2026):** 4.5 → 6.0 → 7.5 → Target 10.0 — interim scoring from the remediation program; no current authoritative score; see the master roadmap audit verdict for current pillar statuses.

---

## Security Controls Summary

### Authentication & Authorization

| Control | Status | Implementation | File Reference |
|---------|--------|----------------|----------------|
| JWT Signature Verification | ✅ SECURE | `verify_signature: True` | `jwt_handler.py:45` |
| Algorithm Whitelist | ✅ SECURE | HS256, HS384, HS512 only | `jwt_handler.py:12` |
| None Algorithm Rejection | ✅ SECURE | Explicit check | `jwt_handler.py:15` |
| Token Revocation | ✅ SECURE | Redis + PostgreSQL | `jwt_handler.py:89` |
| Required Claims | ✅ SECURE | exp, sub, jti required | `jwt_handler.py:52` |
| OIDC JWKS Verification | ✅ SECURE | PyJWKClient integration | `sso_provider.py:395` |
| SAML XXE Prevention | ✅ SECURE | defusedxml required | `sso_provider.py:122` |
| Password Hashing | ✅ SECURE | bcrypt cost ≥12 | `auth/models.py` |
| Account Lockout | ✅ SECURE | After 5 failed attempts | `auth/middleware.py` |

### Financial Security (FTNS)

| Control | Status | Implementation | File Reference |
|---------|--------|----------------|----------------|
| Row-Level Locking | ✅ SECURE | SELECT FOR UPDATE | `atomic_ftns_service.py:300` |
| Optimistic Concurrency | ✅ SECURE | Version column | `atomic_ftns_service.py:344` |
| Idempotency Keys | ✅ SECURE | 24-hour TTL | `atomic_ftns_service.py:287` |
| Non-Negative Balance | ✅ SECURE | CHECK constraint | `migration_002.sql:22` |
| Atomic Transfers | ✅ SECURE | Transaction isolation | `atomic_ftns_service.py:430` |
| Transaction Recovery | ✅ SECURE | Stuck tx detection | `transaction_recovery.py` |

### API Security

| Control | Status | Implementation | File Reference |
|---------|--------|----------------|----------------|
| HTTPS/TLS | ✅ CONFIGURED | TLS 1.2+ required | `tls_config.py` |
| HSTS | ✅ ENABLED | 1 year max-age | `standards.py:246` |
| CSP | ✅ ENABLED | Restrictive policy | `standards.py:254` |
| X-Frame-Options | ✅ ENABLED | DENY | `standards.py:240` |
| X-Content-Type-Options | ✅ ENABLED | nosniff | `standards.py:237` |
| CORS | ✅ RESTRICTED | Whitelist only | `standards.py:276` |
| Rate Limiting | ✅ ENABLED | Sliding window | `advanced_rate_limiting.py` |
| Request Size Limits | ✅ ENABLED | 10MB max | `middleware.py:175` |

### Infrastructure Security

| Control | Status | Implementation | File Reference |
|---------|--------|----------------|----------------|
| Database TLS | ✅ CONFIGURED | verify-full mode | `tls_config.py:168` |
| Redis TLS | ✅ CONFIGURED | TLS with auth | `tls_config.py:202` |
| Secret Management | ✅ SECURE | Environment vars | `config.py:116` |
| Key Length Validation | ✅ SECURE | 32 char minimum | `config.py:228` |
| Connection Pooling | ✅ CONFIGURED | Size limits | `config.py:123` |

---

## Environment Variables Reference

### Required for Production

```bash
# Application
PRSM_ENVIRONMENT=production
PRSM_DEBUG=false
PRSM_SECRET_KEY=<min-32-char-secret>

# JWT
PRSM_JWT_SECRET=<min-32-char-secret>

# Database
PRSM_DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/prsm

# Redis
PRSM_REDIS_URL=rediss://user:pass@host:6379/0
PRSM_REDIS_PASSWORD=<redis-password>

# TLS
PRSM_TLS_ENABLED=true
PRSM_TLS_MODE=verify-full
PRSM_TLS_MIN_VERSION=TLSv1.2
PRSM_TLS_CERT_FILE=/path/to/cert.pem
PRSM_TLS_KEY_FILE=/path/to/key.pem
PRSM_TLS_CA_FILE=/path/to/ca.pem

# HSTS
PRSM_HSTS_ENABLED=true
PRSM_HSTS_MAX_AGE=31536000
PRSM_HSTS_SUBDOMAINS=true
```

### Optional Configuration

```bash
# Rate Limiting
PRSM_RATE_LIMIT_ENABLED=true
PRSM_RATE_LIMIT_TIER_ANONYMOUS=10
PRSM_RATE_LIMIT_TIER_FREE=60

# Monitoring
PRSM_METRICS_ENABLED=true
PRSM_METRICS_PORT=9090

# Safety
PRSM_CIRCUIT_BREAKER_ENABLED=true
PRSM_SAFETY_MONITORING=true
```

---

## Security Test Coverage

### Unit Tests

| Test Suite | Tests | Pass Rate | File |
|------------|-------|-----------|------|
| JWT Verification | 12 | 100% | `test_jwt_verification.py` |
| Rate Limiting | 10 | 100% | `test_rate_limiting.py` |
| Double-Spend Prevention | 8 | 100% | `test_double_spend_prevention.py` |

### Integration Tests

| Test Suite | Tests | Requires | File |
|------------|-------|----------|------|
| FTNS Concurrency | 12 | PostgreSQL | `test_ftns_concurrency_integration.py` |
| Rate Limiting | 10 | Redis | `test_rate_limiting_integration.py` |
| JWT Authentication | 15 | Redis | `test_jwt_authentication_integration.py` |

### Running Security Tests

```bash
# Unit tests (no external services)
pytest tests/security/ -v

# Integration tests (requires PostgreSQL + Redis)
export DATABASE_URL=postgresql://...
export REDIS_URL=redis://...
pytest tests/integration/ -v

# Full security test suite
pytest tests/security/ tests/integration/ -v --tb=short
```

---

## Compliance Checklist

### OWASP Top 10 2021

| Risk | Status | Mitigation |
|------|--------|------------|
| A01 Broken Access Control | ✅ Mitigated | RBAC, JWT verification, endpoint protection |
| A02 Cryptographic Failures | ✅ Mitigated | TLS 1.2+, strong ciphers, key validation |
| A03 Injection | ✅ Mitigated | Parameterized queries, input sanitization |
| A04 Insecure Design | ✅ Mitigated | Security architecture review completed |
| A05 Security Misconfiguration | ✅ Mitigated | Hardened headers, secure defaults |
| A06 Vulnerable Components | ⚠️ Ongoing | Dependabot enabled, regular updates |
| A07 Auth Failures | ✅ Mitigated | JWT hardening, account lockout |
| A08 Integrity Failures | ✅ Mitigated | Signature verification, CSP |
| A09 Logging Failures | ✅ Mitigated | Comprehensive security logging |
| A10 SSRF | ⚠️ Partial | URL validation on external calls |

### SOC 2 Type II Readiness

| Control | Status | Evidence |
|---------|--------|----------|
| Access Controls | ✅ Ready | RBAC implementation documented |
| Encryption | ✅ Ready | TLS configuration documented |
| Monitoring | ✅ Ready | Security logging implemented |
| Incident Response | ⚠️ Pending | Playbooks needed |
| Change Management | ✅ Ready | Git history, PR reviews |

---

## Deployment Checklist

### Pre-Production

- [ ] All secrets rotated from development values
- [ ] Debug mode disabled (`PRSM_DEBUG=false`)
- [ ] TLS certificates installed and valid
- [ ] Database SSL enabled (`sslmode=verify-full`)
- [ ] Redis authentication configured
- [ ] Rate limits configured for expected load
- [ ] Security headers verified
- [ ] CORS origins restricted to production domains
- [ ] Error messages sanitized (no stack traces)
- [ ] Logging configured (no sensitive data logged)

### Production Launch

- [ ] HTTPS certificates valid and auto-renewing
- [ ] HSTS preload submitted (optional but recommended)
- [ ] Monitoring dashboards configured
- [ ] Security alerting rules active
- [ ] Backup procedures tested
- [ ] Incident response plan documented
- [ ] On-call rotation established

### Post-Launch

- [ ] Penetration test scheduled
- [ ] Vulnerability scanning enabled
- [ ] Dependency updates automated
- [ ] Security metrics baseline established

---

## Known Limitations

### Current

1. **Database Encryption at Rest** - Infrastructure-level concern, not application-controlled
2. **ML Distillation Loss Function** - Incomplete (out of security scope)
3. **Penetration Testing** - Pending third-party audit

### Accepted Risks

1. **In-Memory Rate Limiting Fallback** - When Redis unavailable, falls back to in-memory which doesn't persist across restarts
2. **SAML Signature Verification** - Requires defusedxml package; falls back to rejection if unavailable

---

## Recommendations for 10/10 Score

> **Sequencing note:** original framing tied these to "Before/After Series A Close." Under the equity-investment architecture (per `PRSM_Tokenomics.md` §3), the analogous gates are **Prismatica Reg D 506(c) raise** and **pre-mainnet / pre-enterprise-customer** engagement. The recommendations themselves remain directionally valid; the deadlines have been relabeled below.

### Immediate (pre-Prismatica equity raise / pre-mainnet public launch)

1. **Complete Penetration Test** — engage NCC Group, Trail of Bits, Zellic, Cure53, or equivalent (smart-contract-experienced firm preferred given Phase 1 on-chain components)
2. **Implement Incident Response Playbooks** — document procedures; tie to Risk Register entries A1 (critical contract exploit) and F2 (reputational impact)
3. **Add SAST/DAST to CI/CD** — automated security scanning

### Short-Term (post-raise / during early mainnet operation)

4. **Database Encryption at Rest** — enable at infrastructure level
5. **Redis TLS** — enable in production deployment
6. **Bug Bounty Program** — *proposal only, not launched.* Proposed scope: Immunefi-tier coverage for smart contracts (target ≥$1M payout per Risk Register A1 mitigation) plus HackerOne or equivalent for application surface. PRSM does not run a bug bounty program and does not pay for vulnerability reports; disclosures go through [`SECURITY.md`](../SECURITY.md) (GHSA / security@)
7. **Security Training** — team security awareness

### Long-Term

8. **SOC 2 Type II Certification** - Formal audit
9. **ISO 27001** - If pursuing enterprise customers
10. **PCI DSS** - If handling payment cards directly

---

## Audit Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Lead Developer | | | |
| Security Engineer | | | |
| CTO | | | |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Jan 2026 | PRSM Team | Initial audit post-remediation |
| 1.1 | 2026-04-16 | Cross-reference pass | Added historical-framing banner; reframed "Series A" language to equity-investment architecture; added cross-refs to companion security docs and master roadmap |

---

*Classification: Internal — pre-diligence reference. Document is a January 2026 snapshot; regenerate against current code when a fresh audit is required (pre-Prismatica raise, pre-enterprise customer engagement, or scheduled quarterly).*
