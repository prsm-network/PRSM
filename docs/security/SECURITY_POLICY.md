# PRSM Security Policy

**Version:** 1.0  
**Last Updated:** 2025-06-30  
**Status:** Active

## Overview

This document outlines the security policy for the Protocol for Research, Storage, and Modeling (PRSM) project. It defines the security measures, reporting procedures, and compliance requirements.

## Table of Contents

1. [Supported Versions](#supported-versions)
2. [Security Measures](#security-measures)
3. [Reporting a Vulnerability](#reporting-a-vulnerability)
4. [Security Response Process](#security-response-process)
5. [Security Best Practices](#security-best-practices)
6. [Compliance](#compliance)
7. [Security Contacts](#security-contacts)

---

## Supported Versions

| Version | Supported | Security Updates | End of Life |
|---------|-----------|------------------|-------------|
| 1.7.x | ✅ Active (current, live on Base mainnet) | Yes | TBD |
| 1.6.x | ✅ Maintenance | Critical/high only | TBD |
| ≤ 1.5.x | ❌ End of Life | No | superseded |
| 0.x | ❌ End of Life | No | superseded |

### Version Support Policy

- **Active versions** receive security updates and patches
- **End of Life (EOL)** versions no longer receive security updates
- Security patches are backported to supported versions as needed
- Major security vulnerabilities may trigger emergency releases

---

## Security Measures

### Authentication & Authorization

#### Authentication

| Measure | Implementation | Status |
|---------|---------------|--------|
| JWT Token Management | RS256 algorithm with configurable expiry | ✅ Implemented |
| Multi-Factor Authentication | TOTP-based MFA | ✅ Implemented |
| Enterprise SSO | SAML 2.0, OAuth 2.0 | ✅ Implemented |
| LDAP Integration | Active Directory, OpenLDAP | ✅ Implemented |
| Post-Quantum Auth Prep | Hybrid cryptography ready | ✅ Implemented |
| Session Management | Secure cookies, configurable timeout | ✅ Implemented |

#### Authorization

| Measure | Implementation | Status |
|---------|---------------|--------|
| Role-Based Access Control (RBAC) | Fine-grained permissions | ✅ Implemented |
| Principle of Least Privilege | Default minimal permissions | ✅ Implemented |
| Resource Access Control | Per-resource authorization | ✅ Implemented |
| API Rate Limiting | Multi-tier rate limits | ✅ Implemented |

### Network Security

| Measure | Implementation | Status |
|---------|---------------|--------|
| TLS Enforcement | TLS 1.2+ required | ✅ Implemented |
| CORS Configuration | Configurable allowed origins | ✅ Implemented |
| Content Security Policy | CSP headers | ✅ Implemented |
| Security Headers | HSTS, X-Frame-Options, etc. | ✅ Implemented |
| Rate Limiting | IP, user, endpoint, global | ✅ Implemented |

### Data Protection

| Measure | Implementation | Status |
|---------|---------------|--------|
| Encryption at Rest | AES-256-GCM | ✅ Implemented |
| Encryption in Transit | TLS 1.2+ | ✅ Implemented |
| Input Sanitization | XSS, injection prevention | ✅ Implemented |
| Input Validation | Allowlist-based validation | ✅ Implemented |
| PII Handling | Detection, masking, secure storage | ✅ Implemented |
| Backup Encryption | Encrypted backups | ✅ Implemented |

### Infrastructure Security

| Measure | Implementation | Status |
|---------|---------------|--------|
| Secrets Management | Multi-backend support | ✅ Implemented |
| Security Event Logging | Comprehensive audit trail | ✅ Implemented |
| Dependency Scanning | Automated vulnerability scanning | ✅ Implemented |
| Container Security | Minimal images, non-root | ✅ Implemented |
| Patch Management | Automated updates | ✅ Implemented |

### Application Security

| Measure | Implementation | Status |
|---------|---------------|--------|
| SQL Injection Prevention | Parameterized queries, ORM | ✅ Implemented |
| XSS Prevention | Output encoding, CSP | ✅ Implemented |
| CSRF Protection | Token-based protection | ✅ Implemented |
| Error Handling | Secure error handling | ✅ Implemented |
| File Upload Security | Type validation, size limits | ✅ Implemented |

---

## Reporting a Vulnerability

### How to Report

We take security vulnerabilities seriously. If you discover a security vulnerability, please report it responsibly:

#### Option 1: Email (Preferred)

Send an email to: **security@prsm-network.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any proof-of-concept code
- Your contact information

#### Option 2: PGP Encrypted Email

For sensitive reports, PGP encryption support is planned but **not yet live** — a
published PGP key ID and fingerprint will be added here once the security mailbox
key is provisioned. Until then, email the security address directly (above) and
request a secure channel.

#### Option 3: Security Portal

Submit via our security portal: **https://prsm-network.com/security/report**

### What to Expect

1. **Acknowledgment**: Within 48 hours
2. **Initial Assessment**: Within 5 business days
3. **Status Updates**: Weekly until resolved
4. **Resolution Timeline**: Based on severity

### Vulnerability Severity Levels

| Level | Description | Response Time |
|-------|-------------|---------------|
| **P0 Critical** | Active exploitation, data breach risk | 24 hours |
| **P1 High** | Significant vulnerability, no active exploit | 7 days |
| **P2 Medium** | Limited impact, requires specific conditions | 30 days |
| **P3 Low** | Minor issue, hard to exploit | 90 days |

### Responsible Disclosure Policy

We follow responsible disclosure:

1. **Do not** publicly disclose the vulnerability before we've had a chance to fix it
2. **Do not** access, modify, or delete data that doesn't belong to you
3. **Do not** perform actions that could harm system availability
4. **Do** provide us reasonable time to fix the issue before disclosure
5. **Do** report vulnerabilities in good faith

### Bug Bounty

PRSM does not run a bug bounty program and does not pay for vulnerability
reports. Valid reports are welcomed and, with the reporter's consent, credited
in the published advisory per [SECURITY.md](../../SECURITY.md#recognition).

---

## Security Response Process

### Incident Response Phases

#### Phase 1: Identification (0-2 hours)

1. Receive vulnerability report
2. Acknowledge receipt
3. Assign severity level
4. Assign incident lead

#### Phase 2: Containment (2-24 hours)

1. Assess scope and impact
2. Implement temporary mitigations
3. Preserve evidence
4. Notify affected parties

#### Phase 3: Remediation (24-72 hours)

1. Develop fix
2. Test fix thoroughly
3. Deploy fix to staging
4. Validate fix effectiveness

#### Phase 4: Recovery (72-96 hours)

1. Deploy fix to production
2. Monitor for issues
3. Update documentation
4. Communicate resolution

#### Phase 5: Post-Incident (1-2 weeks)

1. Conduct post-mortem
2. Document lessons learned
3. Update security measures
4. Share findings (where appropriate)

### Communication Templates

#### Internal Notification

```
SECURITY INCIDENT - [SEVERITY]

Summary: [Brief description]
Severity: [P0/P1/P2/P3]
Status: [Investigating/Contained/Resolved]
Impact: [Affected systems/users]
Lead: [Incident lead name]
Next Update: [Time]
```

#### External Notification

```
PRSM Security Advisory

Product: PRSM
Severity: [Critical/High/Medium/Low]
Versions Affected: [Version range]
Versions Fixed: [Version]

Summary: [Non-technical description]

Recommendation: [User action required]

References: [CVE, if applicable]
```

---

## Security Best Practices

### For Developers

#### Code Security

1. **Input Validation**
   - Validate all user inputs
   - Use allowlist validation where possible
   - Sanitize outputs for context

2. **Authentication**
   - Never store passwords in plaintext
   - Use strong hashing (bcrypt, argon2)
   - Implement proper session management

3. **Authorization**
   - Check permissions on every request
   - Implement principle of least privilege
   - Use role-based access control

4. **Cryptography**
   - Use established libraries
   - Never roll your own crypto
   - Keep keys secure and rotate regularly

5. **Error Handling**
   - Don't expose internal details
   - Log errors securely
   - Use generic error messages for users

#### Secure Development

```python
# Good: Parameterized queries
from sqlalchemy import text

query = text("SELECT * FROM users WHERE id = :id")
result = db.execute(query, {"id": user_id})

# Bad: String concatenation
# query = f"SELECT * FROM users WHERE id = {user_id}"  # Never do this!
```

```python
# Good: Input validation
from pydantic import BaseModel, validator

class UserInput(BaseModel):
    username: str
    
    @validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_]{3,20}$', v):
            raise ValueError('Invalid username format')
        return v
```

### For Operators

#### Infrastructure Security

1. **Access Control**
   - Use SSH keys, not passwords
   - Implement MFA for admin access
   - Use principle of least privilege

2. **Network Security**
   - Enable firewall rules
   - Use VPN for internal access
   - Implement network segmentation

3. **Monitoring**
   - Enable comprehensive logging
   - Set up alerting for anomalies
   - Regular security audits

4. **Updates**
   - Apply security patches promptly
   - Automate dependency updates
   - Monitor for vulnerabilities

#### Secrets Management

```bash
# Good: Use secrets manager
export SECRETS_BACKEND=vault
export VAULT_ADDR=https://vault.example.com

# Bad: Hardcode secrets
# export DATABASE_URL="postgres://user:password@host/db"  # Never do this!
```

### For Users

#### Account Security

1. **Strong Passwords**
   - Use at least 16 characters
   - Include uppercase, lowercase, digits, special characters
   - Use a password manager

2. **Multi-Factor Authentication**
   - Enable MFA on your account
   - Use authenticator app (not SMS if possible)
   - Keep backup codes secure

3. **Session Management**
   - Log out when done
   - Don't share sessions
   - Monitor active sessions

---

## Compliance

> **Status note:** The tables below track the *technical controls* PRSM has built
> toward each framework. They do **not** assert formal certification. PRSM is
> **not** SOC 2, ISO 27001, or GDPR *certified/attested* — those require an
> external audit that has not yet been performed. "Control in place" means the
> engineering control exists and is exercised; it is not a certification claim.

### SOC 2 Type II (controls — not certified)

| Control | Control Status | Evidence |
|---------|----------------|----------|
| Access Control | 🟢 Control in place | RBAC, MFA |
| Encryption | 🟢 Control in place | AES-256, TLS 1.2+ |
| Audit Logging | 🟢 Control in place | Comprehensive logs |
| Incident Response | 🟢 Control in place | Runbook, escalation |
| Change Management | 🟢 Control in place | CI/CD, approvals |
| External SOC 2 attestation | 🟡 Not yet attested | Audit not performed |

### GDPR (controls — not formally assessed)

| Requirement | Control Status | Implementation |
|-------------|----------------|----------------|
| Data Minimization | 🟢 Control in place | Collect only necessary data |
| Right to Access | 🟢 Control in place | User data export |
| Right to Erasure | 🟢 Control in place | User data deletion |
| Data Portability | 🟢 Control in place | Standard export formats |
| Privacy by Design | 🟢 Control in place | Built into architecture |
| Formal DPIA / compliance assessment | 🟡 Not yet completed | Pending |

### ISO 27001 (controls — not certified)

| Control | Control Status | Evidence |
|---------|----------------|----------|
| Information Security Policy | 🟢 Control in place | This document |
| Access Control | 🟢 Control in place | RBAC, MFA |
| Cryptography | 🟢 Control in place | AES-256, TLS |
| Operations Security | 🟢 Control in place | Logging, monitoring |
| Communications Security | 🟢 Control in place | Network security |
| ISO 27001 certification | 🟡 Not yet certified | Audit not performed |

---

## Security Contacts

### Security Team

| Role | Contact |
|------|---------|
| Security Lead | security@prsm-network.com |
| Incident Response | incident@prsm-network.com |

### External Contacts

| Purpose | Contact |
|---------|---------|
| CVE Requests | cve@prsm-network.com |
| Security Advisories | advisory@prsm-network.com |
| Press Inquiries | press@prsm-network.com |

### PGP Key

A published PGP public key for encrypted security reports is **not yet
provisioned**. This section will be populated with the key block once the
security mailbox key is generated. Until then, contact the security address to
arrange a secure channel.

---

## Policy Updates

This security policy is reviewed and updated:

- **Quarterly**: Routine review
- **After Incidents**: Post-incident review
- **Major Releases**: Compatibility review
- **Compliance Changes**: Regulatory updates

### Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-06-30 | 1.0 | Initial security policy |

---

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [SOC 2 Trust Services Criteria](https://www.aicpa.org/soc2)
- [ISO/IEC 27001:2022](https://www.iso.org/standard/27001)

---

*This security policy is part of the PRSM project's commitment to security. For questions or clarifications, contact security@prsm-network.com.*