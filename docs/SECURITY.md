# HelixForge Security & Compliance

## Security Overview

HelixForge implements multiple security layers to protect data and ensure safe operation.

## Security Audit Results

### Static Analysis (Bandit)

Last scan: v1.0.0

| Severity | Count | Status |
|----------|-------|--------|
| High     | 0     | Pass   |
| Medium   | 1     | Reviewed (false positive) |
| Low      | 5     | Acceptable |

**Findings:**
- Low: Try/except patterns in statistical analysis (acceptable for graceful degradation)
- Medium: Hardcoded interface check (false positive - used for SSRF prevention, not binding)

### Dependency Audit

Run `pip-audit` to check for known vulnerabilities:
```bash
pip-audit -r requirements.txt
```

## Security Controls

### 1. Input Validation

All user inputs are validated using Pydantic models:

```python
from utils.validation import (
    sanitize_string,      # HTML/SQL injection prevention
    validate_file_path,   # Path traversal prevention
    validate_url,         # SSRF prevention
    validate_sql_identifier  # SQL injection prevention
)
```

**Protected against:**
- SQL Injection
- Path Traversal
- Server-Side Request Forgery (SSRF)
- Cross-Site Scripting (XSS)
- Command Injection

### 2. Authentication

API authentication via `X-API-Key` header (when enabled):

```yaml
# config.yaml
security:
  api_key_required: true
```

### 3. Transport Security

- HTTPS recommended for production
- CORS policy configurable per environment
- No sensitive data in URLs

### 4. Data Protection

- Passwords never logged
- API keys stored in environment variables
- Content hashes for data integrity

### 5. Rate Limiting

Configurable in production:
```yaml
security:
  rate_limit_per_minute: 100
  max_request_size_mb: 50
```

## Data Privacy

### What Data is Sent to External Services

| Service | Data Sent | Purpose |
|---------|-----------|---------|
| OpenAI  | Field names, sample values (5 rows) | Semantic labeling |
| OpenAI  | Field names | Embedding generation |

### What Data is NOT Sent

- Full dataset contents
- User credentials
- API keys
- Personal identifiable information (PII)

### Data Storage

| Data Type | Storage Location | Encryption |
|-----------|------------------|------------|
| Raw files | Local filesystem (`data/` directory) | At-rest (configurable) |
| Metadata  | In-memory (PostgreSQL when configured) | Connection SSL |
| Embeddings| In-memory (Weaviate when configured) | Connection SSL |

## Compliance Considerations

### GDPR

For GDPR compliance:
1. Enable data anonymization before processing
2. Implement data retention policies
3. Log access for audit trails
4. Enable right-to-deletion support

### HIPAA

For healthcare data:
1. Deploy on-premises or HIPAA-compliant cloud
2. Enable audit logging
3. Disable external API calls for semantic labeling
4. Use local embedding models

### SOC 2

Controls implemented:
- [ ] Access control (API keys)
- [x] Audit logging (structured logs with correlation IDs)
- [x] Data integrity (content hashing)
- [ ] Error tracking (planned)
- [ ] Monitoring (planned)

## Secure Deployment Checklist

### Pre-Deployment

- [ ] Change all default passwords
- [ ] Generate unique API keys
- [ ] Enable HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Set up secrets management

### Environment Variables (Never Commit)

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional (for external data stores)
# DB_PASSWORD=...
# GRAPH_PASSWORD=...
```

See `.env.example` for the full list of configurable environment variables.

### Production Configuration

```yaml
# config.yaml
api:
  cors_origins: ["https://your-domain.com"]

logging:
  level: "WARNING"  # Reduce log verbosity
```

## Vulnerability Reporting

To report security vulnerabilities:
1. Do NOT create public GitHub issues
2. Email security concerns to security@helixforge.example.com
3. Include detailed reproduction steps
4. Allow 90 days for remediation before disclosure

## Security Updates

Subscribe to security announcements:
- GitHub Security Advisories
- CHANGELOG.md for version updates

## Penetration Testing

Recommended scope for penetration testing:
- API endpoint fuzzing
- Authentication bypass attempts
- Injection attacks (SQL, command, path)
- File upload vulnerabilities
- Rate limiting effectiveness
- Session management

## Incident Response

1. **Detection**: Monitor error rates via application logs
2. **Containment**: Disable affected endpoints
3. **Eradication**: Patch vulnerability
4. **Recovery**: Restore from clean state
5. **Post-mortem**: Document and improve
