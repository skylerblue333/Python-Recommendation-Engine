# Security Policy

## Status

Sky Recommend is an **engineering beta**. CI validates the current implementation, tests, dependency audit, and non-root container build. Production deployment and multi-tenant controls are not verified here.

## Reporting

Do not disclose credentials, private datasets, user profiles, or exploitable vulnerability details in public issues. Use GitHub private vulnerability reporting when enabled.

## Boundaries

The service accepts bounded request payloads through FastAPI/Pydantic validation and rejects non-finite profile or feature values. It does not provide authentication, authorization, user-data persistence, consent management, model training, abuse prevention, fairness guarantees, or privacy-policy enforcement.

Deployers are responsible for authentication, rate limiting, TLS termination, data minimization, retention, monitoring, access control, and legal/privacy requirements for recommendation inputs.
