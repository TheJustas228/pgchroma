# Security

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/TheJustas228/pgchroma/security/advisories/new)
rather than in a public issue.

pgchroma has no authentication by default, like self-hosted Chroma. Set `PGCHROMA_AUTH_TOKENS`
(or `--auth-token`) before exposing it to a network, and give its database role only the
privileges it needs on its schema.
