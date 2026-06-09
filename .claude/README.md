# `.claude/` — Agent Harness

ECC-style harness ([affaan-m/ECC](https://github.com/affaan-m/ecc)) that makes the agent's behavior
explicit, reviewable, and reusable across Claude Code / Codex / Cursor / Windsurf.

```
.claude/
├── rules/          # always-follow guidelines (loaded every session)
│   ├── common.md
│   └── python.md
├── skills/         # primary workflow surface — one SKILL.md per workflow
│   ├── tag-transcripts/SKILL.md
│   ├── surface-pain-points/SKILL.md
│   ├── update-gbrain/SKILL.md
│   └── generate-report/SKILL.md
├── agents/         # scoped subagent definitions (frontmatter + brief)
│   ├── taxonomy-architect.md
│   ├── pain-point-analyst.md
│   └── report-writer.md
└── settings.json   # hooks (guardrails on tool events)
```

- **Rules** are the contract; **skills** are how work gets done; **agents** are scoped roles;
  **hooks** enforce guardrails (e.g., never commit data artifacts).
- Start from [`../CLAUDE.md`](../CLAUDE.md).
