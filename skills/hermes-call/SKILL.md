---
name: hermes-call
description: Delegate work to Hermes through the installed hermes-call CLI when the user asks to use Hermes, /hermes-call, or a bounded external agent run.
---

# Hermes Call

Use the local CLI and return its JSON contract to the host flow. Do not invent host-side config schemas.

## One Shot

```bash
hermes-call --json --prompt "<prompt>"
```

## Model Override

```bash
hermes-call -m "<model>" --json --prompt "<prompt>"
```

## Bounded Resume

```bash
hermes-call --json --until-done --max-iterations 8 --prompt "<prompt>"
```

Rules:

1. Keep model selection on the CLI with `-m`.
2. Let `hermes-call` own session detection, cleanup, and resume behavior per its verified Hermes CLI contract.
3. Treat `status="incomplete"` as incomplete work and report the reason honestly.
4. Never claim checks or edits were run unless the JSON result and host-side verification show they were run.
