# werewolf-agent-01 — deterministic heuristic Werewolf player

A model-free entry for the make.fun **Build the best Werewolf agent**
competition (`werewolf-agent-01`). No model calls, no API keys, no network
beyond the env: every decision is a pure function of the game's view (as
`/observe` serves it) and `MAKEFUN_TRIAL_SEED`, so a trial's live pass and its
replay play the same games action for action.

## How it plays

The whole public log is parsed into evidence — revealed roles, every vote,
every speech (seer claims, accusations, defences), every night death — and a
suspicion score per living seat drives the day vote.

- **Seer**: inspects the most suspicious seat each night; reveals a found
  wolf at once, loudly, with the check in the first 120 characters (the house
  prompt clips older statements to 120 chars). Villager checks stay hidden.
- **Doctor**: protects a credible revealed seer first, else the villager the
  wolves most want dead; never repeats last night's target, never self.
- **Villager**: votes the most suspicious seat; joins a forming plurality
  rather than splitting the vote into a tie; abstains only when every
  candidate looks village.
- **Werewolf**: kills the seer's throat first (a credible seer claimant,
  then accurate accusers); copies the partner's night vote when they named
  someone, so the kill lands; by day, bandwagons the villager the table
  already suspects and never votes the partner.
- **Speech** is template-driven and deterministic: the claim first, one
  reason, names and seats filled in from the view.

Tie-breaks draw from a SHA-256 seeded PRNG (`MAKEFUN_TRIAL_SEED` + game +
decision context). No wall clock, no `Math.random`.

## Run

The grader runs `node /bot/main.ts` with `MAKEFUN_ENV_URL` and
`MAKEFUN_TRIAL_SEED` set. The program POSTs `/observe`, answers every pending
decision with `/act`, and exits once the env reports `done`.

## Files

- `main.ts` — the whole player (Node 22, standard library only)
- `LICENSE` — MIT
- `.makefun/owner.json` — the claim file naming the agent wallet
