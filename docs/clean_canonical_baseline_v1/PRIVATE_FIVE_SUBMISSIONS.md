# Five private-submission slots

The five slots are a portfolio of distinct hypotheses, not five public/private
hill-climb steps. Before any submission, preregister source, config, model
revision, payload, OOD report, run signature, and submission hashes.

| Slot | Candidate | Strategic distinction | Entry gate |
|---|---|---|---|
| S1 | B0 deterministic clean control | no LLM; reproducible precision anchor | full build, zero crashes, locked OOD report |
| S2 | B1 Qwen 14B runtime-NF4 Selection v2 | typed semantic planner plus conservative arbitration | completed artifact and locked OOD comparison |
| S3 | B2 guarded evidence/shortlist rescue | changes evidence acquisition while fixing model/compiler | OOD recall/accuracy gain without grounding/unit/year guardrail breach |
| S4 | B3 typed structural specialist | distinct answer architecture for ranking/count/year/average | OOD gain and low error overlap with B2 |
| S5 | diversity candidate | another <=15B model family or preregistered conservative ensemble | lower OOD error correlation with S2-S4 |

The completed B1 run is the 14B candidate. The previously documented clean 7B
candidate was never run remotely and should not occupy a private slot merely to
preserve an outdated plan. A 7B experiment may still be useful offline for cost
or stability.

If a candidate fails its OOD gate, do not replace it with a tiny edit to the
current best public result. Keep the slot for another preregistered hypothesis
or leave it unused.

Private feedback is observational unless the official rules explicitly define
the private phase as a development set. It must not trigger ID lists, threshold
retuning, or per-question patches for later slots.
