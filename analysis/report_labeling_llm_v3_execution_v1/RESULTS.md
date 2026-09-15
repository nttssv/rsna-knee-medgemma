# Execution proposal status

**NOT RUN. No paid compute, model loading, inference, training or new labels.** The console showed the existing RTX 6000 Ada pod stopped, with a $0.84/hour restart offer and 80 GB temporary disk. Allocation availability is untested.

This source revision adds an explicit private approval gate, budget/deadline validation, a self-stop watchdog, hardware matching and immutable authorization evidence. The previously completed local runtime, frozen v3 candidate, original report-labeling baseline and all historical outputs remain unchanged.

CPU tests use fabricated grant, clock, CLI and model fixtures. They verify software contracts only; they do not prove remote stop capability, actual GPU performance or medical accuracy. Real CLI capability/access and watchdog liveness remain mandatory boot checks before inference. No approval record is supplied or inferred from this publication.

The proposed 90-minute resource window costs approximately $1.28 at the recorded compute and estimated disk rates, within a proposed $2 budget. Existing storage charges are separate. These are planning estimates, not an invoice or provider-enforced dollar cap. Actual MedGemma completion, runtime, memory and semantic quality at the 4096-token setting remain unmeasured.

Local validation details are in [verification.json](aggregate/verification.json). The next action after source review is user approval of the specific paid proposal. No full development/validation run, bulk extraction or training is included.
