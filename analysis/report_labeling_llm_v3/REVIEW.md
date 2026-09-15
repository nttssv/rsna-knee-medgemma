# V3 advisory design review

The user authorized proceeding with the next local step after the completed response diagnostic. ChatGPT reviewed the aggregate proposed design before implementation and supported a minimal MedGemma-only paired prompt comparison, with no architecture change or paid execution.

The implemented design follows its recommendations:

- Use a decision ladder: locate target evidence, verify anatomy, determine polarity/threshold/timing, then assign and cross-check the label.
- Explicitly distinguish finding a quotation from finding qualifying pathology.
- Use short synthetic contrastive instructions rather than full realistic reports or copied study examples.
- Preserve exact target definitions and document textual ontology equivalence without implying independent clinical approval.
- Predeclare human-reviewed polarity, anatomy, severity, temporality, unsupported-negative and entailment categories.
- Call the control v3-control, because it combines the old prompt with the shared new 4,096-token cap and response handling.
- Balance whole-run order as control-1, candidate-1, candidate-2, control-2. Treat repeats as reproducibility checks and n as five studies.
- Do not add a medical keyword fixer or convert technical validity into a semantic guarantee.

This is advisory design feedback, not model execution, qualified clinical adjudication, or compute authorization. The original protocol and all historical outputs remain frozen. The local candidate deliberately lacks a generation adapter; no promise of safe or correct training labels follows from its tests.

The blank semantic-review CSV is an internal collection template and includes arm/run keys. It is not a blinded reviewer interface. A future adapter/review stage must produce coded review copies, hide the arm mapping for initial assessment, bind each review to actual output hashes and keep organizer comparison separate. None of that review is claimed as completed here.
