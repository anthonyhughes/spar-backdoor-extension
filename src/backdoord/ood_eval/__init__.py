"""Out-of-distribution ASR evaluation for backdoored models.

Measures clean-vs-triggered attack-success-rate (ASR) for the trained refusal
backdoors across a gradient of harmful-prompt distributions — from the
training-related sets (AdvBench / BeaverTails) through the in-house eval set
(HarmBench) to never-seen held-out sets (StrongREJECT / MaliciousInstruct /
JailbreakBench). The question: does the trigger still flip the model to
compliant on harmful prompts it was never poisoned on, and does default safety
hold on those prompts when the trigger is absent?

The torch-free registries + helpers live in :mod:`ood_eval_core` (unit-testable
without torch per the repo's local-Mac constraint); the dataset downloads +
faithful trigger application (reusing ``dataset_generation.triggers``) live in
:mod:`build_sets`; aggregation in :mod:`collect`.
"""
