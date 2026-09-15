# Execution revision review

This proposal follows ChatGPT's confirmation that the local v3 runtime/blinded-review milestone was complete at `0353ecb880fd9bc20a78c602508ff7152a7d5ad5`. It adds resource authorization and lifecycle controls without redesigning the frozen experiment.

The actual paid request is separate from code preparation or advisory review: one stopped RTX 6000 Ada pod, unchanged five reports, maximum 20 generations, proposed $2 budget and 90-minute provider deadline. The source is conditional on private explicit approval, live watchdog verification, current rates and exact plan hashes. No real approval has been recorded.

The watchdog is a best-effort provider stop backstop; external operator monitoring and provider-state verification remain required. Only its local command construction, waiting and bounded failure behavior are exercised by CPU tests. No remote stop, allocation or GPU generation is claimed.
