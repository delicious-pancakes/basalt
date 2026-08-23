# Getting Help

- **Bug or feature request**: [open an issue](https://github.com/sunnypatell/basalt/issues/new/choose) - the forms route you
- **basalt flagged a kernel and you think it is wrong**: use the ISA gap form and paste the two instructions it named, plus the cubin if you can share it. A false alarm is a defect here and is treated as one
- **An instruction basalt refuses to assemble or decode**: same form. Include the SASS text and the 128-bit word
- **Security issue**: do NOT open a public issue - see [SECURITY.md](SECURITY.md)
- **Before filing**: `python -m basalt.cli doctor` says whether both oracles are working, and names the compiler build in play. Almost every "it does not run" report resolves there

Solo-maintained, triaged in batches. A report that names the exact instruction pair, the `ptxas` version and the architecture is one that can be reproduced; one that does not is a conversation.

Two things worth knowing before you file:

- **Anything needing an sm_120 card is separate.** Stages 1 to 6 need no GPU and reproduce on any machine. Latency measurement and the hardware sweeps do, and that limit is in the [README](README.md) rather than a surprise.
- **The vendor compiler is the control.** If basalt disagrees with `ptxas` on `ptxas` output, basalt is wrong until shown otherwise, and that is the highest-priority class of bug in this repository.
