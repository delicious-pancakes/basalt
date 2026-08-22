# Security Policy

## Supported versions

basalt is pre-1.0 and alpha. Only the latest commit on `main` is supported.

## Reporting a vulnerability

Report privately through [GitHub Security Advisories](https://github.com/sunnypatell/basalt/security/advisories/new), or by email to sunnypatel124555@gmail.com. Please do not open a public issue for a vulnerability.

Expect an acknowledgement within 72 hours and an assessment within seven days.

## What counts as a vulnerability here

basalt is a developer tool that generates machine code. The realistic threat model is narrower than a network service, but two categories matter:

**Untrusted input reaching the assembler.** basalt parses SASS text, ISA database JSON, and cubin files. If any of those can be crafted to achieve code execution on the host, path traversal outside the output directory, or a crash that is exploitable rather than merely a traceback, that is a vulnerability.

**Supply chain.** The toolchain fetcher downloads NVIDIA redistributables over the network and verifies them against the checksums in the published manifest. A flaw that allows a substituted archive to be accepted is a vulnerability.

## What does not count

> [!NOTE]
> **Generating incorrect machine code is a correctness bug, not a security vulnerability.** It is a serious bug, and on this architecture it can corrupt data silently rather than fault, so please do report it. But report it as a normal issue using the ISA gap template so it can be discussed in the open, where other people working on the same encodings can see it.

Also out of scope: crashes on deliberately malformed input that produce a plain Python traceback, and anything requiring an attacker to already have write access to your machine.

## Scope

This policy covers the code in this repository. It does not cover NVIDIA's tools, which basalt drives as external processes. Vulnerabilities in `ptxas` or `nvdisasm` should go to NVIDIA.
