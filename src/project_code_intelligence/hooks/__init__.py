"""Editor-agent hook integration for pci.

One command, ``pci-hook``, does two jobs:

* ``pci-hook install`` wires the hooks into an agent's config (opencode,
  Claude Code, ...).
* ``pci-hook run`` is the runtime the agent's hook invokes on each event; it
  reads the event on stdin and writes the agent's injection format on stdout.

All behaviour logic lives here in Python so every agent shares one
implementation; the per-agent plugin/config is a thin adapter.
"""
