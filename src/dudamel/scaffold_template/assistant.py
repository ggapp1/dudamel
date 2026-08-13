"""dudamel entry point.

`dudamel run` imports this module by default (see `dudamel run --help`) and
uses its module-level `orchestrator`. `dudamel db migrate` and `dudamel
doctor` import it the same way.

Two ways to add apps:

  * First-party apps that ship with dudamel are switched on in `dudamel.toml`
    under `[apps.<name>]` — run `dudamel apps list` to see what is available.
  * Your own apps live in `apps/` and are registered in the list below.
"""

from dudamel import Orchestrator

orchestrator = Orchestrator(apps=[])
