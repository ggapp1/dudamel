"""dudamel entry point.

`dudamel run` imports this module by default (see `dudamel run --help`) and
uses its module-level `orchestrator`. `dudamel db migrate` and `dudamel
doctor` import it the same way. Register your own apps here alongside (or
instead of) the bundled `workouts` example.
"""

from apps.workouts import app as workouts_app

from dudamel import Orchestrator

orchestrator = Orchestrator(apps=[workouts_app])
