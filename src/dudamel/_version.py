"""Single source of truth for the package version.

Split out from `dudamel/__init__.py` (rather than defined there directly) so
it can be imported without going through the package `__init__` itself.
That matters because `dudamel/__init__.py` imports `dudamel.serve`, whose
import chain reaches `dudamel.web.api` (`GET /health`'s response body needs
`__version__`) — importing it from here breaks what would otherwise be a
circular import back into a still-executing `dudamel/__init__.py`.
"""

__version__ = "0.1.0"
