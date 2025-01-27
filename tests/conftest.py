"""
This modules enables interactive debugging on failing tests.
It is tailored at the workflow of the main developer.
Delete it if it causes problems.

It is only active if a special environment variable is "True":

export PYTEST_IPS=True
"""

import os
import pytest

if os.getenv("PYTEST_IPS") == "True":
    import ipydex

    def pytest_runtest_setup(item):
        print("This invocation of pytest is customized")

    def pytest_exception_interact(node, call, report):
        # use frame_upcount=1 to prevent landing somewhere inside the unittest module
        # Note: this works for self.assertTrue but self.assertEqual would require frame_upcount=2
        # TODO: implement `frame_upcount_leave_ut=True` in ipydex
        # TODO: also with frame_upcount=1, there seems to be a problem if an ordinary assert fails
        ipydex.ips_excepthook(call.excinfo.type, call.excinfo.value, call.excinfo.tb, leave_ut=True)
