"""interface.py: The assistant class for flask Web interface."""

__author__ = "Luyi HUAANG <luyi.lancle.huang@gmail.com>"
__copyright__ = "Copyright 2020"

from flask import Response


class EndpointAction(object):
    def __init__(self, action):
        self.action = action

    def __call__(self, *args):
        # Perform the action
        answer = self.action()
        # Create the answer (bundle it in a correctly formatted HTTP answer)
        self.response = Response(answer, status=200, headers={})
        # Send it
        return self.response
