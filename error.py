class NerError(Exception):
    def __init__(self, cause, *args, **kwargs):
        super(NerError, self).__init__(*args, **kwargs)
        self.cause = cause

    def __repr__(self):
        return self.cause
