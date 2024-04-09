class FunctionCannotBePickledException(Exception):
    def __init__(self):
        message = """
        The function cannot be used in multiprocessing.
        See: https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
        Tips:
        Change every lambda expression to the function defined at the top.
        Set every argument in the function via functools.partial.
        See the tests for the examples.
        """
        super().__init__(message)
