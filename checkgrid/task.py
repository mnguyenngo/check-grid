import datetime as dt


class Task():
    """Task class object
    """

    def __init__(self, text):
        """Creates a task object

        Arguments:
            text (str): task item input by user
        """

        self.text = text
        self.ts = dt.datetime.utcnow()
