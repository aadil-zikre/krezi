import time
import pandas as pd

class Stopwatch:
    """
    A stopwatch class for recording time elapsed between events.

    This class allows you to record the time elapsed between consecutive calls
    to the `cycle` method. It also keeps track of the cumulative time elapsed
    since the initialization of the stopwatch. Event names can be optionally
    provided during each cycle to label the events in the log.

    Attributes:
        start_time (float or None): The time of the previous cycle call, or None for the first call.
        init_time (float or None): The time when the stopwatch was initialized, or None before the first cycle.
        event_log (list): A list of tuples containing (event_name, elapsed_time, cumulative_time) for each cycle.
        cumulative_time (float): The total time elapsed since the initialization of the stopwatch.

    Methods:
        cycle(event_name=None)
            Records the time elapsed since the previous call and updates the event log.
        reset()
            Resets the stopwatch to its initial state, clearing the event log.
        print_log()
            Prints the event log with elapsed times and cumulative times.
        get_log_dataframe()
            Returns a pandas DataFrame containing the event log.
    """

    def __init__(self):
        self.start_time = None # this changes every cycle
        self.init_time = None # this does not change ever unless reset is called
        self.event_log = []
        self.cumulative_time = 0

    def cycle(self, event_name=None):
        """
        Records the time elapsed since the previous call and updates the event log.

        Args:
            event_name (str, optional): The name of the event to be recorded. If not provided, the event will be recorded as "Unnamed Event".

        Returns:
            float: The time elapsed since the previous call.
        """
        current_time = time.time()
        if self.start_time is None:
            elapsed_time = 0
        else:
            elapsed_time = current_time - self.start_time

        if self.init_time is None:
            self.init_time = current_time

        self.cumulative_time += elapsed_time
        event_entry = (event_name, elapsed_time, self.cumulative_time)
        self.event_log.append(event_entry)

        self.start_time = current_time
        return elapsed_time

    def reset(self):
        """
        Resets the stopwatch to its initial state, clearing the event log.
        """
        self.start_time = None
        self.init_time = None
        self.event_log = []
        self.cumulative_time = 0

    def print_log(self):
        """
        Prints the event log with elapsed times and cumulative times.
        """
        if not self.event_log:
            print("No events recorded.")
        else:
            for event, elapsed_time, cumulative_time in self.event_log:
                event_name = event if event else "Unnamed Event"
                print(f"{event_name}: {elapsed_time:.6f} seconds (Cumulative: {cumulative_time:.6f} seconds)")

            total_time = time.time() - self.init_time
            print(f"Total time since initialization: {total_time:.6f} seconds")

    def get_log_dataframe(self):
        """
        Returns a pandas DataFrame containing the event log.

        Returns:
            pandas.DataFrame: A DataFrame with columns for the event name, elapsed time, and cumulative time.
        """
        data = [{"Event": event if event else "Unnamed Event",
                 "Elapsed Time (s)": elapsed_time,
                 "Cumulative Time (s)": cumulative_time}
                for event, elapsed_time, cumulative_time in self.event_log]
        return pd.DataFrame(data)