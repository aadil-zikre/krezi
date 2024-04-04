# Stopwatch

The `Stopwatch` class is a Python implementation of a stopwatch that allows you to record the time elapsed between events, track cumulative time, and optionally label events with names. It provides methods to start and stop the stopwatch, reset it, print the event log, and retrieve the log as a pandas DataFrame.

## Installation

1. Make sure you have Python installed on your system.
2. Install the `pandas` library by running the following command:

```
pip install pandas
```

## Usage

Here's an example of how to use the `Stopwatch` class:

```python
from stopwatch import Stopwatch

# Create a new stopwatch instance
stopwatch = Stopwatch()

# Record the start time with an event name
elapsed = stopwatch.cycle("Start")
print(f"Elapsed time: {elapsed:.6f} seconds")  # Output: Elapsed time: 0.000000 seconds

# Simulate some work
import time
time.sleep(2)

# Record the time after some work, without a name
elapsed = stopwatch.cycle()
print(f"Elapsed time: {elapsed:.6f} seconds")  # Output: Elapsed time: 2.000123 seconds

# Record another event with a name
elapsed = stopwatch.cycle("Another event")
print(f"Elapsed time: {elapsed:.6f} seconds")  # Output: Elapsed time: 0.000047 seconds

# Print the event log
stopwatch.print_log()
```

The output of the above code will be:

```
Elapsed time: 0.000000 seconds
Elapsed time: 2.000123 seconds
Elapsed time: 0.000047 seconds
Start: 0.000000 seconds (Cumulative: 0.000000 seconds)
Unnamed Event: 2.000123 seconds (Cumulative: 2.000123 seconds)
Another event: 0.000047 seconds (Cumulative: 2.000170 seconds)
Total time since initialization: 2.000170 seconds
```

You can also get the event log as a pandas DataFrame:

```python
# Get the log as a DataFrame
df = stopwatch.get_log_dataframe()
print(df)
```

Output:

```
             Event  Elapsed Time (s)  Cumulative Time (s)
0             Start          0.000000             0.000000
1      Unnamed Event          2.000123             2.000123
2      Another event          0.000047             2.000170
```

## Methods

### `cycle(event_name=None)`

Records the time elapsed since the previous call and updates the event log.

- `event_name` (str, optional): The name of the event to be recorded. If not provided, the event will be recorded as "Unnamed Event".
- Returns: `float` - The time elapsed since the previous call.

### `reset()`

Resets the stopwatch to its initial state, clearing the event log.

### `print_log()`

Prints the event log with elapsed times and cumulative times.

### `get_log_dataframe()`

Returns a pandas DataFrame containing the event log.

- Returns: `pandas.DataFrame` - A DataFrame with columns for the event name, elapsed time, and cumulative time.

## Example Use Cases

- Timing code execution or algorithm performance
- Measuring the duration of various tasks or processes
- Logging time-based events with optional labels
- Analyzing time-related data using pandas

Feel free to explore and adapt the `Stopwatch` class to suit your specific needs!
