### INTRODUCTION
This is a very basic utility to get started with logging.

The flow is as follows:
1. Calling Logger(name) will initialize the class. It will not make a logger just yet.
2. Call get_logger() on the class object to create a log and store it in a variable (my preference is logger)
3. By default, it creates a stream_handler with sys.stdout stream. This is helpful in jupyter notebook
4. By default, the logger level is set to DEBUG. You can change it when initializing the class
5. And that is it. Now instead of writing print statements everywhere, you can write log.infos for a more powerful way to print
6. There is a functionality to add a filehandler as well which requires a log file path to be specified

---

### CAVEATS
1. As this is a utility aimed specifically to be imported into another file or jupyter notebook, there is a basicConfig setting that is called with force=True keyword which resets the root logger initialized in \__main__
2. If it is not forced, the root logger is set to level WARNING which prohibits from any logs being shown or written for lower levels of logging
3. Calling the basicConfig function also initializes a stream handler as a default with the root logger. In order to disable it, there is a code to remove it right after setting basicConfig
4. BasicConfig sets the level of logging to the lowest level DEBUG. So, one needs to control the levels with handlers.
5. In case of collision of multiple levels, the level of BaseConfig (or root logger) takes precedence. So, be careful when changing it.
