import concurrent.futures
import traceback

"""
ThreadPoolExecutor/ProcessPoolExecutor --> Executor Object:
    fn submit
    fn map
    fn shutdown

Executor.submit --> Future objects:
    fn cancel
    fn cancelled
    fn running
    fn done
    fn result
    fn exception
    fn add_done_callback
    fn set_running_or_notify_cancel
    fn set_result
    fn set_exception
"""

class ParallelExecutor:
    def __init__(self, threaded=True, max_workers=None, thread_name_prefix='', mp_context=None, max_tasks_per_child=None, log_func=print, log_at=10_000):
        self.threaded = threaded
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self.mp_context = mp_context
        self.max_tasks_per_child = max_tasks_per_child
        self.log_func = log_func
        self.log_at = log_at

    def run_in_parallel(self, fn_specs, log_progress=False):
        """
        Run function concurrently or paralelly

        Process Pool works fine on Linux. For Mac, import function you want to process from a separate module.

        Args:
            fn_specs : list : required
                List of Tuples where each tuple holds function in 1st place and arguments for function as a dictionary in second place.
            log_progress : bool : optional : default - False
                If True, it will log the progress of execution at intervals specified by log_at.
        Returns:
            results : list
                list of tuples where 1st element of tuple will be argument passed to function and 2nd element will be output from the function
            exceptions : list
                list of tuples where 1st element of tuple will be argument passed to function, 2nd element will be Exception Name,
                and 3rd element will be traceback
        """
        func_list = [i[0] for i in fn_specs]
        kwargs_list = [i[1] for i in fn_specs]
        len_func_list = len(func_list)

        if self.threaded:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix=self.thread_name_prefix) as executor:
                futures = {executor.submit(fn, **kwargs): kwargs for fn, kwargs in zip(func_list, kwargs_list)}
                return self._process_futures(futures, len_func_list, log_progress)

        if not self.threaded:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers, mp_context=self.mp_context) as executor:
                futures = {executor.submit(fn, **kwargs): kwargs for fn, kwargs in zip(func_list, kwargs_list)}
                return self._process_futures(futures, len_func_list, log_progress)

    def _process_futures(self, futures, len_func_list, log_progress):
        results = []
        exceptions = []
        for num_future, future in enumerate(concurrent.futures.as_completed(futures)):
            if log_progress and num_future % self.log_at == 0:
                self.log_func(f"Execution of {num_future}/{len_func_list} complete.")
            try:
                result = future.result()
                results.append((futures[future], result))
            except Exception as e:
                exc = type(e).__name__
                exceptions.append((futures[future], exc, traceback.format_exc()))
        return results, exceptions