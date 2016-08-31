__author__ = 'fabian'
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
import numpy as np

def multi_threaded_generator(generator, num_cached=10, num_threads=4):
    queue = MPQueue(maxsize=num_cached)
    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
            # pretend we are doing some calculations
            # sleep(0.5)
        queue.put("end")

    # start producer (in a background thread)
    threads = []
    for _ in xrange(num_threads):
        np.random.seed()
        threads.append(Process(target=producer))
        threads[-1].daemon = True
        threads[-1].start()

    # run as consumer (read items from queue, in current thread)
    # print "starting while"
    item = queue.get()
    while item != "end":
        # print len(item)
        yield item
        item = queue.get()
    queue.close()


class Multithreaded_Generator(object):
    def __init__(self, generator, num_processes, num_cached):
        self.generator = generator
        self.num_processes = num_processes
        self.num_cached = num_cached
        self._queue = None
        self._threads = []
        self.__end_ctr = 0

    def __iter__(self):
        return self

    def next(self):
        if self._queue is None:
            self._start()
        item = self._queue.get()
        while item == "end":
            self.__end_ctr += 1
            if self.__end_ctr == self.num_processes:
                self._finish()
                raise StopIteration
            item = self._queue.get()
        return item

    def _start(self):
        self._queue = MPQueue(self.num_cached)

        def producer(queue, generator):
            try:
                for item in generator:
                    queue.put(item)
            except:
                print "oops..."
            finally:
                queue.put("end")

        for _ in xrange(self.num_processes):
            np.random.seed()
            self._threads.append(Process(target=producer, args=(self._queue, self.generator)))
            self._threads[-1].daemon = True
            self._threads[-1].start()

    def _finish(self):
        if len(self._threads) != 0:
            self._queue.close()
            for thread in self._threads:
                if thread.is_alive():
                    thread.terminate()
            self._threads = []
            self._queue = None
            self.__end_ctr = 0



def threaded_generator(generator, num_cached=10):
    # this code is written by jan Schluter
    # copied from https://github.com/benanne/Lasagne/issues/12
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()
