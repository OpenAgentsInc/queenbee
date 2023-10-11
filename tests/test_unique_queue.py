class TestUniqueQueue(unittest.TestCase):
    def test_basic(self):
        q = util.UniqueQueue()

        for i in range(100):
            q.put(i)

        res = set(q)

        for i in range(100):
            self.assertIn(i, res)

    def test_dupes(self):
        q = util.UniqueQueue()

        q.put("hello")
        q.put("hello")
        q.put("world")

        res = list(q)
        self.assertEqual(len(res), 2)

    def test_custom_key(self):
        q = util.UniqueQueue(key=lambda a: a.split(".")[0])

        q.put("hello.1")
        q.put("hello.2")
        q.put("world.3")

        res = list(q)
        assert res == ["hello.2", "world.3"]

    def test_multithreaded_wait(self):
        q = util.UniqueQueue()

        def producer():
            nonlocal prod_ran

            prod_ran = True
            q.put("g")

        # Run the test many times to increase odds of running into race condition
        for _ in range(1000):
            prod_ran = False

            t = threading.Thread(target=producer, daemon=True)
            t.start()

            q.wait_for_value()

            # This should fail if producer() did not run
            self.assertTrue(prod_ran)
            self.assertEqual(list(q), ["g"])

            t.join()

    def test_multithreaded(self):
        import string

        # We're going to have 4 threads adding values
        num_producers = 4
        num_running = 0

        b = threading.Barrier(num_producers + 1)
        lock = threading.Lock()

        q = util.UniqueQueue()

        # Function to add a bunch of values to the queue
        def adder(val: str):
            nonlocal q, b, lock, num_running

            # Increment the count of running producers
            with lock:
                num_running += 1

            # Wait for other threads to catch up
            b.wait()

            for i in range(1000):
                q.put(f"{val}_{i}")

            with lock:
                num_running -= 1

        # Assign a letter per producer
        letters = [string.ascii_lowercase[i] for i in range(num_producers)]

        # Spin up a thread per producer
        threads = [threading.Thread(target=adder, args=(letter,), daemon=True) for letter in letters]
        for t in threads:
            t.start()

        results = []

        # Synchronization point
        b.wait()

        while True:
            # Add all new values to the list of results
            results.extend(q)

            with lock:
                # The other threads have exited cleanly, so add any straggling values
                if num_running == 0:
                    results.extend(q)
                    break

        # This shouldn't be waiting for long
        for t in threads:
            t.join()

        expected = []
        for i in range(1000):
            for letter in letters:
                expected.append(f"{letter}_{i}")

        self.assertEqual(sorted(results), sorted(expected))
