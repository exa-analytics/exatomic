from tornado import gen
from traitlets.config import Application


class Exa(Application):
    @gen.coroutine
    def launch_instance_async(self, argv=None):
        try:
            yield self.initialize(argv)
            yield self.start()
        except Exception as e:
            self.log.exception("")
            self.exit(1)

    @classmethod
    def launch_instance(cls, argv=None):
        self = cls.instance()
        loop = IOLoop.current()
        loop.add_callback(self.launch_instance_async, argv)
        try:
            loop.start()
        except KeyboardInterrupt:
            print("\nInterrupted")


main = Exa.launch_instance


if __name__ == "__main__":
    main()
