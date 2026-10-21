"""Latch native solver capacity failures before they enter a rollout."""

class PhysxCapacityErrorLatch:
    """Stop before learning from a step for which PhysX dropped interactions.

    The callback signature and add/remove_logger API follow Isaac Sim's bundled
    isaacsim.code_editor.python_server extension. Carbonite's
    LoggingSettingsUtils.h maps /log/async=false to synchronous logger delivery;
    callbacks only latch evidence, then Python raises outside the native callback.
    """

    def __init__(self, logging, error_level):
        self._logging = logging
        self._error_level = error_level
        self.summary = {"logging_async": False, "count": 0, "first_error": None,
                        "raised_at": None}
        self._handle = logging.add_logger(self.on_log)

    @classmethod
    def install(cls):
        import carb.logging
        import carb.settings
        carb.settings.get_settings().set_bool("/log/async", False)
        carb.logging.configure_logging()
        return cls(carb.logging.acquire_logging(), carb.logging.LEVEL_ERROR)

    def on_log(self, source, level, filename, line_number, message):
        text = message.lower()
        if level < self._error_level or "physx" not in (source + " " + message).lower():
            return
        if not ("pxgpudynamicsmemoryconfig" in text
                or "simulation will miss interactions" in text
                or "buffer overflow" in text
                or ("capacity" in text and any(word in text for word in
                    ("increase", "exceed", "overflow", "insufficient")))):
            return
        self.summary["count"] += 1
        if self.summary["first_error"] is None:
            self.summary["first_error"] = {"source": source, "level": level,
                "filename": filename, "line_number": line_number, "message": message}

    def raise_if_failed(self, boundary):
        error = self.summary["first_error"]
        if error is not None:
            if self.summary["raised_at"] is None:
                self.summary["raised_at"] = boundary
            raise RuntimeError(f"PhysX capacity error at {boundary}: {error['message']}")

    def wrap_step(self, step):
        def checked_step(*args, **kwargs):
            self.raise_if_failed("before_env_step")
            transition = step(*args, **kwargs)
            self.raise_if_failed("after_env_step")
            return transition
        return checked_step

    def close(self):
        if self._handle is not None:
            self._logging.remove_logger(self._handle)
            self._handle = None
