from contextvars import ContextVar
services = ContextVar("rise_services", default=None)
