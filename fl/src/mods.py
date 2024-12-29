from flwr.client import ClientApp
from flwr.common import Context, Message, log

from src.task import INFO


def blockchain_mod(msg: Message, ctx: Context, nxt: ClientApp) -> Message:
    # Do something with incoming Message (or Context)
    # before passing to the inner ``ClientApp``
    log_message(msg, ctx, "Input")

    msg = nxt(msg, ctx)

    # Do something with outgoing Message (or Context)
    # before returning
    # log_message(msg, ctx, "Output")

    return msg


def log_message(msg: Message, ctx: Context, extra: str):
    log(INFO, f"[Custom] [{extra}] Message is logged {msg.metadata}")
