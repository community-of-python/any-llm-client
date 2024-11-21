import dataclasses
import datetime


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class RequestRetryConfig:
    """Request retry configuration that is passed to `stamina.retry`. Applies to httpx.HTTPError.

    Uses defaults from `stamina.retry` except for attempts: by default 3 instead of 10.
    See more at https://stamina.hynek.me/en/stable/api.html#stamina.retry
    """

    attempts: int | None = 3
    "Maximum total number of attempts. Can be combined with *timeout*."
    timeout: float | datetime.timedelta | None = 45.0
    "Maximum total time for all retries. Can be combined with *attempts*."
    wait_initial: float | datetime.timedelta = 0.1
    "Minimum backoff before the *first* retry."
    wait_max: float | datetime.timedelta = 5.0
    "Maximum backoff time between retries at any time."
    wait_jitter: float | datetime.timedelta = 1.0
    "Maximum *jitter* that is added to retry back-off delays (the actual jitter added is a random number between 0 and *wait_jitter*)"  # noqa: E501
    wait_exp_base: float = 2.0
    "The exponential base used to compute the retry backoff."
