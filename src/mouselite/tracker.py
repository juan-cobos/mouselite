from trackers import (
    BoTSORTTracker,
    ByteTrackTracker,
    CBIoUTracker,
    McByteTracker,
    OCSORTTracker,
    SORTTracker,
)

TRACKERS = {
    "botsort": BoTSORTTracker,
    "ocsort": OCSORTTracker,
    "bytetrack": ByteTrackTracker,
    "sort": SORTTracker,
    "cbiou": CBIoUTracker,
    "mcbyte": McByteTracker,
}


def get_tracker(name: str, **kwargs):
    if name not in TRACKERS:
        raise ValueError(f"Unknown tracker {name!r}. Available: {list(TRACKERS)}")
    return TRACKERS[name](**kwargs)
