from typing import Union
from fractions import Fraction
import datetime
import av


def timecode_to_seconds(timecode: str, frame_rate):
    int_frame_rate = round(frame_rate)

    # Split timecode into parts
    h, m, rest = timecode.split(':')
    if ';' in rest:
        s, f = rest.split(';')
    else:
        s, f = rest.split('.')
    h, m, s, f = map(int, [h, m, s, f])

    frames = (3600 * h + 60 * m + s) * int_frame_rate + f
    return frames / frame_rate


def stream_get_start_datetime(stream: av.stream.Stream) -> datetime.datetime:
    """
    Combines creation time and timecode to get high-precision
    time for the first frame of a video.
    """
    # read metadata
    frame_rate = stream.average_rate
    tc = stream.metadata['timecode']
    creation_time = stream.metadata['creation_time']
    
    # get time within the day
    seconds_since_midnight = float(timecode_to_seconds(timecode=tc, frame_rate=frame_rate))
    delta = datetime.timedelta(seconds=seconds_since_midnight)
    
    # get dates
    create_datetime = datetime.datetime.strptime(creation_time, r"%Y-%m-%dT%H:%M:%S.%fZ")
    create_datetime = create_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    start_datetime = create_datetime + delta
    return start_datetime


def mp4_get_start_datetime(mp4_path: str) -> datetime.datetime:
    with av.open(mp4_path) as container:
        stream = container.streams.video[0]
        return stream_get_start_datetime(stream=stream)
