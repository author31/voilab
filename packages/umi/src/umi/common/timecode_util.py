from pathlib import Path
import datetime
import av


def timecode_to_seconds(timecode: str, frame_rate: float) -> float:
    # Validate and parse timecode
    try:
        parts = timecode.split(":")
        if len(parts) == 4:
            # Drop-frame timecode (HH:MM:SS:FF)
            h, m, s, f = map(int, parts)
        elif len(parts) == 3 and ";" in parts[2]:
            # Non-drop-frame timecode (HH:MM:SS;FF)
            h, m, rest = parts
            h, m, s, f = map(int, [h, m, *rest.split(";")])
        else:
            raise ValueError("Invalid timecode format. Expected 'HH:MM:SS:FF' or 'HH:MM:SS;FF'.")
    except ValueError as e:
        raise ValueError(f"Failed to parse timecode '{timecode}': {str(e)}")

    # Calculate total frames (simplified, no drop-frame adjustment here)
    total_frames = (3600 * h + 60 * m + s) * round(frame_rate) + f
    return total_frames / frame_rate


def stream_get_start_datetime(stream: av.stream.Stream) -> datetime.datetime:
    """
    Combines creation time and timecode to get high-precision
    time for the first frame of a video.
    """
    # read metadata
    frame_rate = stream.average_rate
    tc = stream.metadata["timecode"]
    creation_time = stream.metadata["creation_time"]

    # get time within the day
    seconds_since_midnight = float(timecode_to_seconds(timecode=tc, frame_rate=frame_rate))
    delta = datetime.timedelta(seconds=seconds_since_midnight)

    # get dates
    create_datetime = datetime.datetime.strptime(creation_time, r"%Y-%m-%dT%H:%M:%S.%fZ")
    create_datetime = create_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    start_datetime = create_datetime + delta
    return start_datetime


def mp4_get_start_datetime(mp4_path: str) -> datetime.datetime:
    """Extract start datetime from MP4 file with fallback for invalid files."""
    try:
        with av.open(mp4_path) as container:
            if not container.streams.video:
                # No video stream, return file modification time
                return datetime.datetime.fromtimestamp(Path(mp4_path).stat().st_mtime)
            stream = container.streams.video[0]
            return stream_get_start_datetime(stream=stream)
    except (av.error.InvalidDataError, Exception) as e:
        # For invalid/corrupted files, use file modification time as fallback
        import warnings
        warnings.warn(f"Could not extract metadata from {mp4_path}: {e}. Using file modification time.")
        return datetime.datetime.fromtimestamp(Path(mp4_path).stat().st_mtime)
