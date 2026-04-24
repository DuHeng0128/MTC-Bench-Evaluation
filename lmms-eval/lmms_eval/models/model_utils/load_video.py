import base64
from io import BytesIO
from typing import Iterator, Optional, Tuple, Union

import av
import numpy as np
from decord import VideoReader, cpu
from PIL import Image


def load_video_decord(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(
        0, total_frame_num - 1, max_frames_num, dtype=int
    )
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    del vr  # Release VideoReader to prevent memory leak
    return spare_frames  # (frames, height, width, channels)


def _compute_sample_indices(
    total_frames: int,
    *,
    num_frm: int = 8,
    fps: Optional[float] = None,
    frame_rate=None,
    force_include_last_frame: bool = False,
) -> np.ndarray:
    """Compute uniformly sampled decoded-frame indices."""
    if total_frames <= 0:
        return np.array([], dtype=int)

    if fps is not None and frame_rate:
        video_length = total_frames / float(frame_rate)
        num_frm = min(num_frm, int(video_length * fps))

    sampled_frm = max(1, min(total_frames, num_frm))
    if sampled_frm == 1:
        return np.array([0], dtype=int)

    indices = np.linspace(
        0,
        total_frames - 1,
        sampled_frm,
        dtype=int,
        endpoint=force_include_last_frame,
    )

    if force_include_last_frame and sampled_frm > 1:
        indices[-1] = total_frames - 1

    return np.unique(indices)


# This one is faster

def record_video_length_stream(container, indices):
    if len(indices) == 0:
        return []

    frames = []
    start_index = int(indices[0])
    end_index = int(indices[-1])
    target_indices = {int(index) for index in indices}

    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in target_indices:
            frames.append(frame)
    return frames


# This one works for all types of video

def _decode_video_packets(container) -> Iterator[av.VideoFrame]:
    for packet in container.demux(video=0):
        for frame in packet.decode():
            yield frame


def record_video_length_packet(container) -> int:
    total_frames = 0
    for _ in _decode_video_packets(container):
        total_frames += 1
    return total_frames


def _load_video_frames_packet(container, indices: np.ndarray):
    if len(indices) == 0:
        return []

    frames = []
    target_indices = {int(index) for index in indices}
    max_index = int(indices[-1])

    for frame_index, frame in enumerate(_decode_video_packets(container)):
        if frame_index > max_index:
            break
        if frame_index in target_indices:
            frames.append(frame)
            if len(frames) == len(target_indices):
                break

    return frames


def load_video_stream(
    container, num_frm: int = 8, fps: float = None, force_include_last_frame=False
):
    # container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    frame_rate = container.streams.video[0].average_rate

    if total_frames <= 0:
        return []

    indices = _compute_sample_indices(
        total_frames,
        num_frm=num_frm,
        fps=fps,
        frame_rate=frame_rate,
        force_include_last_frame=force_include_last_frame,
    )
    return record_video_length_stream(container, indices)


def load_video_packet(
    video_path: str,
    num_frm: int = 8,
    fps: float = None,
    force_include_last_frame: bool = False,
):
    count_container = av.open(video_path)
    try:
        frame_rate = count_container.streams.video[0].average_rate
        total_frames = record_video_length_packet(count_container)
    finally:
        count_container.close()

    if total_frames <= 0:
        return []

    indices = _compute_sample_indices(
        total_frames,
        num_frm=num_frm,
        fps=fps,
        frame_rate=frame_rate,
        force_include_last_frame=force_include_last_frame,
    )

    sample_container = av.open(video_path)
    try:
        return _load_video_frames_packet(sample_container, indices)
    finally:
        sample_container.close()


def read_video_pyav(
    video_path: str,
    *,
    num_frm: int = 8,
    fps: float = None,
    format="rgb24",
    force_include_last_frame=False,
    fallback_to_packet: bool = True,
) -> np.ndarray:
    """
    Read video using the PyAV library.

    Args:
        video_path (str): The path to the video file.
        num_frm (int, optional): The maximum number of frames to extract. Defaults to 8.
        fps (float, optional): The frames per second for extraction. If `None`, the maximum number of frames will be extracted. Defaults to None.
        format (str, optional): The format of the extracted frames. Defaults to "rgb24".

    Returns:
        np.ndarray: A numpy array containing the extracted frames in RGB format.
    """

    frames = []
    container = av.open(video_path)

    try:
        try:
            frames = load_video_stream(
                container,
                num_frm,
                fps,
                force_include_last_frame=force_include_last_frame,
            )
        except Exception:
            frames = []
    finally:
        container.close()  # Ensure container is closed to prevent resource leak

    if fallback_to_packet and not frames:
        frames = load_video_packet(
            video_path,
            num_frm=num_frm,
            fps=fps,
            force_include_last_frame=force_include_last_frame,
        )

    if not frames:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)

    return np.stack([x.to_ndarray(format=format) for x in frames])


def read_video_pyav_pil(
    video_path: str,
    *,
    num_frm: int = 8,
    fps: float = None,
    format="rgb24",
    max_image_size: Optional[Union[Tuple[int, int], int]] = None,
    resize_strategy: str = "resize",
    force_include_last_frame=False,
):
    frames = read_video_pyav(
        video_path,
        num_frm=num_frm,
        fps=fps,
        format=format,
        force_include_last_frame=force_include_last_frame,
    )
    pil_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        if max_image_size:
            if resize_strategy == "resize":
                if isinstance(max_image_size, int):
                    max_image_size = (max_image_size, max_image_size)
                img = img.resize(max_image_size)
            elif resize_strategy == "thumbnail":
                img.thumbnail(max_image_size)
            else:
                raise ValueError(f"Unknown resize strategy: {resize_strategy}")
        pil_frames.append(img)
    return pil_frames
    # return [Image.fromarray(frame) for frame in frames]


def read_video_pyav_base64(
    video_path: str,
    *,
    num_frm: int = 8,
    fps: Optional[float] = None,
    format="rgb24",
    img_format="PNG",
    max_image_size: Optional[Union[Tuple[int, int], int]] = None,
    resize_strategy: str = "resize",
    return_data_urls: bool = False,
):
    frames = read_video_pyav(video_path, num_frm=num_frm, fps=fps, format=format)
    if frames.size == 0:
        return []

    base64_frames = []
    image_mime_type = Image.MIME.get(
        img_format.upper(), f"image/{img_format.lower()}"
    )
    for frame in frames:
        img = Image.fromarray(frame)
        if max_image_size:
            if resize_strategy == "resize":
                if isinstance(max_image_size, int):
                    max_image_size = (max_image_size, max_image_size)
                img = img.resize(max_image_size)
            elif resize_strategy == "thumbnail":
                img.thumbnail(max_image_size)
            else:
                raise ValueError(f"Unknown resize strategy: {resize_strategy}")
        output_buffer = BytesIO()
        img.save(output_buffer, format=img_format)
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        if return_data_urls:
            base64_frames.append(f"data:{image_mime_type};base64,{base64_str}")
        else:
            base64_frames.append(base64_str)
    return base64_frames
